"""LangGraph 图定义 — 替代 legacy_agents 的 Agent 编排"""
import asyncio
import os
import re
from datetime import datetime
from typing import Dict

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from loguru import logger

from graph_state import XianyuState
from langgraph_tools import get_langgraph_tools

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models"))


# ── 自定义 ToolNode 包装器 ─────────────────────────────────────

_tool_node: ToolNode = None


def _init_tool_node(tools):
    global _tool_node
    _tool_node = ToolNode(tools)


async def logged_tool_node(state: XianyuState) -> dict:
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        for tc in last_msg.tool_calls:
            logger.debug(f"[ToolNode] 调用工具: {tc.get('name', '?')}, 参数: {tc.get('args', {})}")
    result = await _tool_node.ainvoke(state)
    if "messages" in result:
        for msg in result["messages"]:
            logger.debug(f"[ToolNode] 工具输出: {msg.content[:200]}")
    return result


# ── Prompt 缓存 ──────────────────────────────────────────────

_PROMPTS: Dict[str, str] = {}


def _load_prompt(name: str) -> str:
    """加载提示词，优先 {name}.txt，否则 {name}_example.txt"""
    if name in _PROMPTS:
        return _PROMPTS[name]
    prompt_dir = os.path.join(os.path.dirname(__file__), "prompts")
    target = os.path.join(prompt_dir, f"{name}.txt")
    if os.path.exists(target):
        path = target
    else:
        path = os.path.join(prompt_dir, f"{name}_example.txt")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    _PROMPTS[name] = content
    return content


# ── 安全过滤 ─────────────────────────────────────────────────

_BLOCKED = ["微信", "QQ", "支付宝", "银行卡", "线下"]


def _safe_filter(text: str) -> str:
    if not text:
        return ""
    return "请通过平台沟通" if any(p in text for p in _BLOCKED) else text


# ── 辅助函数 ─────────────────────────────────────────────────

def _format_history(context: list) -> str:
    msgs = [m for m in context if m["role"] in ("user", "assistant")]
    return "\n".join(f"{m['role']}: {m['content']}" for m in msgs)


def _extract_bargain_count(context: list) -> int:
    for msg in context:
        content = msg.get("content") or ""
        if msg.get("role") == "system" and "议价次数" in content:
            match = re.search(r"议价次数[:：]\s*(\d+)", content)
            if match:
                return int(match.group(1))
    return 0


def _build_system_msg(prompt: str, item_desc: str, formatted_context: str,
                      bargain_count: int = 0, extra: str = "") -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    parts = [f"【当前日期】{today}", f"【商品信息】{item_desc}",
             f"【你与客户对话历史】{formatted_context}", prompt]
    if extra:
        parts.append(extra)
    return "\n".join(parts)


def _build_llm(temperature: float, use_tools: bool, enable_search: bool):
    extra_body = {}
    extra_body["enable_thinking"] = False
    if enable_search:
        extra_body["enable_search"] = True
    model = ChatOpenAI(
        model=os.getenv("MODEL_NAME", "qwen-max"),
        openai_api_base=os.getenv("MODEL_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        openai_api_key=os.getenv("API_KEY"),
        temperature=temperature,
        max_tokens=500,
        top_p=0.8,
        extra_body=extra_body if extra_body else None,
    )
    if use_tools:
        model = model.bind_tools(get_langgraph_tools())
    return model


# ── 节点函数 ─────────────────────────────────────────────────

async def intent_detection_node(state: XianyuState) -> dict:
    from intent_predictor import predict_intent
    cleaned = re.sub(r"[^\w一-龥]", "", state["user_msg"])
    result = await asyncio.to_thread(predict_intent, cleaned)
    intent = result["intent"]
    bargain_count = _extract_bargain_count(state["context"])
    logger.info(f"意图识别: {intent}, 议价次数: {bargain_count}")
    return {"intent": intent, "bargain_count": bargain_count}


async def prepare_price_node(state: XianyuState) -> dict:
    prompt = _load_prompt("price_prompt")
    bc = state["bargain_count"]
    temp = min(0.3 + bc * 0.15, 0.9)
    extra = f"\n▲当前议价轮次：{bc}" if bc > 0 else ""
    sys_content = _build_system_msg(prompt, state["item_desc"],
                                    state["formatted_context"], bc, extra)
    return {
        "messages": [SystemMessage(content=sys_content),
                     HumanMessage(content=state["user_msg"])],
        "temperature": temp, "use_tools": True, "enable_search": False,
    }


async def prepare_booking_node(state: XianyuState) -> dict:
    prompt = _load_prompt("booking_prompt")
    sys_content = _build_system_msg(prompt, state["item_desc"],
                                    state["formatted_context"])
    return {
        "messages": [SystemMessage(content=sys_content),
                     HumanMessage(content=state["user_msg"])],
        "temperature": 0.4, "use_tools": True, "enable_search": False,
    }


async def prepare_bargain_node(state: XianyuState) -> dict:
    prompt = _load_prompt("bargain_prompt")
    sys_content = _build_system_msg(prompt, state["item_desc"],
                                    state["formatted_context"])
    return {
        "messages": [SystemMessage(content=sys_content),
                     HumanMessage(content=state["user_msg"])],
        "temperature": 0.4, "use_tools": False, "enable_search": True,
    }


async def prepare_consult_node(state: XianyuState) -> dict:
    prompt = _load_prompt("default_prompt")
    sys_content = _build_system_msg(prompt, state["item_desc"],
                                    state["formatted_context"])
    return {
        "messages": [SystemMessage(content=sys_content),
                     HumanMessage(content=state["user_msg"])],
        "temperature": 0.7, "use_tools": False, "enable_search": False,
    }


async def llm_call_node(state: XianyuState) -> dict:
    model = _build_llm(state["temperature"], state["use_tools"],
                       state["enable_search"])
    response = await model.ainvoke(state["messages"])
    tool_loop_count = state.get("tool_loop_count", 0) + 1

    if hasattr(response, "tool_calls") and response.tool_calls:
        logger.debug(f"LLM 请求工具调用 (第{tool_loop_count}轮): {response.tool_calls}")
        return {"messages": [response], "tool_loop_count": tool_loop_count}

    content = _safe_filter(response.content or "")
    return {"response": content, "messages": [response], "tool_loop_count": tool_loop_count}


async def safety_filter_node(state: XianyuState) -> dict:
    return {"response": _safe_filter(state.get("response") or "工具调用达到上限，请补充更明确的酒店和日期后重试。")}


async def no_reply_node(state: XianyuState) -> dict:
    return {"response": "-"}


# ── 路由函数 ─────────────────────────────────────────────────

def route_intent(state: XianyuState) -> str:
    intent = state["intent"]
    if intent == "no_reply":
        return "no_reply"
    if intent == "查询价格":
        return "prepare_price"
    if intent == "下单预订":
        return "prepare_booking"
    if intent == "议价砍价":
        return "prepare_bargain"
    return "prepare_consult"


def should_call_tools(state: XianyuState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        if state.get("tool_loop_count", 0) >= 3:
            logger.warning("工具调用已达 3 轮上限，强制结束")
            return "end"
        return "tools"
    return "end"


# ── 构建图 ───────────────────────────────────────────────────

def _build_graph() -> StateGraph:
    _init_tool_node(get_langgraph_tools())

    g = StateGraph(XianyuState)

    g.add_node("intent_detection", intent_detection_node)
    g.add_node("prepare_price", prepare_price_node)
    g.add_node("prepare_booking", prepare_booking_node)
    g.add_node("prepare_bargain", prepare_bargain_node)
    g.add_node("prepare_consult", prepare_consult_node)
    g.add_node("llm_call", llm_call_node)
    g.add_node("tools", logged_tool_node)
    g.add_node("safety_filter", safety_filter_node)
    g.add_node("no_reply", no_reply_node)

    g.add_edge(START, "intent_detection")
    g.add_conditional_edges("intent_detection", route_intent, {
        "no_reply": "no_reply",
        "prepare_price": "prepare_price",
        "prepare_booking": "prepare_booking",
        "prepare_bargain": "prepare_bargain",
        "prepare_consult": "prepare_consult",
    })

    g.add_edge("prepare_price", "llm_call")
    g.add_edge("prepare_booking", "llm_call")
    g.add_edge("prepare_bargain", "llm_call")
    g.add_edge("prepare_consult", "llm_call")

    g.add_conditional_edges("llm_call", should_call_tools, {
        "tools": "tools",
        "end": "safety_filter",
    })
    g.add_edge("tools", "llm_call")

    g.add_edge("safety_filter", END)
    g.add_edge("no_reply", END)

    return g


xianyu_graph = _build_graph().compile()
