import re
from typing import List, Dict
import json
import os
from datetime import datetime
from openai import OpenAI
from loguru import logger
from legacy_tool_dispatcher import get_tool_schema, execute_tool, has_tool
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models"))
from intent_predictor import predict_intent


def _clean_content(text: str) -> str:
    """清理不可编码字符（surrogates）"""
    if not text:
        return text
    return text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')


def _safe_truncate(text: str, n: int) -> str:
    """安全的截断"""
    return text[:n]


class XianyuReplyBot:
    def __init__(self):
        # 初始化OpenAI客户端
        self.client = OpenAI(
            api_key=os.getenv("API_KEY"),
            base_url=os.getenv("MODEL_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        )
        self._init_system_prompts()
        self._init_agents()
        self.router = IntentRouter(self.agents['classify'])
        self.last_intent = None  # 记录最后一次意图


    def _init_agents(self):
        """初始化各领域Agent"""
        self.agents = {
            'classify': ClassifyAgent(self.client, self.classify_prompt, self._safe_filter),
            '查询价格': PriceAgent(self.client, self.price_prompt, self._safe_filter),
            '议价砍价': TechAgent(self.client, self.tech_prompt, self._safe_filter),
            '下单预订': BookingAgent(self.client, self.booking_prompt, self._safe_filter),
            '常规咨询': DefaultAgent(self.client, self.default_prompt, self._safe_filter),
        }

    def _init_system_prompts(self):
        """初始化各Agent专用提示词，优先加载用户自定义文件，否则使用Example默认文件"""
        prompt_dir = "prompts"
        
        def load_prompt_content(name: str) -> str:
            """尝试加载提示词文件"""
            # 优先尝试加载 target.txt
            target_path = os.path.join(prompt_dir, f"{name}.txt")
            if os.path.exists(target_path):
                file_path = target_path
            else:
                # 尝试默认提示词 target_example.txt
                file_path = os.path.join(prompt_dir, f"{name}_example.txt")

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                logger.debug(f"已加载 {name} 提示词，路径: {file_path}, 长度: {len(content)} 字符")
                return content

        try:
            # 加载分类提示词
            self.classify_prompt = load_prompt_content("classify_prompt")
            # 加载价格提示词
            self.price_prompt = load_prompt_content("price_prompt")
            # 加载技术提示词
            self.tech_prompt = load_prompt_content("bargain_prompt")
            # 加载预订提示词
            self.booking_prompt = load_prompt_content("booking_prompt")
            # 加载默认提示词
            self.default_prompt = load_prompt_content("default_prompt")
                
            logger.info("成功加载所有提示词")
        except Exception as e:
            logger.error(f"加载提示词时出错: {e}")
            raise

    def _safe_filter(self, text: str) -> str:
        """安全过滤模块"""
        blocked_phrases = ["微信", "QQ", "支付宝", "银行卡", "线下"]
        return "[安全提醒]请通过平台沟通" if any(p in text for p in blocked_phrases) else text

    def format_history(self, context: List[Dict]) -> str:
        """格式化对话历史，返回完整的对话记录"""
        # 过滤掉系统消息，只保留用户和助手的对话
        user_assistant_msgs = [msg for msg in context if msg['role'] in ['user', 'assistant']]
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in user_assistant_msgs])

    def generate_reply(self, user_msg: str, item_desc: str, context: List[Dict]) -> str:
        """生成回复主流程"""
        # 记录用户消息
        # logger.debug(f'用户所发消息: {user_msg}')
        
        formatted_context = self.format_history(context)
        # logger.debug(f'对话历史: {formatted_context}')
        
        # 1. 路由决策
        detected_intent = self.router.detect(user_msg, item_desc, formatted_context)

        # 2. 获取对应Agent

        internal_intents = {'classify'}  # 定义不对外开放的Agent

        if detected_intent == 'no_reply':
            # 无需回复的情况
            logger.info(f'意图识别完成: no_reply - 无需回复')
            self.last_intent = 'no_reply'
            return "-"  # 返回特殊标记，表示无需回复
        elif detected_intent in self.agents and detected_intent not in internal_intents:
            agent = self.agents[detected_intent]
            logger.info(f'意图识别完成: {detected_intent}')
            self.last_intent = detected_intent  # 保存当前意图
        else:
            agent = self.agents['常规咨询']
            logger.info(f'意图识别完成: 常规咨询')
            self.last_intent = '常规咨询'  # 保存当前意图
        
        # 3. 获取议价次数
        bargain_count = self._extract_bargain_count(context)
        logger.info(f'议价次数: {bargain_count}')

        # 4. 生成回复
        return agent.generate(
            user_msg=user_msg,
            item_desc=item_desc,
            context=formatted_context,
            bargain_count=bargain_count
        )
    
    def _extract_bargain_count(self, context: List[Dict]) -> int:
        """
        从上下文中提取议价次数信息
        
        Args:
            context: 对话历史
            
        Returns:
            int: 议价次数，如果没有找到则返回0
        """
        # 查找系统消息中的议价次数信息
        for msg in context:
            if msg['role'] == 'system' and '议价次数' in msg['content']:
                try:
                    # 提取议价次数
                    match = re.search(r'议价次数[:：]\s*(\d+)', msg['content'])
                    if match:
                        return int(match.group(1))
                except Exception:
                    pass
        return 0

    def reload_prompts(self):
        """重新加载所有提示词"""
        logger.info("正在重新加载提示词...")
        self._init_system_prompts()
        self._init_agents()
        logger.info("提示词重新加载完成")


class IntentRouter:
    """意图路由决策器"""

    def __init__(self, classify_agent):
        self.rules = {
            'tech': {  # 技术类优先判定
                'keywords': ['参数', '规格', '型号', '连接', '对比'],
                'patterns': [
                    r'和.+比'             
                ]
            },
            'booking': {
                'keywords': ['预订', '入住', '订房', '几号', '几天', '有没有房', '还能订'],
                'patterns': [r'\d+月\d+日', r'\d+号']
            },
            'price': {
                'keywords': ['便宜', '价', '砍价', '少点'],
                'patterns': [r'\d+元', r'能少\d+']
            }
        }
        self.classify_agent = classify_agent

    def detect(self, user_msg: str, item_desc, context) -> str:
        """三级路由策略（技术优先）"""
        text_clean = re.sub(r'[^\w\u4e00-\u9fa5]', '', user_msg)
        r = predict_intent(text_clean)
        intent = r['intent']
        return intent
        # # 1. 技术类关键词优先检查
        # if any(kw in text_clean for kw in self.rules['tech']['keywords']):
        #     # logger.debug(f"技术类关键词匹配: {[kw for kw in self.rules['tech']['keywords'] if kw in text_clean]}")
        #     return 'tech'
            
        # # 2. 技术类正则优先检查
        # for pattern in self.rules['tech']['patterns']:
        #     if re.search(pattern, text_clean):
        #         # logger.debug(f"技术类正则匹配: {pattern}")
        #         return 'tech'

        # # 3. 预订类检查
        # if any(kw in text_clean for kw in self.rules['booking']['keywords']):
        #     return 'booking'
        # for pattern in self.rules['booking']['patterns']:
        #     if re.search(pattern, text_clean):
        #         return 'booking'

        # # 4. 价格类检查
        # for intent in ['price']:
        #     if any(kw in text_clean for kw in self.rules[intent]['keywords']):
        #         # logger.debug(f"价格类关键词匹配: {[kw for kw in self.rules[intent]['keywords'] if kw in text_clean]}")
        #         return intent
            
        #     for pattern in self.rules[intent]['patterns']:
        #         if re.search(pattern, text_clean):
        #             # logger.debug(f"价格类正则匹配: {pattern}")
        #             return intent
        
        # # 4. 大模型兜底
        # # logger.debug("使用大模型进行意图分类")
        # return self.classify_agent.generate(
        #     user_msg=user_msg,
        #     item_desc=item_desc,
        #     context=context
        # )


class BaseAgent:
    """Agent基类"""

    def __init__(self, client, system_prompt, safety_filter, tools: bool = False):
        self.client = client
        self.system_prompt = system_prompt
        self.safety_filter = safety_filter
        self.tools = get_tool_schema() if tools else None

    def generate(self, user_msg: str, item_desc: str, context: str, bargain_count: int = 0) -> str:
        """生成回复模板方法"""
        messages = self._build_messages(user_msg, item_desc, context)
        response_text = self._call_llm_with_tools(messages)
        return self.safety_filter(response_text)

    def _build_messages(self, user_msg: str, item_desc: str, context: str) -> List[Dict]:
        """构建消息链"""
        today = datetime.now().strftime("%Y-%m-%d")
        return [
            {"role": "system", "content": f"【当前日期】{today}\n【商品信息】{item_desc}\n【你与客户对话历史】{context}\n{self.system_prompt}"},
            {"role": "user", "content": user_msg}
        ]

    def _call_llm(self, messages: List[Dict], temperature: float = 0.4, tools=None) -> str:
        """调用大模型"""
        # 清理所有消息内容中的不可编码字符
        clean_messages = []
        for msg in messages:
            clean_msg = dict(msg)
            if "content" in clean_msg and clean_msg["content"] is not None:
                if isinstance(clean_msg["content"], str):
                    clean_msg["content"] = _clean_content(clean_msg["content"])
            clean_messages.append(clean_msg)

        kwargs = {
            "model": os.getenv("MODEL_NAME", "qwen-max"),
            "messages": clean_messages,
            "temperature": temperature,
            "max_tokens": 500,
            "top_p": 0.8
        }
        if tools:
            kwargs["tools"] = tools
        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0]

    def _call_llm_with_tools(self, messages: List[Dict], temperature: float = 0.4) -> str:
        """调用大模型，支持 tool-use 循环（最多2轮）"""
        if not self.tools:
            choice = self._call_llm(messages, temperature)
            return choice.message.content

        for _ in range(3):
            choice = self._call_llm(messages, temperature, tools=self.tools)

            if choice.message.tool_calls:
                assistant_msg = {
                    "role": "assistant",
                    "content": _clean_content(choice.message.content) if choice.message.content else None,
                    "tool_calls": []
                }
                for tool_call in choice.message.tool_calls:
                    assistant_msg["tool_calls"].append({
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        }
                    })
                messages.append(assistant_msg)

                for tool_call in choice.message.tool_calls:
                    tool_name = tool_call.function.name
                    if has_tool(tool_name):
                        try:
                            params = json.loads(tool_call.function.arguments)
                        except Exception:
                            params = {}
                        tool_result = execute_tool(tool_name, params)
                        logger.info(f"Tool调用: {tool_name} → {_safe_truncate(_clean_content(tool_result), 200)}")
                    else:
                        tool_result = f"Unknown tool: {tool_name}"

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": _clean_content(tool_result),
                    })
                continue

            return choice.message.content

        return "抱歉，处理请求时遇到一些问题，请稍后再试。"


class PriceAgent(BaseAgent):
    """议价处理Agent — 支持实时查价"""

    def __init__(self, client, system_prompt, safety_filter):
        super().__init__(client, system_prompt, safety_filter, tools=True)

    def generate(self, user_msg: str, item_desc: str, context: str, bargain_count: int=0) -> str:
        """重写生成逻辑"""
        dynamic_temp = self._calc_temperature(bargain_count)
        messages = self._build_messages(user_msg, item_desc, context)
        messages[0]['content'] += f"\n▲当前议价轮次：{bargain_count}"

        response_text = self._call_llm_with_tools(messages, dynamic_temp)
        return self.safety_filter(response_text)

    def _calc_temperature(self, bargain_count: int) -> float:
        """动态温度策略"""
        return min(0.3 + bargain_count * 0.15, 0.9)


class TechAgent(BaseAgent):
    """技术咨询Agent"""
    def generate(self, user_msg: str, item_desc: str, context: str, bargain_count: int=0) -> str:
        """重写生成逻辑"""
        messages = self._build_messages(user_msg, item_desc, context)

        response = self.client.chat.completions.create(
            model=os.getenv("MODEL_NAME", "qwen-max"),
            messages=messages,
            temperature=0.4,
            max_tokens=500,
            top_p=0.8,
            extra_body={
                "enable_search": True,
            }
        )

        return self.safety_filter(response.choices[0].message.content)


class ClassifyAgent(BaseAgent):
    """意图识别Agent"""

    def generate(self, **args) -> str:
        response = super().generate(**args)
        return response


class DefaultAgent(BaseAgent):
    """默认处理Agent"""

    def _call_llm(self, messages: List[Dict], *args) -> str:
        """限制默认回复长度"""
        return super()._call_llm(messages, temperature=0.7)


class BookingAgent(BaseAgent):
    """酒店预订Agent — 支持调用酒店查询工具"""

    def __init__(self, client, system_prompt, safety_filter):
        super().__init__(client, system_prompt, safety_filter, tools=True)