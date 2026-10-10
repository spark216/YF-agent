"""
Tool 定义 + 分发器
"""
from typing import Any, Dict

MAX_RESULT_CHARS = 50000

tool_definitions = [
    {
        "name": "search_hotel_price",
        "description": (
            "根据酒店名搜索华住会酒店并获取实时房型价格。"
            "当用户询问酒店价格、问'多少钱'、问'最新报价'、提及酒店名称时使用。"
            "需要提供酒店名称、入住日期和离店日期。"
            "日期未提供时从对话上下文中推断。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "hotel_name": {
                    "type": "string",
                    "description": "酒店名称，如'全季上海虹桥火车站酒店'",
                },
                "check_in": {
                    "type": "string",
                    "description": "入住日期，格式 YYYY-MM-DD",
                },
                "check_out": {
                    "type": "string",
                    "description": "离店日期，格式 YYYY-MM-DD",
                },
            },
            "required": ["hotel_name", "check_in", "check_out"],
        },
    },
    {
        "name": "crawl_hotels",
        "description": (
            "批量爬取华住会指定城市全部酒店数据并保存到本地数据库。"
            "当本地酒店数据为空、数据过期、或需要更新酒店信息时调用。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名称，如'上海'、'北京'",
                },
                "check_in": {
                    "type": "string",
                    "description": "入住日期，格式 YYYY-MM-DD",
                },
                "check_out": {
                    "type": "string",
                    "description": "离店日期，格式 YYYY-MM-DD",
                },
            },
            "required": ["city", "check_in", "check_out"],
        },
    },
]

_tool_names = {t["name"] for t in tool_definitions}


def get_tool_schema() -> list:
    """返回 tool schema 供 LLM API 调用"""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
            },
        }
        for t in tool_definitions
    ]


def has_tool(name: str) -> bool:
    return name in _tool_names


def execute_tool(name: str, params: Dict[str, Any]) -> str:
    """工具执行分发器"""
    handlers = {
        "search_hotel_price": _search_hotel_price,
        "crawl_hotels": _crawl_hotels,
    }

    handler = handlers.get(name)
    if not handler:
        return f"Unknown tool: {name}"

    try:
        result = handler(params)
        return _truncate_result(result)
    except Exception as e:
        return f"Error executing {name}: {e}"


def _search_hotel_price(params: Dict) -> str:
    from utils.hotel_tools import search_hotel_price
    return search_hotel_price(
        hotel_name=params["hotel_name"],
        check_in=params["check_in"],
        check_out=params["check_out"],
    )


def _crawl_hotels(params: Dict) -> str:
    from utils.hotel_tools import crawl_hotels
    return crawl_hotels(
        city=params["city"],
        check_in=params["check_in"],
        check_out=params["check_out"],
    )


def _truncate_result(result: str) -> str:
    if len(result) <= MAX_RESULT_CHARS:
        return result
    keep_each = (MAX_RESULT_CHARS - 60) // 2
    return result[:keep_each] + f"\n\n[... truncated {len(result) - keep_each * 2} chars ...]\n\n" + result[-keep_each:]
