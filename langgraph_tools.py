"""LangChain 工具包装 — 桥接现有 hotel_tools 实现"""
import asyncio
from langchain_core.tools import tool


@tool
def search_hotel_price(hotel_name: str, check_in: str, check_out: str) -> str:
    """根据酒店名搜索华住会酒店并获取实时房型价格。

    当用户询问酒店价格、问'多少钱'、问'最新报价'、提及酒店名称时使用。
    需要提供酒店名称、入住日期和离店日期。日期格式为 YYYY-MM-DD。

    Args:
        hotel_name: 酒店名称，如'全季上海虹桥火车站酒店'
        check_in: 入住日期，格式 YYYY-MM-DD
        check_out: 离店日期，格式 YYYY-MM-DD
    """
    from utils.hotel_tools import search_hotel_price as _search_hotel_price
    return _search_hotel_price(hotel_name=hotel_name, check_in=check_in, check_out=check_out)


@tool
def crawl_hotels(city: str, check_in: str, check_out: str) -> str:
    """批量爬取华住会指定城市全部酒店数据并保存到本地数据库。

    当本地酒店数据为空、数据过期、或需要更新酒店信息时调用。

    Args:
        city: 城市名称，如'上海'、'北京'
        check_in: 入住日期，格式 YYYY-MM-DD
        check_out: 离店日期，格式 YYYY-MM-DD
    """
    from utils.hotel_tools import crawl_hotels as _crawl_hotels
    return _crawl_hotels(city=city, check_in=check_in, check_out=check_out)


_tools = [search_hotel_price, crawl_hotels]


def get_langgraph_tools() -> list:
    """返回 LangChain 工具列表，供 ToolNode 使用"""
    return _tools
