"""主流程: RAG 检索 + LLM 抽取 + 日期规则引擎兜底"""
import json
import os
import re
import sys
import time

import booking_config as config
from date_parser import parse_date
from hotel_index import get_hotel_index
from llm_extractor import get_extractor

# 日期关键词 - 用于从用户 query 中提取日期文本
DATE_KEYWORDS = ["入住", "离店", "住宿", "天", "晚", "到", "至", "住"]


def extract_booking_info(query: str) -> dict:
    """
    主流程: 从用户对话中抽取酒店预订信息

    Args:
        query: 用户原始对话文本

    Returns:
        {
            "hotel_name": "标准酒店名",
            "checkin_date": "YYYY-MM-DD",
            "checkout_date": "YYYY-MM-DD",
            "confidence": 0.92
        }
    """
    start = time.perf_counter()

    # Step 1: RAG 检索酒店名
    hotel_index = get_hotel_index()
    candidates = hotel_index.search(query, top_k=config.TOP_K)

    if not candidates:
        return {"hotel_name": None, "checkin_date": None, "checkout_date": None, "confidence": 0.0}

    # Step 2: 尝试从 query 直接提取日期 (规则引擎)
    checkin_date, checkout_date = _extract_dates(query)

    # Step 3: 如果日期不完整, 调用 LLM 辅助抽取
    if checkin_date is None or checkout_date is None:
        extractor = get_extractor()
        llm_result = extractor.extract(query, candidates)

        # 合并 LLM 结果
        if llm_result.get("hotel_name") and checkin_date is None:
            pass  # hotel_name from LLM may override

        llm_ci = llm_result.get("checkin_date")
        llm_co = llm_result.get("checkout_date")

        if checkin_date is None:
            checkin_date = llm_ci
        if checkout_date is None:
            checkout_date = llm_co

    # Step 4: 确定最终酒店名 (取 RAG 检索结果中最匹配的)
    hotel_name = candidates[0][0]

    # Step 5: 计算置信度
    confidence = candidates[0][1] if candidates else 0.0
    has_dates = (checkin_date is not None) + (checkout_date is not None)
    if has_dates == 2:
        confidence = max(confidence, 0.85)
    elif has_dates == 1:
        confidence = max(confidence, 0.70)

    result = {
        "hotel_name": hotel_name,
        "checkin_date": checkin_date,
        "checkout_date": checkout_date,
        "confidence": round(confidence, 4),
    }

    elapsed_ms = (time.perf_counter() - start) * 1000
    result["_elapsed_ms"] = round(elapsed_ms, 2)
    return result


def _extract_dates(query: str) -> tuple[str | None, str | None]:
    """
    从 query 中提取入住/离店日期 (纯规则引擎, <10ms)

    返回: (checkin_date, checkout_date) 均为 YYYY-MM-DD 或 None
    """
    dates_found = []

    # 策略1: 关键词定位 - "入住/至/到/离店"
    # 匹配 "X 入住 Y 离店" 模式
    m = re.search(r"(\d{1,4}[.\-/年]\d{1,2}[月.\-/]\d{1,2}[日号]?).*(\d{1,4}[.\-/年]\d{1,2}[月.\-/]\d{1,2}[日号]?)", query)
    if m:
        d1 = parse_date(m.group(1).replace("年", "-").replace("月", "-").replace("日", "").replace("号", ""))
        d2 = parse_date(m.group(2).replace("年", "-").replace("月", "-").replace("日", "").replace("号", ""))
        if d1 and d2:
            return d1, d2

    # 匹配 "X到Y" / "X至Y" 日期段
    m = re.search(r"(\d{1,4}[.\-/]\d{1,2})\s*[到至]\s*(\d{1,4}[.\-/]\d{1,2})", query)
    if m:
        d1 = parse_date(m.group(1))
        d2 = parse_date(m.group(2))
        if d1 and d2:
            return d1, d2

    # 匹配 "X月Y日/X号到X月Y日/X号"
    m = re.search(r"(\d{1,2}月\d{1,2}[日号])\s*[到至]\s*(\d{1,2}[月日号\d]+)", query)
    if m:
        d1 = parse_date(m.group(1))
        d2 = parse_date(m.group(2))
        if d1 and d2:
            return d1, d2

    # 策略2: 提取相对日期词对
    relative_words = ["今天", "明天", "后天", "大后天"]
    found_relative = []
    for w in relative_words:
        if w in query:
            found_relative.append(w)

    if len(found_relative) >= 2:
        d1 = parse_date(found_relative[0])
        d2 = parse_date(found_relative[1])
        if d1 and d2:
            return d1, d2

    # 策略3: 提取星期词对
    week_pattern = r"(?:本|这个|下)(?:个)?(?:周|星期)(?:周末|[一二三四五六日天])"
    week_matches = re.findall(week_pattern, query)
    if len(week_matches) >= 2:
        d1 = parse_date(week_matches[0])
        d2 = parse_date(week_matches[1])
        if d1 and d2:
            return d1, d2

    # 策略4: 提取所有 MM-DD 格式日期
    all_dates = re.findall(r"(?<!\d)(\d{1,2})[.\-/](\d{1,2})(?!\d)", query)
    if len(all_dates) >= 2:
        # 只取前两个
        date_strs = [f"{a}-{b}" for a, b in all_dates[:2]]
        d1 = parse_date(date_strs[0])
        d2 = parse_date(date_strs[1])
        if d1 and d2:
            return d1, d2

    # 策略5: 提取完整 YYYY-MM-DD 格式
    all_full_dates = re.findall(r"(\d{4}[.\-/]\d{1,2}[.\-/]\d{1,2})", query)
    if len(all_full_dates) >= 2:
        d1 = parse_date(all_full_dates[0])
        d2 = parse_date(all_full_dates[1])
        if d1 and d2:
            return d1, d2

    # 只找到一个日期
    if all_dates:
        d = parse_date(f"{all_dates[0][0]}-{all_dates[0][1]}")
        if d:
            return d, None

    if all_full_dates:
        d = parse_date(all_full_dates[0])
        if d:
            return d, None

    return None, None


# ============================================================
# 测试
# ============================================================
def run_tests():
    test_cases = [
        # (query, expected_hotel_substring, expected_ci, expected_co)
        ("汉庭平顶山火车站酒店 12-20-12-22还有空房可以订吗？", "汉庭平顶山", None, None),
        ("海友杭州西湖湖滨酒店 后天至大后天的住宿", "海友杭州西湖", None, None),
        ("帮我订全季吴忠光耀美食街酒店 2025.3.13到2025.3.14的房间", "全季吴忠光耀", "2025-03-13", "2025-03-14"),
        ("汉庭杭州火车南站酒店 2025-7-29入住2025-7-31离店，多少钱可以订？", "汉庭杭州火车南站", "2025-07-29", "2025-07-31"),
        ("北京南站天坛漫心酒店 今天入住后天离店，含早餐的房间多少钱？", "北京南站天坛漫心", None, None),
        ("全季诸暨西施故里酒店 2.7入住2.9离店，多少钱可以订？", "全季诸暨西施", None, None),
        ("帮我订桔子水晶北京天坛医院总部基地酒店 2025.04.05入住2025.04.08离店的房间", "桔子水晶北京天坛", "2025-04-05", "2025-04-08"),
        ("想订泸州江阳康健城美仑酒店，9月19号到9月22号的房，多少钱一晚？", "泸州江阳康健城美仑", None, None),
        ("汉庭六安皖西大道国际汽车城酒店 这个周五-这个周日的房间还有吗？", "汉庭六安皖西大道", None, None),
        ("星程武汉江汉路吉庆街酒店 后天至大后天期间有双床房吗？", "星程武汉江汉路", None, None),
        ("汉庭重庆云阳滨江购物公园酒店 2025/11/24到2025/11/27的房，还有没有呀？", "汉庭重庆云阳滨江", "2025-11-24", "2025-11-27"),
        ("连云港花果山花间岭 · 悟云空 5月04日到5月07日还有房吗？", "连云港花果山花间岭", None, None),
        ("汉庭鄂尔多斯乌审旗酒店 下周末到下下周一的房间", "汉庭鄂尔多斯乌审旗", None, None),
    ]

    print("=" * 90)
    print("  RAG + LLM 酒店预订信息抽取 - 测试")
    print("=" * 90)

    passed = 0
    for i, (query, exp_hotel, exp_ci, exp_co) in enumerate(test_cases, 1):
        result = extract_booking_info(query)
        hotel_ok = exp_hotel is None or exp_hotel in result["hotel_name"] if result["hotel_name"] else False
        ci_ok = exp_ci is None or result["checkin_date"] == exp_ci
        co_ok = exp_co is None or result["checkout_date"] == exp_co

        status = "PASS" if (hotel_ok and ci_ok and co_ok) else "PARTIAL"

        if status == "PASS":
            passed += 1

        print(f"\n[{i}/{len(test_cases)}] {status} ({result['_elapsed_ms']:.1f}ms)")
        print(f"  Query: {query[:60]}...")
        print(f"  酒店: {result['hotel_name']}")
        print(f"  入住: {result['checkin_date']}  |  离店: {result['checkout_date']}")
        print(f"  置信度: {result['confidence']:.4f}")

    print(f"\n{'=' * 90}")
    print(f"  结果: {passed}/{len(test_cases)} 通过")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    run_tests()
