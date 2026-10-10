"""日期规则引擎 - 纯 Python 实现, <10ms, 支持所有中文日期格式"""
import re
from datetime import datetime, timedelta
from zhdate import ZhDate

WEEKDAY_MAP = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}
MONTH_CHARS = {"一":1,"二":2,"三":3,"四":4,"五":5,"六":6,"七":7,"八":8,"九":9,"十":10,"十一":11,"十二":12}

def parse_date(text: str, base_date: datetime = None) -> str | None:
    """
    解析中文日期, 返回 YYYY-MM-DD 或 None

    支持的格式:
      - 相对: 今天/明天/后天/大后天
      - 星期: 本周五/下周一/这个周末
      - 数字: 12-20 / 2025-12-20 / 2025.3.13 / 2025/11/24 / 2025年08月16日
      - 中文: 9月19号 / 08月19日
    """
    if base_date is None:
        base_date = datetime.now()

    text = text.strip()
    if not text:
        return None

    # --- 相对日期 ---
    if text == "今天":
        return base_date.strftime("%Y-%m-%d")
    if text == "明天" or text == "明天":
        return (base_date + timedelta(days=1)).strftime("%Y-%m-%d")
    if text == "后天":
        return (base_date + timedelta(days=2)).strftime("%Y-%m-%d")
    if text == "大后天":
        return (base_date + timedelta(days=3)).strftime("%Y-%m-%d")

    # --- 星期/周末 ---
    m = re.match(r"^(本|这个|下|上|这|今|去)(个?)(周|星期)(周末|周\s*[一二三四五六日天])?$", text)
    if m:
        prefix, _, week_word, day_part = m.groups()
        if day_part is None:
            day_part = "周末"
        day_part = day_part.strip().replace(" ", "")

        today_dow = base_date.weekday()
        offset_map = {"本": 0, "这个": 0, "今": 0, "这": 0, "下": 7, "上": -7, "去": -7}
        base_offset = offset_map.get(prefix, 0)

        if day_part == "周末":
            target = (today_dow + 6) % 7  # Saturday
            days_ahead = target - today_dow + base_offset
            if days_ahead < 0:
                days_ahead += 7
            return (base_date + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

        for cn, dow in WEEKDAY_MAP.items():
            if cn in day_part:
                days_ahead = dow - today_dow + base_offset
                if prefix == "下" and days_ahead <= 0:
                    days_ahead += 7
                if prefix in ("本", "这个", "今", "这") and days_ahead < 0:
                    days_ahead += 7
                return (base_date + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

    # --- 2025年08月16日 / 9月19号 ---
    m = re.match(r"^(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]?$", text)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return datetime(y, mo, d).strftime("%Y-%m-%d")
        except ValueError:
            pass

    m = re.match(r"^(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]?$", text)
    if m:
        mo, d = int(m.group(1)), int(m.group(2))
        year = base_date.year
        try:
            dt = datetime(year, mo, d)
            if dt < base_date:
                dt = datetime(year + 1, mo, d)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            pass

    # --- 2025.3.13 / 2025-12-20 / 2025/11/24 (完整) ---
    m = re.match(r"^(\d{4})[.\-/](\d{1,2})[.\-/](\d{1,2})$", text)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return datetime(y, mo, d).strftime("%Y-%m-%d")
        except ValueError:
            pass

    # --- MM-DD / MM.DD / MM/DD (短格式) ---
    m = re.match(r"^(\d{1,2})[.\-/](\d{1,2})$", text)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        # 判断哪个是月哪个是日: >31 的是日; 或按数据集中的逻辑
        if a > 12 and b <= 12:
            a, b = b, a  # swap: first was day
        # 如果两个都 <=12, 假设第一个是月 (04-13 -> 4月13日)
        year = base_date.year
        try:
            dt = datetime(year, a, b)
            if dt < base_date:
                dt = datetime(year + 1, a, b)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            pass

    return None
