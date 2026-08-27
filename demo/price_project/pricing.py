"""供 LocalLoop 演示使用、故意保留缺陷的小型模块。"""


def calculate_total(
    prices: list[float], discount_rate: float = 0.0, threshold: float = 0.0
) -> float:
    """返回保留两位小数的总额，并在达到设定阈值时应用折扣。"""

    subtotal = sum(prices)
    if subtotal > threshold:
        subtotal *= 1 - discount_rate
    return round(subtotal, 2)
