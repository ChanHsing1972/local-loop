"""Small intentionally buggy module used by the LocalLoop demonstration."""


def calculate_total(
    prices: list[float], discount_rate: float = 0.0, threshold: float = 0.0
) -> float:
    """Return a two-decimal total, applying a discount at the configured threshold."""

    subtotal = sum(prices)
    if subtotal > threshold:
        subtotal *= 1 - discount_rate
    return round(subtotal, 2)

