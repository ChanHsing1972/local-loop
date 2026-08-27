from pricing import calculate_total


def test_discount_applies_at_threshold_boundary():
    assert calculate_total([60.0, 40.0], discount_rate=0.1, threshold=100.0) == 90.0


def test_decimal_money_is_not_rounded_before_discount():
    prices = [0.335, 0.335, 99.33]
    assert calculate_total(prices, discount_rate=0.1, threshold=100.0) == 90.0


def test_no_discount_below_threshold():
    assert calculate_total([20.0, 30.0], discount_rate=0.2, threshold=100.0) == 50.0

