# Price calculation demo

This deliberately small project begins with failing boundary and money-precision tests. The task is to keep `calculate_total` and its float-based public signature unchanged while using decimal arithmetic internally, applying discounts inclusively at the threshold, adding a relevant boundary test, and making `pytest -q` pass.

