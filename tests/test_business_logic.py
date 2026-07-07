import pytest
from datetime import date, timedelta
from business_logic import check_return_eligibility
from models import Order

def get_mock_order(category="Apparel", delivery_date_str="2026-07-01"):
    return Order(
        order_id="ORD123",
        customer_name="Test User",
        product="Test Product",
        category=category,
        amount_inr=1000,
        order_date="2026-06-25",
        delivery_date=delivery_date_str,
        status="Delivered",
        payment_method="UPI",
        pincode="110001"
    )

def test_missing_delivery_date():
    order = get_mock_order(delivery_date_str=None)
    result = check_return_eligibility(order, None, date(2026, 7, 5))
    assert result["eligible"] is None
    assert "unavailable" in result["reason"].lower()

def test_future_delivery_date():
    order = get_mock_order(delivery_date_str="2026-07-10")
    # Current date is earlier than delivery
    result = check_return_eligibility(order, date(2026, 7, 10), date(2026, 7, 5))
    assert result["eligible"] is False
    assert result["days_since_delivery"] < 0
    assert "future" in result["reason"].lower()

def test_non_returnable_category():
    order = get_mock_order(category="Cosmetics")
    result = check_return_eligibility(order, date(2026, 7, 1), date(2026, 7, 2))
    assert result["eligible"] is False
    assert "non-returnable" in result["reason"].lower()

def test_electronics_day_4():
    order = get_mock_order(category="Electronics")
    # Delivered on 1st, checked on 5th -> 4 days
    result = check_return_eligibility(order, date(2026, 7, 1), date(2026, 7, 5))
    assert result["eligible"] is False
    assert result["window"] == 3
    assert result["days_since_delivery"] == 4

def test_exactly_day_3():
    order = get_mock_order(category="Appliances")
    # Delivered on 1st, checked on 4th -> 3 days
    result = check_return_eligibility(order, date(2026, 7, 1), date(2026, 7, 4))
    assert result["eligible"] is True
    assert result["window"] == 3
    assert result["days_since_delivery"] == 3

def test_exactly_day_7():
    order = get_mock_order(category="Apparel")
    # Delivered on 1st, checked on 8th -> 7 days
    result = check_return_eligibility(order, date(2026, 7, 1), date(2026, 7, 8))
    assert result["eligible"] is True
    assert result["window"] == 7
    assert result["days_since_delivery"] == 7

def test_day_8():
    order = get_mock_order(category="Home & Kitchen")
    # Delivered on 1st, checked on 9th -> 8 days
    result = check_return_eligibility(order, date(2026, 7, 1), date(2026, 7, 9))
    assert result["eligible"] is False
    assert result["window"] == 7
    assert result["days_since_delivery"] == 8
