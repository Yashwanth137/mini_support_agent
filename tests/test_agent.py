import pytest
from agent import SupportAgent
from models import Intent

def test_unsupported_query():
    agent = SupportAgent()
    # A query that should fall through routing if no LLM or just default to unsupported
    # Or explicitly tested against the default fallback
    response = agent.process_query("Tell me a joke")
    assert response.intent_used in [Intent.unsupported, Intent.knowledge]

def test_order_missing():
    agent = SupportAgent()
    response = agent.process_query("What's the status of ORD9999?")
    assert response.intent_used == Intent.order_lookup
    assert "couldn't find any information" in response.answer.lower()

def test_hybrid_reasoning():
    agent = SupportAgent()
    # Assuming ORD1001 exists
    # This query has both an order ID and a policy keyword
    response = agent.process_query("Can I still return ORD1001?")
    # Based on our deterministic keyword router, it will route to hybrid
    assert response.intent_used == Intent.hybrid

def test_missing_delivery_date():
    from business_logic import check_return_eligibility
    from datetime import date
    from models import Order
    
    # Mock an order with no delivery date
    order = Order(
        order_id="ORD9999",
        customer_name="Test",
        product="Test",
        category="Electronics",
        amount_inr=100,
        order_date="2026-07-01",
        delivery_date=None,
        status="Shipped",
        payment_method="UPI",
        pincode="123456"
    )
    
    result = check_return_eligibility(order, None, date.today())
    assert result["eligible"] is None
    assert "unavailable" in result["reason"].lower()

