import pytest
from tools import OrderTools

def test_order_lookup():
    tools = OrderTools()
    
    # Assuming ORD1001 exists from initial data view
    assert tools.order_exists("ORD1001") is True
    
    order = tools.get_order("ORD1001")
    assert order is not None
    assert order.order_id == "ORD1001"
    
    status = tools.get_status("ORD1001")
    assert status in ["Delivered", "Shipped", "Processing", "Confirmed", "Cancelled"]

def test_invalid_order():
    tools = OrderTools()
    assert tools.order_exists("ORD9999") is False
    assert tools.get_order("ORD9999") is None
    assert tools.get_status("ORD9999") is None
