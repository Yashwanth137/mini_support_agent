from datetime import date, datetime
from typing import Optional, Dict, Any
from models import Order

def check_return_eligibility(order: Order, delivery_date: Optional[date], current_date: date) -> Dict[str, Any]:
    """
    Deterministically computes return eligibility based on category and delivery date.
    Returns a dictionary of computed facts to be injected into the LLM context.
    """
    if not delivery_date:
        return {
            "eligible": None,
            "days_since_delivery": None,
            "window": None,
            "reason": "Delivery date unavailable."
        }
        
    days_since_delivery = (current_date - delivery_date).days
    
    # Negative days mean the delivery is in the future, which is logically an error, but we'll treat it gracefully.
    if days_since_delivery < 0:
        return {
            "eligible": False,
            "days_since_delivery": days_since_delivery,
            "window": None,
            "reason": "Delivery date is in the future."
        }

    category = order.category.lower()
    
    # Check non-returnable categories
    non_returnable = ["cosmetics", "innerwear", "perishable"]
    if any(nr in category for nr in non_returnable):
        return {
            "eligible": False,
            "days_since_delivery": days_since_delivery,
            "window": 0,
            "reason": f"Items in the {order.category} category are non-returnable for hygiene/safety reasons."
        }
        
    # Check specific window rules
    if "electronics" in category or "appliances" in category:
        window = 3
    else:
        window = 7
        
    eligible = days_since_delivery <= window
    
    reason = f"Within {window}-day return window." if eligible else f"Exceeds {window}-day return window."
    
    return {
        "eligible": eligible,
        "days_since_delivery": days_since_delivery,
        "window": window,
        "reason": reason
    }
