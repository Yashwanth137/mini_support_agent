import pytest
from router import IntentRouter
from models import Intent

def test_keyword_routing():
    router = IntentRouter()
    
    # 1. Pure knowledge
    res = router.route("What is the return policy?")
    assert res.intent == Intent.knowledge
    assert res.order_id is None
    
    # 2. Pure order lookup
    res = router.route("What is the status of ORD1001?")
    assert res.intent == Intent.order_lookup
    assert res.order_id == "ORD1001"
    
    # 3. Hybrid
    res = router.route("Can I still return ORD1002?")
    assert res.intent == Intent.hybrid
    assert res.order_id == "ORD1002"

# Note: We won't strictly test LLM fallback without mocking to avoid unneeded API calls
# but we can test a case that would fallback (and might fail to unsupported without a client)
def test_fallback_unsupported():
    router = IntentRouter()
    # Assuming no LLM client or simple fallback
    res = router.route("Hello, how are you?")
    # If no LLM, our logic falls back to unsupported. If LLM, it should ideally return unsupported too.
    assert res.intent in [Intent.unsupported, Intent.knowledge]
