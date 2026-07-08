import re
import logging
from google import genai
from config import settings
from models import Intent, RouteDecision
from prompts import INTENT_CLASSIFIER_PROMPT

logger = logging.getLogger(__name__)

class IntentRouter:
    def __init__(self):
        self.client = None
        api_key = settings.gemini_api_key or settings.google_api_key
        if api_key:
            self.client = genai.Client(api_key=api_key)
        
        self.order_id_pattern = re.compile(r'ORD\d+', re.IGNORECASE)
        self.knowledge_keywords = ['return', 'policy', 'shipping', 'payment', 'refund', 'how long', 'how much']
        self.order_keywords = ['status', 'track', 'where is', 'cancel', 'update']

    def route(self, query: str) -> RouteDecision:
        query_lower = query.lower()
        
        match = self.order_id_pattern.search(query)
        order_id = match.group(0).upper() if match else None

        has_knowledge_kw = any(kw in query_lower for kw in self.knowledge_keywords)
        has_order_kw = any(kw in query_lower for kw in self.order_keywords)

        if order_id:
            if has_knowledge_kw:
                return RouteDecision(intent="hybrid", order_id=order_id, reason="Order ID and policy keyword detected")
            else:
                return RouteDecision(intent="order_lookup", order_id=order_id, reason="Order ID detected")
        
        if has_knowledge_kw and not has_order_kw:
            return RouteDecision(intent="knowledge", reason="Policy keyword detected")

        return self._llm_fallback(query)

    def _llm_fallback(self, query: str) -> RouteDecision:
        if not self.client:
            return RouteDecision(intent="unsupported", reason="Ambiguous query and no LLM client configured")
            
        try:
            prompt = INTENT_CLASSIFIER_PROMPT.format(query=query)
            response = self.client.models.generate_content(
                model=settings.llm_model_name,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=RouteDecision,
                    temperature=0.1
                ),
            )
            
            decision = RouteDecision.model_validate_json(response.text)
            decision.reason = decision.reason or "LLM fallback classification"
            return decision

        except Exception as e:
            logger.debug("LLM routing unavailable, falling back to unsupported intent: %s", e)
            return RouteDecision(intent="unsupported", reason="LLM error")
