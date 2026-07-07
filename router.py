import re
from google import genai
from config import settings
from models import Intent, RouteDecision
from prompts import INTENT_CLASSIFIER_PROMPT

class IntentRouter:
    def __init__(self):
        # Initialize GenAI client if needed for fallback
        self.client = None
        api_key = settings.gemini_api_key or settings.google_api_key
        if api_key:
            self.client = genai.Client(api_key=api_key)
        
        self.order_id_pattern = re.compile(r'ORD\d+', re.IGNORECASE)
        self.knowledge_keywords = ['return', 'policy', 'shipping', 'payment', 'refund', 'how long', 'how much']
        self.order_keywords = ['status', 'track', 'where is', 'cancel', 'update']

    def route(self, query: str) -> RouteDecision:
        query_lower = query.lower()
        
        # 1. Extract Order ID
        match = self.order_id_pattern.search(query)
        order_id = match.group(0).upper() if match else None

        # 2. Keyword matching
        has_knowledge_kw = any(kw in query_lower for kw in self.knowledge_keywords)
        has_order_kw = any(kw in query_lower for kw in self.order_keywords)

        # 3. Deterministic Routing Rules
        if order_id:
            if has_knowledge_kw:
                return RouteDecision(intent="hybrid", order_id=order_id, reason="Order ID and policy keyword detected")
            else:
                return RouteDecision(intent="order_lookup", order_id=order_id, reason="Order ID detected")
        
        if has_knowledge_kw and not has_order_kw:
            return RouteDecision(intent="knowledge", reason="Policy keyword detected")

        # 4. Fallback to LLM if ambiguous
        return self._llm_fallback(query)

    def _llm_fallback(self, query: str) -> RouteDecision:
        if not self.client:
            # If no API key, default to knowledge if unsure, or unsupported
            return RouteDecision(intent="unsupported", reason="Ambiguous query and no LLM client configured")
            
        try:
            prompt = INTENT_CLASSIFIER_PROMPT.format(query=query)
            # Use structured output
            response = self.client.models.generate_content(
                model=settings.llm_model_name,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=RouteDecision,
                    temperature=0.1
                ),
            )
            
            # The structured output is guaranteed to follow RouteDecision schema, 
            # though the SDK returns it as text which can be parsed, or if parsed directly:
            # According to `google-genai` SDK, structured outputs might be returned as JSON text.
            decision = RouteDecision.model_validate_json(response.text)
            decision.reason = decision.reason or "LLM fallback classification"
            return decision

        except Exception as e:
            print(f"LLM routing failed: {e}")
            return RouteDecision(intent="unsupported", reason="LLM error")
