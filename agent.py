import json
from google import genai
from datetime import datetime
from config import settings
from models import Intent, AgentResponse
from router import IntentRouter
from tools import OrderTools
from rag import RAGPipeline
from business_logic import check_return_eligibility
from prompts import RAG_QA_PROMPT, ORDER_SUMMARIZER_PROMPT, HYBRID_REASONING_PROMPT

class SupportAgent:
    def __init__(self):
        self.router = IntentRouter()
        self.tools = OrderTools()
        self.rag = RAGPipeline()
        
        self.client = None
        api_key = settings.gemini_api_key or settings.google_api_key
        if api_key:
            self.client = genai.Client(api_key=api_key)

    def process_query(self, query: str) -> AgentResponse:
        # 1. Route the query
        classification = self.router.route(query)
        try:
            intent = Intent(classification.intent)
        except ValueError:
            intent = Intent.unsupported
        order_id = classification.order_id

        # Execute appropriate workflow
        if intent == Intent.knowledge:
            return self._handle_knowledge(query)
        elif intent == Intent.order_lookup:
            return self._handle_order_lookup(query, order_id)
        elif intent == Intent.hybrid:
            return self._handle_hybrid(query, order_id)
        else:
            return AgentResponse(
                answer="I'm sorry, I can only help with questions about store policies or specific orders.",
                intent_used=Intent.unsupported
            )

    def _call_llm(self, prompt: str) -> str:
        if not self.client:
            return "Error: LLM client not configured (Missing API key)."
        try:
            response = self.client.models.generate_content(
                model=settings.llm_model_name,
                contents=prompt
            )
            return response.text
        except Exception as e:
            return f"Error generating response: {e}"

    def _handle_knowledge(self, query: str) -> AgentResponse:
        docs = self.rag.retrieve(query)
        if not docs:
            return AgentResponse(
                answer="I couldn't find any relevant policy information to answer your question.",
                intent_used=Intent.knowledge
            )
        
        context = "\n\n".join([doc.content for doc in docs])
        prompt = RAG_QA_PROMPT.format(context=context, query=query)
        
        answer = self._call_llm(prompt)
        sources = list(set([doc.metadata.get("source", "unknown") for doc in docs]))
        
        return AgentResponse(
            answer=answer,
            intent_used=Intent.knowledge,
            sources=sources
        )

    def _handle_order_lookup(self, query: str, order_id: str) -> AgentResponse:
        if not order_id:
            return AgentResponse(
                answer="I couldn't identify an order ID in your request. Please provide an order ID (e.g., ORD1001).",
                intent_used=Intent.order_lookup
            )

        order = self.tools.get_order(order_id)
        if not order:
            return AgentResponse(
                answer=f"I couldn't find any information for order {order_id}.",
                intent_used=Intent.order_lookup
            )

        order_data = order.model_dump_json(indent=2)
        prompt = ORDER_SUMMARIZER_PROMPT.format(order_data=order_data, query=query)
        
        answer = self._call_llm(prompt)
        
        return AgentResponse(
            answer=answer,
            intent_used=Intent.order_lookup,
            sources=["orders.csv"]
        )

    def _handle_hybrid(self, query: str, order_id: str) -> AgentResponse:
        if not order_id:
            return AgentResponse(
                answer="I couldn't identify an order ID in your request. Please provide an order ID.",
                intent_used=Intent.hybrid
            )

        order = self.tools.get_order(order_id)
        if not order:
            return AgentResponse(
                answer=f"I couldn't find any information for order {order_id}.",
                intent_used=Intent.hybrid
            )

        docs = self.rag.retrieve(query)
        context = "\n\n".join([doc.content for doc in docs]) if docs else "No policy documents found."
        
        # Calculate deterministically
        try:
            delivery_date = datetime.strptime(order.delivery_date, "%Y-%m-%d").date() if order.delivery_date else None
        except ValueError:
            delivery_date = None
            
        current_date = datetime.today().date()
        eligibility = check_return_eligibility(order, delivery_date, current_date)
        
        computed_facts = json.dumps(eligibility, indent=2)
        prompt = HYBRID_REASONING_PROMPT.format(computed_facts=computed_facts, context=context, query=query)
        
        answer = self._call_llm(prompt)
        sources = ["orders.csv"] + list(set([doc.metadata.get("source", "unknown") for doc in docs]))
        
        return AgentResponse(
            answer=answer,
            intent_used=Intent.hybrid,
            sources=sources
        )
