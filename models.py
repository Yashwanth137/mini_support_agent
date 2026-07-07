from enum import Enum
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field

class Intent(str, Enum):
    knowledge = "knowledge"
    order_lookup = "order_lookup"
    hybrid = "hybrid"
    unsupported = "unsupported"

class RouteDecision(BaseModel):
    intent: Literal[
        "knowledge",
        "order_lookup",
        "hybrid",
        "unsupported",
    ]
    order_id: Optional[str] = None
    reason: Optional[str] = None

class Order(BaseModel):
    order_id: str
    customer_name: str
    product: str
    category: str
    amount_inr: int
    order_date: str
    delivery_date: Optional[str] = None
    status: str
    payment_method: str
    pincode: str

class Document(BaseModel):
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class AgentResponse(BaseModel):
    answer: str
    intent_used: Intent
    sources: List[str] = Field(default_factory=list)
