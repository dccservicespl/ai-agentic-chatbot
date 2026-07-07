from typing import Dict, Optional, List, Literal
from pydantic import BaseModel, Field
from ai_agentic_chatbot.agent.registry import IntentType


class IntentResult(BaseModel):
    intent: IntentType = Field(..., description="Classified intent")
    entities: Dict[str, str] = Field(default_factory=dict)
    confidence: float = Field(..., ge=0.0, le=1.0)
    requires_followup: bool = False


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class StreamRequest(BaseModel):
    thread_id: str = Field(
        ..., description="Unique identifier for the conversation thread."
    )
    messages: List[Message]
    context_id: Optional[str] = Field(
        default=None,
        description="Context slug from config.yaml's contexts: block. If omitted, the user's default context is used (see TODO 18).",
    )
