from typing import Dict, Optional
from pydantic import BaseModel, Field
from ai_agentic_chatbot.agent.registry import IntentType


class IntentResult(BaseModel):
    intent: IntentType = Field(..., description="Classified intent")
    entities: Dict[str, str] = Field(default_factory=dict)
    confidence: float = Field(..., ge=0.0, le=1.0)
    requires_followup: bool = False
