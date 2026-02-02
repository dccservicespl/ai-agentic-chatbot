from pathlib import Path
from typing import Literal

from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from ai_agentic_chatbot.agent.schema import IntentResult
from ai_agentic_chatbot.agent.state import AgentState
from ai_agentic_chatbot.infrastructure.llm import get_llm
from ai_agentic_chatbot.agent.registry import IntentType
from ai_agentic_chatbot.infrastructure.llm.types import LLMProvider, ModelType

BASE_DIR = Path(__file__).resolve().parent.parent
PROMPT_PATH = BASE_DIR / "prompts" / "custom_prompts.md"


class RouterDecision(BaseModel):
    """Router classification schema with strict validation."""

    next_step: Literal["greeting", "actionable", "idiotic"] = Field(
        description=(
            "greeting: User is saying hi/hello or introducing themselves. "
            "actionable: User wants data, charts, analysis, or specific help. "
            "idiotic: Input is gibberish, spam, offensive, or completely irrelevant."
        )
    )
    reasoning: str = Field(
        description="Brief explanation (max 20 words) for this decision."
    )


class RouterNode:
    def __init__(self, state: AgentState):
        self.state: AgentState = state
        self.llm = get_llm()

    def classify(self) -> dict:
        structured_llm = self.llm.with_structured_output(RouterDecision, strict=True)
        last_msg = self.state.get("messages")[-1].content
        prompt = SystemMessage(
            content=(
                "You are an intent classifier. Analyze the user's message and classify it. "
                "Be strict: only mark as 'actionable' if they want actual data/charts/help."
            )
        )
        decision = structured_llm.invoke([prompt, last_msg])
        return {"next_step": decision.next_step}
