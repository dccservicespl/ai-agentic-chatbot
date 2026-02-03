from pathlib import Path
from typing import Literal, Optional

from langchain_core.messages import SystemMessage
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from ai_agentic_chatbot.agent.state import AgentState
from ai_agentic_chatbot.infrastructure.llm import get_llm

BASE_DIR = Path(__file__).resolve().parent.parent
PROMPT_PATH = BASE_DIR / "prompts" / "custom_prompts.md"


class ClarificationDecision(BaseModel):
    is_ambiguous: bool = Field(
        description="True if the user's message is ambiguous or unclear."
    )
    clarification_question: Optional[str] = Field(
        description="Question to ask the user to clarify their intent."
    )
    missing_info: Optional[str] = Field(
        description="The comma separated list of information missing from the user's message. e.g. time period, location, etc."
    )


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
    clarification: Optional[ClarificationDecision] = Field(
        description="Clarification decision if the user's message is ambiguous or unclear."
    )


class RouterNode:
    def __init__(self, state: AgentState):
        self.state: AgentState = state
        self.llm = get_llm()

    def classify(self) -> dict:
        print(f"[ROUTER DEBUG] Sees {len(self.state['messages'])} messages")
        for i, msg in enumerate(self.state["messages"]):
            print(f"  [{i}] {type(msg).__name__}: {msg.content[:50]}")
        structured_llm = self.llm.with_structured_output(RouterDecision, strict=True)
        msgs = self.state["messages"]
        prompt = SystemMessage(
            content=(
                "You are an intent classifier for a SQL Data Assistant."
                "1. If user wants data/charts (e.g. 'Show sales', 'growth rate'), classify as 'sql_query'."
                "2. If user greets ('hi', 'thanks'), classify as 'greeting'."
                "3. If nonsense/unrelated, classify as 'idiotic'."
                "CRITICAL: If 'sql_query', check for ambiguity."
                "- 'Sales' -> Ambiguous (needs year/product)."
                "- 'Sales 2024' -> Not Ambiguous."
                "If Ambiguous, set is_ambiguous=True and write a polite clarification_question."
            )
        )
        decision = structured_llm.invoke([prompt] + msgs)
        if decision.clarification and decision.clarification.is_ambiguous:
            return {
                "next_step": "ask_clarification",
                "messages": [
                    AIMessage(content=decision.clarification.clarification_question),
                ],
            }

        return {"next_step": decision.next_step}
