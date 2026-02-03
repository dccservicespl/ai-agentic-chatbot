import os
from pathlib import Path
from typing import Literal, Optional

from ai_agentic_chatbot.logging_config import get_logger
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

    next_step: Literal["greeting", "sql_query", "idiotic"] = Field(
        description=(
            "greeting: User is saying hi/hello or introducing themselves. "
            "sql_query: User wants data, charts, analysis, or specific help. That can be achieved by running a SQL query."
            "idiotic: Input is gibberish, spam, offensive, or completely irrelevant."
        )
    )
    reasoning: str = Field(
        description="Brief explanation (max 20 words) for this decision."
    )
    clarification: Optional[ClarificationDecision] = Field(
        description="Clarification decision if the user's message is ambiguous or unclear."
    )


logger = get_logger(__name__)


class RouterNode:
    def __init__(self, state: AgentState):
        self.state: AgentState = state
        self.llm = get_llm()

    def classify(self) -> dict:
        logger.debug(f"[ROUTER DEBUG] Sees {len(self.state['messages'])} messages")
        for i, msg in enumerate(self.state["messages"]):
            logger.debug(f"  [{i}] {type(msg).__name__}: {msg.content[:50]}")

        structured_llm = self.llm.with_structured_output(RouterDecision, strict=True)
        msgs = self.state["messages"]

        with open(os.environ["ROUTER_PROMPT_PATH"], "r") as f:
            prompt_text = f.read()
        prompt = SystemMessage(content=(prompt_text))
        decision = structured_llm.invoke([prompt] + msgs)

        if decision.clarification and decision.clarification.is_ambiguous:
            return {
                "next_step": "ask_clarification",
                "messages": [
                    AIMessage(content=decision.clarification.clarification_question),
                ],
            }

        return {"next_step": decision.next_step}
