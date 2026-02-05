import os
from pathlib import Path
from typing import Literal, Optional

from ai_agentic_chatbot.logging_config import get_logger
from ai_agentic_chatbot.schema_extractor.transform_schema_to_text import (
    load_schema_summary,
)
from ai_agentic_chatbot.utils.prompt_loader import get_system_prompt
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


class RouterDecision(BaseModel):
    """Router classification schema with strict validation."""

    intent: Literal["greeting", "sql_query", "nonsense"] = Field(
        description=(
            "greeting: User is saying hi/hello or introducing themselves. "
            "sql_query: User wants data, charts, analysis, or specific help. That can be achieved by running a SQL query."
            "nonsense: Input is gibberish, spam, offensive, or completely irrelevant."
        )
    )
    reasoning: str = Field(
        description="Brief explanation (max 20 words) for this decision."
    )
    is_answerable: bool = Field(
        description="True if the query can be answered using the AVAILABLE TABLES provided in the system prompt. False if data is missing.True if further clarification needed."
    )
    missing_data_reason: Optional[str] = Field(
        description="If is_answerable is False, explain what data is missing (e.g., 'We do not have employee salary data')."
    )
    clarification: Optional[ClarificationDecision] = Field(
        description="Clarification decision if the user's message is ambiguous or unclear."
    )
    relevant_tables: Optional[list[str]] = Field(
        description="If is_answerable is True, list the relevant tables that can be used to answer the query."
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
        schema_summary = load_schema_summary()
        schema_summary_text = "\n".join(
            [f"- {table}: {desc}" for table, desc in schema_summary.items()]
        )
        base_prompt = SystemMessage(content=get_system_prompt())
        prompt = SystemMessage(
            content=prompt_text.format(schema_text=schema_summary_text)
        )
        decision = structured_llm.invoke([base_prompt, prompt] + msgs)

        if decision.intent == "greeting":
            return {
                "next_step": "greeting",
            }

        if not decision.is_answerable:
            response_msg = (
                f"\n\nI can help you with: {', '.join(schema_summary.keys())}."
            )
            response_msg += (
                decision.missing_data_reason or "I don't have the data to answer that."
            )
            return {
                "next_step": "nonsense",
                "messages": [AIMessage(content=response_msg)],
            }

        if (
            decision.intent == "sql_query"
            and decision.clarification
            and decision.clarification.is_ambiguous
        ):
            return {
                "next_step": "ask_clarification",
                "messages": [
                    AIMessage(content=decision.clarification.clarification_question),
                ],
            }

        return {
            "next_step": decision.intent,
            "relevant_tables": decision.relevant_tables,
        }
