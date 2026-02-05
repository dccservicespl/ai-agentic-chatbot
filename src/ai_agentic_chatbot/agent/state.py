from typing import Optional

from langgraph.graph import MessagesState


class AgentState(MessagesState):
    next_step: str
    relevant_tables: Optional[list[str]]
