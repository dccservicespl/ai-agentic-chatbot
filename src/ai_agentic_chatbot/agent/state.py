from typing import Optional, Dict, Any

from langgraph.graph import MessagesState


class AgentState(MessagesState):
    next_step: str
    relevant_tables: Optional[list[str]]
    visualization: Optional[Dict[str, Any]]
