from typing import TypedDict, List

from langgraph.graph import MessagesState


class AgentState(MessagesState):
    next_step: str
