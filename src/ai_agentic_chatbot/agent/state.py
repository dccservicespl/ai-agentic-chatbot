from typing import Optional, Dict, Any

from langgraph.graph import MessagesState


class AgentState(MessagesState):
    next_step: str
    relevant_tables: Optional[list[str]]
    visualization: Optional[Dict[str, Any]]
    analysis: Optional[str]
    # Which config.yaml context (DbContextConfig.context_id) this thread is bound to.
    # Injected by the /stream route (TODO 18) — not yet populated as of TODO 10.
    # A thread must not switch context_id mid-conversation; enforcing that against
    # the LangGraph checkpoint is also TODO 18's responsibility, not this state model's.
    db_context_id: str
