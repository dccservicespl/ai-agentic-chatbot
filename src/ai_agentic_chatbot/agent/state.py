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
    # Surfaced from the SQL subgraph's result by sql_query_node (agent/graph.py)
    # on the success path only — absent/default on error paths, which is exactly
    # what lets server.py's post-stream prompt_history write skip failed turns
    # for free by checking generated_sql's truthiness.
    generated_sql: Optional[str]
    was_cache_hit: bool
    cache_row_id: Optional[int]
