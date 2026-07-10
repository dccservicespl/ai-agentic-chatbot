from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict


class PromptHistoryListItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    thread_id: str
    db_context_id: int
    raw_prompt: str
    chart_type: Optional[str]
    was_cache_hit: bool
    executed_at: datetime


class PromptHistoryDetail(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    thread_id: str
    db_context_id: int
    raw_prompt: str
    generated_sql: Optional[str]
    was_cache_hit: bool
    chart_type: Optional[str]
    result_snapshot: Optional[dict]
    executed_at: datetime


class RefreshResponse(BaseModel):
    visualization: dict
    generated_sql: str
    analysis: Optional[str] = None


class RegenerateResponse(BaseModel):
    visualization: dict
    generated_sql: str
    explanation: Optional[str] = None
    analysis: Optional[str] = None