from pydantic import BaseModel, ConfigDict


class ContextAdminResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    context_id: str
    display_name: str
    schema_name: str
    is_active: bool


class UserContextResponse(BaseModel):
    context_id: str
    display_name: str
    is_default: bool