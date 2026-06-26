from datetime import datetime

from pydantic import BaseModel, ConfigDict


class UserCreate(BaseModel):
    username: str
    email: str
    password: str


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    email: str
    is_active: bool
    is_superuser: bool
    created_at: datetime


class Token(BaseModel):
    access_token: str
    refresh_token: str | None = None
    token_type: str = "bearer"


class TokenData(BaseModel):
    username: str | None = None


class RefreshRequest(BaseModel):
    refresh_token: str


class LogoutRequest(BaseModel):
    refresh_token: str