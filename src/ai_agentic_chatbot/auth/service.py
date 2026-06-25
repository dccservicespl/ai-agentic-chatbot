from sqlalchemy.orm import Session

from ai_agentic_chatbot.auth.models import User
from ai_agentic_chatbot.auth.password import hash_password, verify_password
from ai_agentic_chatbot.auth.repository import (
    create_user,
    get_user_by_email,
    get_user_by_username,
)
from ai_agentic_chatbot.auth.schemas import UserCreate


def authenticate_user(session: Session, username: str, password: str) -> User | None:
    user = get_user_by_username(session, username)
    if user is None or not verify_password(password, user.hashed_password):
        return None
    return user


def create_user_account(session: Session, user_data: UserCreate) -> User:
    if get_user_by_username(session, user_data.username) is not None:
        raise ValueError("Username already registered")
    if get_user_by_email(session, user_data.email) is not None:
        raise ValueError("Email already registered")
    return create_user(
        session,
        username=user_data.username,
        email=user_data.email,
        hashed_password=hash_password(user_data.password),
    )