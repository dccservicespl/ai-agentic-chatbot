import uuid
from datetime import datetime

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from ai_agentic_chatbot.auth.models import User, RefreshToken


def get_user_by_username(session: Session, username: str) -> User | None:
    return session.execute(select(User).where(User.username == username)).scalar_one_or_none()


def get_user_by_email(session: Session, email: str) -> User | None:
    return session.execute(select(User).where(User.email == email)).scalar_one_or_none()


def get_user_by_id(session: Session, user_id: int) -> User | None:
    return session.execute(select(User).where(User.id == user_id)).scalar_one_or_none()


def create_user(
    session: Session,
    *,
    username: str,
    email: str,
    hashed_password: str,
    is_superuser: bool = False,
) -> User:
    user = User(
        username=username,
        email=email,
        hashed_password=hashed_password,
        is_superuser=is_superuser,
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


# ── Refresh token CRUD ────────────────────────────────────────────────────────

def create_refresh_token(
    session: Session,
    *,
    user_id: int,
    family_id: uuid.UUID,
    token_hash: str,
    expires_at: datetime,
) -> RefreshToken:
    rt = RefreshToken(
        user_id=user_id,
        family_id=family_id,
        token_hash=token_hash,
        expires_at=expires_at,
    )
    session.add(rt)
    session.commit()
    session.refresh(rt)
    return rt


def get_refresh_token_by_hash(session: Session, token_hash: str) -> RefreshToken | None:
    return session.execute(
        select(RefreshToken).where(RefreshToken.token_hash == token_hash)
    ).scalar_one_or_none()


def mark_token_used(session: Session, token_id: uuid.UUID) -> None:
    session.execute(
        update(RefreshToken).where(RefreshToken.id == token_id).values(used=True)
    )
    session.commit()


def revoke_family(session: Session, family_id: uuid.UUID) -> None:
    session.execute(
        update(RefreshToken).where(RefreshToken.family_id == family_id).values(revoked=True)
    )
    session.commit()


def revoke_token(session: Session, token_id: uuid.UUID) -> None:
    session.execute(
        update(RefreshToken).where(RefreshToken.id == token_id).values(revoked=True)
    )
    session.commit()