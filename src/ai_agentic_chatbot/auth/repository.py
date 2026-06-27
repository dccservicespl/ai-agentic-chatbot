import uuid
from datetime import date, datetime, time, timezone

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from ai_agentic_chatbot.auth.models import User, RefreshToken, PromptLog


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


def update_user_password(db: Session, user: User, new_hashed: str) -> User:
    user.hashed_password = new_hashed
    db.commit()
    db.refresh(user)
    return user


def create_prompt_log(
    db: Session,
    *,
    user_id: int,
    thread_id: str,
    prompt_text: str,
) -> PromptLog:
    log = PromptLog(user_id=user_id, thread_id=thread_id, prompt_text=prompt_text)
    db.add(log)
    db.commit()
    db.refresh(log)
    return log


def count_prompts_today(db: Session, user_id: int) -> int:
    today_start = datetime.combine(date.today(), time.min).replace(tzinfo=timezone.utc)
    result = db.execute(
        select(func.count()).select_from(PromptLog).where(
            PromptLog.user_id == user_id,
            PromptLog.created_at >= today_start,
        )
    ).scalar_one()
    return result


def update_user_prompt_limit(db: Session, user: User, limit: int) -> User:
    user.daily_prompt_limit = limit
    db.commit()
    db.refresh(user)
    return user


def get_all_users(db: Session) -> list[User]:
    return list(db.execute(select(User).order_by(User.id)).scalars().all())