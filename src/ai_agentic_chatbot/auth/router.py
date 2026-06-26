import os
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from ai_agentic_chatbot.auth.dependencies import get_auth_db, get_current_user
from ai_agentic_chatbot.auth.jwt_utils import create_access_token, generate_refresh_token, hash_token
from ai_agentic_chatbot.auth.models import User
from ai_agentic_chatbot.auth.repository import (
    create_refresh_token,
    get_refresh_token_by_hash,
    get_user_by_id,
    mark_token_used,
    revoke_family,
    revoke_token,
)
from ai_agentic_chatbot.auth.schemas import LogoutRequest, RefreshRequest, Token, UserCreate, UserResponse
from ai_agentic_chatbot.auth.service import authenticate_user, create_user_account

router = APIRouter(prefix="/auth", tags=["Auth"])

_REFRESH_EXPIRE_DAYS = int(os.environ.get("JWT_REFRESH_TOKEN_EXPIRE_DAYS", "7"))


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user account",
    description=(
        "Creates a new user with the provided username, email, and password. "
        "Returns the created user profile. Returns HTTP 409 if the username or email is already taken."
    ),
)
def register(user_data: UserCreate, db: Session = Depends(get_auth_db)):
    try:
        user = create_user_account(db, user_data)
        return UserResponse.model_validate(user)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))


@router.post(
    "/login",
    response_model=Token,
    summary="Login and obtain JWT access + refresh tokens",
    description=(
        "Authenticates with username and password (form data). "
        "Returns a short-lived Bearer JWT access token and a long-lived opaque refresh token. "
        "Returns HTTP 401 if credentials are invalid."
    ),
)
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_auth_db),
):
    user = authenticate_user(db, form_data.username, form_data.password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token({"sub": user.username})

    raw_rt = generate_refresh_token()
    create_refresh_token(
        db,
        user_id=user.id,
        family_id=uuid.uuid4(),
        token_hash=hash_token(raw_rt),
        expires_at=datetime.now(timezone.utc) + timedelta(days=_REFRESH_EXPIRE_DAYS),
    )
    return Token(access_token=access_token, refresh_token=raw_rt)


@router.post(
    "/refresh",
    response_model=Token,
    summary="Rotate a refresh token and obtain a new token pair",
    description=(
        "Validates the supplied refresh token, marks it as used, and issues a new "
        "access token + refresh token in the same family. "
        "If a previously-used token is replayed, the entire token family is revoked. "
        "Returns HTTP 401 on any validation failure."
    ),
)
def refresh(body: RefreshRequest, db: Session = Depends(get_auth_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid refresh token",
        headers={"WWW-Authenticate": "Bearer"},
    )

    rt = get_refresh_token_by_hash(db, hash_token(body.refresh_token))
    if rt is None:
        raise credentials_exception
    if rt.revoked:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been revoked",
        )
    if rt.used:
        revoke_family(db, rt.family_id)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token reuse detected — all sessions revoked",
        )
    if rt.expires_at < datetime.now(timezone.utc):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token expired",
        )

    mark_token_used(db, rt.id)

    user = get_user_by_id(db, rt.user_id)
    if user is None or not user.is_active:
        raise credentials_exception

    raw_rt = generate_refresh_token()
    create_refresh_token(
        db,
        user_id=user.id,
        family_id=rt.family_id,
        token_hash=hash_token(raw_rt),
        expires_at=datetime.now(timezone.utc) + timedelta(days=_REFRESH_EXPIRE_DAYS),
    )
    return Token(access_token=create_access_token({"sub": user.username}), refresh_token=raw_rt)


@router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Revoke a refresh token",
    description=(
        "Marks the supplied refresh token as revoked so it cannot be used again. "
        "Always returns HTTP 204 regardless of whether the token existed — "
        "this prevents leaking token existence to an attacker."
    ),
)
def logout(body: LogoutRequest, db: Session = Depends(get_auth_db)):
    rt = get_refresh_token_by_hash(db, hash_token(body.refresh_token))
    if rt is not None and not rt.revoked:
        revoke_token(db, rt.id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get the current authenticated user",
    description=(
        "Returns the profile of the user identified by the Bearer token in the "
        "Authorization header. Returns HTTP 401 if the token is missing, expired, or invalid."
    ),
)
def me(current_user: User = Depends(get_current_user)):
    return UserResponse.model_validate(current_user)