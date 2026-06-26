from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from ai_agentic_chatbot.auth.dependencies import get_auth_db, get_current_user
from ai_agentic_chatbot.auth.jwt_utils import create_access_token
from ai_agentic_chatbot.auth.models import User
from ai_agentic_chatbot.auth.schemas import Token, UserCreate, UserResponse
from ai_agentic_chatbot.auth.service import authenticate_user, create_user_account

router = APIRouter(prefix="/auth", tags=["Auth"])


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
    summary="Login and obtain a JWT access token",
    description=(
        "Authenticates with username and password (form data). "
        "Returns a Bearer JWT token valid for the configured expiry period. "
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
    return Token(access_token=access_token)


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