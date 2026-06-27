# User Authentication Module — Implementation Guide

## Step 1: Add Dependencies to `pyproject.toml`

**1.1 — Open `pyproject.toml`** and add these 4 lines to the `dependencies` list after the last entry (`"pandas-stubs..."`):

```toml
"python-jose[cryptography] (>=3.3.0,<4.0.0)",
"passlib[bcrypt] (>=1.7.4,<2.0.0)",
"python-multipart (>=0.0.9,<1.0.0)",
"alembic (>=1.13.0,<2.0.0)"
```

Your `dependencies` block should end like this:

```toml
    "sqlparse (>=0.4.0,<1.0.0)",
    "pandas-stubs (>=2.3.3,<2.4.0)",
    "python-jose[cryptography] (>=3.3.0,<4.0.0)",
    "passlib[bcrypt] (>=1.7.4,<2.0.0)",
    "python-multipart (>=0.0.9,<1.0.0)",
    "alembic (>=1.13.0,<2.0.0)"
]
```

> Note: no trailing comma on the last entry.

---

**1.2 — Install the packages.** Run this in your terminal from the project root:

```bash
pip install "python-jose[cryptography]>=3.3.0,<4.0.0" "passlib[bcrypt]>=1.7.4,<2.0.0" "python-multipart>=0.0.9,<1.0.0" "alembic>=1.13.0,<2.0.0"
```

> If you are using Poetry: `poetry install` instead.

---

**1.3 — Verify the installs worked:**

```bash
python -c "import jose; import passlib; import multipart; import alembic; print('All OK')"
```

You should see `All OK`. If any import fails, re-run the install for that package.

---

Once you see `All OK`, Step 1 is complete. Move on to Step 2.

---

## Step 2: Add JWT Environment Variables

> `.env` is your real secrets file (never committed). `.env.example` is the committed template with empty values.

**2.1 — Generate a secure `JWT_SECRET_KEY`.** Run this in your terminal:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

Copy the output — you will use it in the next step.

---

**2.2 — Open `.env`** and add this block at the bottom:

```
# JWT Authentication
JWT_SECRET_KEY=<paste the value generated above>
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=60

# First superuser seed (only needed for initial seed run, remove after)
SEED_USERNAME=
SEED_EMAIL=
SEED_PASSWORD=
```

---

**2.3 — Open `.env.example`** and add this block at the bottom (leave `JWT_SECRET_KEY` empty — it is a template):

```
# JWT Authentication
JWT_SECRET_KEY=
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=60

# First superuser seed (only needed for initial seed run, remove after)
SEED_USERNAME=
SEED_EMAIL=
SEED_PASSWORD=
```

---

**2.4 — Verify the vars are loading correctly.** Run this from the project root:

```bash
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print(os.environ.get('JWT_SECRET_KEY', 'MISSING'))"
```

You should see your generated key printed (not `MISSING`).

---

Once you see the key printed, Step 2 is complete. Move on to Step 3.

---

## Step 3: Initialise Alembic and Configure `alembic.ini`

**3.1 — Run Alembic init** from the project root in your terminal:

```bash
alembic init alembic
```

This creates the following structure:

```
ai-agentic-chatbot/
├── alembic.ini          ← new config file
└── alembic/
    ├── env.py           ← migration runner (we edit this in Step 6)
    ├── script.py.mako   ← migration file template
    └── versions/        ← generated migration files go here
```

---

**3.2 — Edit `alembic.ini`** in the project root. Find this line:

```ini
sqlalchemy.url = driver://user:pass@localhost/dbname
```

Replace it with:

```ini
# URL is set dynamically in alembic/env.py from environment variables — do not hard-code here
sqlalchemy.url =
```

That is the only change needed in `alembic.ini`. Everything else (script_location, file_template, etc.) stays as-is.

---

**3.3 — Verify the structure was created correctly.** Run:

```bash
python -c "import alembic; print('Alembic OK')"
```

Also confirm the `alembic/` folder and `alembic.ini` exist at the project root.

---

Once the folder structure is in place and `alembic.ini` is updated, Step 3 is complete. Move on to Step 4.

---

## Step 4: Create the Shared SQLAlchemy `Base`

**This step was completed automatically.** The file was created at:

```
src/ai_agentic_chatbot/infrastructure/database.py
```

Contents:

```python
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass
```

**Why this file exists:** The project currently uses only raw SQL — there is no ORM `DeclarativeBase` anywhere. Every ORM model (starting with `User` in Step 5) must inherit from a single shared `Base` so that Alembic's `autogenerate` can detect the full schema in one place. All future ORM models import `Base` from this file.

---

**4.1 — Verify it imports correctly:**

```bash
python -c "from ai_agentic_chatbot.infrastructure.database import Base; print('Base OK')"
```

You should see `Base OK`.

---

Once verified, Step 4 is complete. Move on to Step 5.

---

## Step 5: Create the `User` ORM Model

**This step was completed manually.** Two files were created:

```
src/ai_agentic_chatbot/auth/__init__.py   ← empty, makes auth a Python package
src/ai_agentic_chatbot/auth/models.py     ← User ORM model
```

Contents of `models.py`:

```python
from datetime import datetime
from typing import Optional

from sqlalchemy import BigInteger, Boolean, DateTime, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from ai_agentic_chatbot.infrastructure.database import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(Text, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    is_superuser: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True, onupdate=func.now()
    )
```

**Verification:**

```bash
python -c "from ai_agentic_chatbot.auth.models import User; print('User model OK')"
```

---

Once verified, Step 5 is complete. Move on to Step 6.

---

## Step 6: Wire `alembic/env.py`

**This step was completed automatically.** `alembic/env.py` was rewritten to:

- Load `.env` via `load_dotenv()` so env vars are available
- Import `Base` from `infrastructure/database.py`
- Import `auth.models` to register `User` with `Base.metadata` (this is what allows `autogenerate` to detect the `users` table)
- Build the PostgreSQL URL dynamically from `POSTGRESQL_*` env vars — no credentials hard-coded
- Set `target_metadata = Base.metadata`

Key section added:

```python
from dotenv import load_dotenv
load_dotenv()

from ai_agentic_chatbot.infrastructure.database import Base
import ai_agentic_chatbot.auth.models  # noqa: F401

target_metadata = Base.metadata

def get_url() -> str:
    user = os.environ["POSTGRESQL_USER"]
    password = os.environ["POSTGRESQL_PASSWORD"]
    host = os.environ["POSTGRESQL_HOST"]
    port = os.environ.get("POSTGRESQL_PORT", "5432")
    db = os.environ["POSTGRESQL_DB"]
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"
```

---

**6.1 — Verify Alembic can see the User model** by running:

```bash
alembic check
```

If it returns `No new upgrade operations detected` or lists the `users` table as a pending change, the wiring is correct. If it errors on imports, check that you are running from the project root with the virtualenv active.

---

Once verified, Step 6 is complete. Move on to Step 7.

---

## Step 7: Generate and Apply the Migration

**7.1 — Generate the migration file.** Run this from the project root:

```bash
alembic revision --autogenerate -m "create users table"
```

This creates a new file inside `alembic/versions/` named something like:
`xxxxxxxxxxxx_create_users_table.py`

---

**7.2 — Inspect the generated file** before applying it. Open the file in `alembic/versions/` and confirm the `upgrade()` function contains only:

- `op.create_table('users', ...)` with all the expected columns
- `op.create_index('ix_users_username', 'users', ['username'], unique=True)`

And the `downgrade()` function contains:

- `op.drop_index('ix_users_username', table_name='users')`
- `op.drop_table('users')`

There should be **no** `drop_table` calls for `orders`, `sales`, `customer`, or any other business table. If you see any, stop and let me know before proceeding.

---

**7.3 — Apply the migration** to create the `users` table in the database:

```bash
alembic upgrade head
```

You should see output like:

```
INFO  [alembic.runtime.migration] Running upgrade  -> xxxxxxxxxxxx, create users table
```

---

**7.4 — Verify the table was created.** Run:

```bash
python -c "
from dotenv import load_dotenv; load_dotenv()
from ai_agentic_chatbot.infrastructure.datasource.datasource_init import initialize_datasources
from ai_agentic_chatbot.infrastructure.datasource.factory import get_engine
from sqlalchemy import text
initialize_datasources()
engine = get_engine('postgresql.primary')
with engine.connect() as conn:
    result = conn.execute(text(\"SELECT column_name FROM information_schema.columns WHERE table_name='users' ORDER BY ordinal_position\"))
    print([row[0] for row in result])
"
```

Expected output:

```
['id', 'username', 'email', 'hashed_password', 'is_active', 'is_superuser', 'created_at', 'updated_at']
```

---

Once you see all 8 columns listed, Step 7 is complete. The `users` table is live in the database. Move on to Step 8.

---

## Step 8: Password Hashing Utilities

**This step was completed automatically.** The file was created at:

```
src/ai_agentic_chatbot/auth/password.py
```

Contents:

```python
import bcrypt


def hash_password(plain_password: str) -> str:
    return bcrypt.hashpw(plain_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
```

- `hash_password()` — encodes the password to bytes, hashes with bcrypt + random salt, returns the hash as a string.
- `verify_password()` — constant-time comparison between plain text and stored hash. Called on login.
- Uses `bcrypt` directly instead of `passlib` — `passlib` has a known incompatibility with `bcrypt >= 4.0` which is already installed as a transitive dependency in this project.

---

**8.1 — Verify it works:**

```bash
python -c "
from ai_agentic_chatbot.auth.password import hash_password, verify_password
hashed = hash_password('testpass123')
print('Hash OK:', hashed[:20], '...')
print('Verify correct:', verify_password('testpass123', hashed))
print('Verify wrong:  ', verify_password('wrongpass', hashed))
"
```

Expected output:

```
Hash OK: $2b$12$... 
Verify correct: True
Verify wrong:   False
```

---

Once verified, Step 8 is complete. Move on to Step 9.

---

## Step 9: JWT Utilities

**This step was completed automatically.** The file was created at:

```
src/ai_agentic_chatbot/auth/jwt_utils.py
```

Contents:

```python
import os
from datetime import datetime, timedelta, timezone

from jose import jwt, JWTError

_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "")
_ALGORITHM = os.environ.get("JWT_ALGORITHM", "HS256")
_EXPIRE_MINUTES = int(os.environ.get("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "60"))


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    payload = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=_EXPIRE_MINUTES))
    payload["exp"] = expire
    return jwt.encode(payload, _SECRET_KEY, algorithm=_ALGORITHM)


def decode_access_token(token: str) -> dict:
    return jwt.decode(token, _SECRET_KEY, algorithms=[_ALGORITHM])
```

- `create_access_token()` — copies the input data, adds an `exp` (expiry) claim, signs and returns the JWT string.
- `decode_access_token()` — verifies the signature and expiry, returns the payload dict. Raises `jose.JWTError` on any failure (expired, tampered, invalid) — the caller (FastAPI dependency in Step 14) translates this into an HTTP 401.
- Config is read from env vars at module load time — `load_dotenv()` in `server.py` ensures they are available before any import.

---

**9.1 — Verify it works:**

```bash
python -c "
from dotenv import load_dotenv; load_dotenv()
from ai_agentic_chatbot.auth.jwt_utils import create_access_token, decode_access_token
from jose import JWTError

token = create_access_token({'sub': 'testuser'})
print('Token created OK:', token[:30], '...')

payload = decode_access_token(token)
print('Decoded sub:', payload.get('sub'))

try:
    decode_access_token('invalid.token.here')
except JWTError as e:
    print('Invalid token caught OK:', type(e).__name__)
"
```

Expected output:

```
Token created OK: eyJhbGciOiJIUzI1NiIsInR5cCI6...
Decoded sub: testuser
Invalid token caught OK: JWTError
```

---

Once verified, Step 9 is complete. Move on to Step 10.

---

## Step 10: Pydantic Schemas

**This step was completed automatically.** The file was created at:

```
src/ai_agentic_chatbot/auth/schemas.py
```

Contents:

```python
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
    token_type: str = "bearer"


class TokenData(BaseModel):
    username: str | None = None
```

- `UserCreate` — request body for `POST /auth/register`. Contains plain `password` (never stored).
- `UserResponse` — returned by `POST /auth/register` and `GET /auth/me`. `from_attributes=True` allows it to be built directly from a SQLAlchemy `User` ORM object via `UserResponse.model_validate(user)`.
- `Token` — login response body. `token_type` defaults to `"bearer"` always.
- `TokenData` — internal use only; holds the `sub` claim extracted from the decoded JWT inside the FastAPI dependency (Step 14).

---

**10.1 — Verify all schemas load correctly:**

```bash
python -c "
from ai_agentic_chatbot.auth.schemas import UserCreate, UserResponse, Token, TokenData
print('UserCreate fields :', list(UserCreate.model_fields.keys()))
print('UserResponse fields:', list(UserResponse.model_fields.keys()))
print('Token fields       :', list(Token.model_fields.keys()))
print('TokenData fields   :', list(TokenData.model_fields.keys()))
print('All schemas OK')
"
```

Expected output:

```
UserCreate fields : ['username', 'email', 'password']
UserResponse fields: ['id', 'username', 'email', 'is_active', 'is_superuser', 'created_at']
Token fields       : ['access_token', 'token_type']
TokenData fields   : ['username']
All schemas OK
```

---

Once verified, Step 10 is complete. Move on to Step 11.

---

## Step 11: User Repository

**This step was completed automatically.** The file was created at:

```
src/ai_agentic_chatbot/auth/repository.py
```

Contents:

```python
from sqlalchemy import select
from sqlalchemy.orm import Session

from ai_agentic_chatbot.auth.models import User


def get_user_by_username(session: Session, username: str) -> User | None:
    return session.execute(select(User).where(User.username == username)).scalar_one_or_none()


def get_user_by_email(session: Session, email: str) -> User | None:
    return session.execute(select(User).where(User.email == email)).scalar_one_or_none()


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
```

- `get_user_by_username` / `get_user_by_email` — SQLAlchemy 2.x `select()` style queries. Returns `None` if not found (no exception raised).
- `create_user` — keyword-only args (`*`) prevent accidentally passing `hashed_password` as a positional arg. Commits and refreshes so the returned `User` has its `id` and `created_at` populated from the DB.
- All functions take a plain `Session` as first arg — the session is provided by the FastAPI dependency `get_auth_db()` added in Step 13.

---

**11.1 — Verify the repository imports correctly:**

```bash
python -c "
from ai_agentic_chatbot.auth.repository import get_user_by_username, get_user_by_email, create_user
print('Repository imports OK')
import inspect
print('get_user_by_username params:', list(inspect.signature(get_user_by_username).parameters.keys()))
print('get_user_by_email params   :', list(inspect.signature(get_user_by_email).parameters.keys()))
print('create_user params         :', list(inspect.signature(create_user).parameters.keys()))
"
```

Expected output:

```
Repository imports OK
get_user_by_username params: ['session', 'username']
get_user_by_email params   : ['session', 'email']
create_user params         : ['session', 'username', 'email', 'hashed_password', 'is_superuser']
```

---

Once verified, Step 11 is complete. Move on to Step 12.

---

## Step 12: Auth Service

**This step was completed automatically.** The file was created at:

```
src/ai_agentic_chatbot/auth/service.py
```

Contents:

```python
from sqlalchemy.orm import Session

from ai_agentic_chatbot.auth.models import User
from ai_agentic_chatbot.auth.password import hash_password, verify_password
from ai_agentic_chatbot.auth.repository import (
    create_user, get_user_by_email, get_user_by_username,
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
```

- `authenticate_user` — fetches the user by username, then calls `verify_password()`. Returns `None` for both "user not found" and "wrong password" — intentionally the same response to prevent username enumeration attacks.
- `create_user_account` — checks for duplicate username and email before creating. Raises `ValueError` on conflict; the router (Step 15) catches this and returns HTTP 409.

---

**12.1 — Verify the service imports correctly:**

```bash
python -c "
from ai_agentic_chatbot.auth.service import authenticate_user, create_user_account
print('Service imports OK')
import inspect
print('authenticate_user params  :', list(inspect.signature(authenticate_user).parameters.keys()))
print('create_user_account params:', list(inspect.signature(create_user_account).parameters.keys()))
"
```

Expected output:

```
Service imports OK
authenticate_user params  : ['session', 'username', 'password']
create_user_account params: ['session', 'user_data']
```

---

Once verified, Step 12 is complete. Move on to Step 13.

---

## Step 13: `get_auth_db()` FastAPI Dependency

**This step was completed automatically.** The file was created at:

```
src/ai_agentic_chatbot/auth/dependencies.py
```

Contents:

```python
from collections.abc import Generator

from sqlalchemy.orm import Session

from ai_agentic_chatbot.infrastructure.datasource.factory import get_session


def get_auth_db() -> Generator[Session, None, None]:
    session = get_session("postgresql.primary")
    try:
        yield session
    finally:
        session.close()
```

**Why this file exists:** The existing `get_db_session()` in `infrastructure/db_depency.py` returns a dict of sessions designed for the chatbot — it cannot be used for auth routes. The auth repository functions (`get_user_by_username`, `create_user`, etc.) all take a plain `Session` as their first argument. `get_auth_db()` is a FastAPI generator dependency that:

1. Opens a `Session` from the existing `DataSourceFactory` (same PostgreSQL engine — no new connection pool).
2. Yields the session to the route function via `Depends(get_auth_db)`.
3. Closes the session in a `finally` block after the route returns or raises — guaranteeing no session leaks.

The `Generator[Session, None, None]` type hint means: yields `Session`, accepts no `.send()` values, returns `None`. FastAPI detects the `yield` and automatically splits execution: before `yield` at request start, after `yield` in the `finally` at request end.

Routes in Step 15 will use it like this:

```python
@router.post("/register")
def register(user_data: UserCreate, db: Session = Depends(get_auth_db)):
    return create_user_account(db, user_data)
```

---

**13.1 — Verify the dependency imports correctly:**

```bash
python -c "
from ai_agentic_chatbot.auth.dependencies import get_auth_db
import inspect
print('get_auth_db is a generator function:', inspect.isgeneratorfunction(get_auth_db))
print('Dependency import OK')
"
```

Expected output:

```
get_auth_db is a generator function: True
Dependency import OK
```

---

**13.2 — Verify it opens and closes a real session:**

```bash
python -c "
from dotenv import load_dotenv; load_dotenv()
from ai_agentic_chatbot.infrastructure.datasource.datasource_init import initialize_datasources
initialize_datasources()

from ai_agentic_chatbot.auth.dependencies import get_auth_db
from sqlalchemy import text

gen = get_auth_db()
session = next(gen)
result = session.execute(text('SELECT 1')).scalar()
print('Session query result:', result)

try:
    next(gen)
except StopIteration:
    pass
print('Session closed OK')
"
```

Expected output:

```
Session query result: 1
Session closed OK
```

---

Once verified, Step 13 is complete. Move on to Step 14.

---

## Step 14: `get_current_user()` FastAPI Dependency

**This step was completed automatically.** `get_current_user()` was added to:

```
src/ai_agentic_chatbot/auth/dependencies.py
```

Full updated contents:

```python
from collections.abc import Generator

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError
from sqlalchemy.orm import Session

from ai_agentic_chatbot.infrastructure.datasource.factory import get_session
from ai_agentic_chatbot.auth.jwt_utils import decode_access_token
from ai_agentic_chatbot.auth.models import User
from ai_agentic_chatbot.auth.repository import get_user_by_username
from ai_agentic_chatbot.auth.schemas import TokenData

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def get_auth_db() -> Generator[Session, None, None]:
    session = get_session("postgresql.primary")
    try:
        yield session
    finally:
        session.close()


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_auth_db),
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = decode_access_token(token)
        username: str | None = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception

    user = get_user_by_username(db, token_data.username)
    if user is None:
        raise credentials_exception
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user",
        )
    return user
```

**How it works end-to-end:**

1. `OAuth2PasswordBearer(tokenUrl="/auth/login")` — tells FastAPI to look for a `Authorization: Bearer <token>` header. FastAPI also uses `tokenUrl` to wire up the Swagger UI "Authorize" button.
2. `decode_access_token(token)` — verifies the JWT signature and expiry. Raises `JWTError` on any failure (expired, tampered, malformed).
3. `payload.get("sub")` — extracts the username from the `sub` claim set during login. Returns `None` if missing → raises 401.
4. `get_user_by_username(db, ...)` — loads the user from the database. If the account was deleted after the token was issued, this returns `None` → raises 401.
5. `user.is_active` check — blocks deactivated accounts even if they have a valid token.
6. Returns the `User` ORM object — routes declare `current_user: User = Depends(get_current_user)` and receive the full user object.

**Security note:** Both "user not found" and "bad token" return the same 401 with `"Could not validate credentials"` — this prevents leaking whether a username exists.

---

**14.1 — Verify the dependency imports correctly:**

```bash
python -c "
from dotenv import load_dotenv; load_dotenv()
from ai_agentic_chatbot.auth.dependencies import get_auth_db, get_current_user, oauth2_scheme
import inspect
print('get_auth_db is generator:', inspect.isgeneratorfunction(get_auth_db))
print('get_current_user is callable:', callable(get_current_user))
print('oauth2_scheme tokenUrl:', oauth2_scheme.model.flow.tokenUrl)
print('Step 14 imports OK')
"
```

Expected output:

```
get_auth_db is generator: True
get_current_user is callable: True
oauth2_scheme tokenUrl: /auth/login
Step 14 imports OK
```

---

Once verified, Step 14 is complete. Move on to Step 15.

---

## Step 15: Auth Router

**This step was completed automatically.** The file was created at:

```
src/ai_agentic_chatbot/auth/router.py
```

Contents:

```python
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
)
def me(current_user: User = Depends(get_current_user)):
    return UserResponse.model_validate(current_user)
```

**The three endpoints:**

| Method | Path | Auth required | What it does |
|--------|------|---------------|--------------|
| `POST` | `/auth/register` | No | Creates a user account. Body: `{ username, email, password }`. Returns `UserResponse` (201). Raises 409 on duplicate username/email. |
| `POST` | `/auth/login` | No | Accepts `application/x-www-form-urlencoded` with `username` + `password` fields (OAuth2 standard form). Returns `{ access_token, token_type }`. Raises 401 on bad credentials. |
| `GET`  | `/auth/me` | Yes — Bearer token | Returns the profile of the currently authenticated user. Raises 401 if token missing/invalid/expired. |

**Why `/login` uses form data, not JSON:** `OAuth2PasswordRequestForm` is the OAuth2 spec standard — it expects `Content-Type: application/x-www-form-urlencoded`. This is what Swagger UI's "Authorize" button sends, and what most OAuth2 clients expect.

---

**15.1 — Verify the router imports correctly:**

```bash
python -c "
from dotenv import load_dotenv; load_dotenv()
from ai_agentic_chatbot.auth.router import router
print('Router prefix :', router.prefix)
print('Router tags   :', router.tags)
routes = [(r.methods, r.path) for r in router.routes]
print('Routes:')
for methods, path in routes:
    print(' ', sorted(methods), path)
print('Router import OK')
"
```

Expected output:

```
Router prefix : /auth
Router tags   : ['Auth']
Routes:
  ['POST'] /register
  ['POST'] /login
  ['GET'] /me
Router import OK
```

---

Once verified, Step 15 is complete. Move on to Step 16.

---

## Step 16: Wire the Auth Router into `server.py`

This step mounts the auth router onto the FastAPI app so the three `/auth/*` endpoints are live.

**Two changes to `src/ai_agentic_chatbot/server.py`:**

**16.1 — Add the import.** Find the existing import block at the top of `server.py` and add this line alongside the other local imports:

```python
from ai_agentic_chatbot.auth.router import router as auth_router
```

**16.2 — Register the router.** Add this line immediately after the `app = FastAPI(...)` block (after the closing parenthesis, before the first `@app.get`):

```python
app.include_router(auth_router)
```

After both changes the relevant section of `server.py` should look like this:

```python
from ai_agentic_chatbot.auth.router import router as auth_router  # ← added

# ... other imports ...

app = FastAPI(
    title="AI Chat Application",
    version="1.0.0",
    description="Agent enabled AI ChatBot application",
    lifespan=lifespan,
)

app.include_router(auth_router)  # ← added

@app.get("/health", ...)
```

---

**16.3 — Verify the routes are registered on the app:**

```bash
python -c "
from dotenv import load_dotenv; load_dotenv()
from ai_agentic_chatbot.server import app
auth_routes = [r.path for r in app.routes if hasattr(r, 'path') and r.path.startswith('/auth')]
print('Auth routes on app:', auth_routes)
assert '/auth/register' in auth_routes
assert '/auth/login' in auth_routes
assert '/auth/me' in auth_routes
print('Step 16 OK')
"
```

Expected output:

```
Auth routes on app: ['/auth/register', '/auth/login', '/auth/me']
Step 16 OK
```

---

Once verified, Step 16 is complete. The auth module is fully wired. Move on to Step 17.

---

## Step 17: End-to-End Live Test

This step starts the server and exercises the full auth flow against a running app — register → login → get profile → error cases.

---

**17.1 — Start the server.**

Open a separate terminal and run:

```bash
python -m uvicorn ai_agentic_chatbot.server:app --host 0.0.0.0 --port 8000 --reload
```

Wait until you see:

```
INFO:     Application startup complete.
```

Leave that terminal running and open a new one for the tests below.

---

**17.2 — Run the full end-to-end test script.**

Run this from the project root (server must be running):

```bash
python -c "
import requests

BASE = 'http://localhost:8000'

# --- Register ---
r = requests.post(f'{BASE}/auth/register', json={
    'username': 'testuser',
    'email': 'testuser@example.com',
    'password': 'SecurePass123'
})
print('Register status :', r.status_code)          # expect 201
print('Register body   :', r.json())

# --- Duplicate register (expect 409) ---
r = requests.post(f'{BASE}/auth/register', json={
    'username': 'testuser',
    'email': 'testuser@example.com',
    'password': 'SecurePass123'
})
print('Duplicate status:', r.status_code)          # expect 409

# --- Login ---
r = requests.post(f'{BASE}/auth/login', data={
    'username': 'testuser',
    'password': 'SecurePass123'
})
print('Login status    :', r.status_code)          # expect 200
token = r.json()['access_token']
print('Token received  :', token[:30], '...')

# --- /me with valid token ---
r = requests.get(f'{BASE}/auth/me', headers={'Authorization': f'Bearer {token}'})
print('/me status      :', r.status_code)          # expect 200
print('/me body        :', r.json())

# --- Wrong password (expect 401) ---
r = requests.post(f'{BASE}/auth/login', data={
    'username': 'testuser',
    'password': 'wrongpassword'
})
print('Bad login status:', r.status_code)          # expect 401

# --- /me with bad token (expect 401) ---
r = requests.get(f'{BASE}/auth/me', headers={'Authorization': 'Bearer invalidtoken'})
print('Bad token status:', r.status_code)          # expect 401

print()
print('All checks passed.')
"
```

Expected output:

```
Register status : 201
Register body   : {'id': 1, 'username': 'testuser', 'email': 'testuser@example.com', 'is_active': True, 'is_superuser': False, 'created_at': '...'}
Duplicate status: 409
Login status    : 200
Token received  : eyJhbGciOiJIUzI1NiIsInR5cCI6...
/me status      : 200
/me body        : {'id': 1, 'username': 'testuser', 'email': 'testuser@example.com', 'is_active': True, 'is_superuser': False, 'created_at': '...'}
Bad login status: 401
Bad token status: 401

All checks passed.
```

---

**17.3 — Optional: Swagger UI manual test.**

Open `http://localhost:8000/docs` in your browser. You will see three new endpoints under the **Auth** tag:

- `POST /auth/register`
- `POST /auth/login`
- `GET /auth/me`

To test via Swagger:
1. Call `POST /auth/register` with a username, email, and password.
2. Call `POST /auth/login` with the same credentials — copy the `access_token` from the response.
3. Click the **Authorize** button (top right), paste the token, click Authorize.
4. Call `GET /auth/me` — it should return your user profile.

---

Once all status codes match expectations, Step 17 is complete. The authentication module is fully implemented and live. Move on to Step 18.

---

## Step 18: Superuser Seed Script

**This step was completed automatically.** The file was created at:

```
src/ai_agentic_chatbot/auth/seed.py
```

**Why this step exists:** All four main endpoints (`/stream`, `/schemaJson`, `/schemaText`, `/ingest`) now require a valid JWT. To obtain a token you must log in. To log in you must have an account. The seed script creates the first superuser account from env vars — a safe, one-time operation that avoids hard-coding credentials anywhere in the codebase.

The `SEED_USERNAME`, `SEED_EMAIL`, and `SEED_PASSWORD` vars were reserved in `.env` back in Step 2 for exactly this purpose.

---

**18.1 — Fill in the seed vars in `.env`.**

Open `.env` and set the three seed values:

```
SEED_USERNAME=admin
SEED_EMAIL=admin@example.com
SEED_PASSWORD=<choose a strong password>
```

---

**18.2 — Run the seed script:**

```bash
python -m ai_agentic_chatbot.auth.seed
```

Expected output:

```
Superuser created: id=1  username=admin  email=admin@example.com
Remove SEED_USERNAME / SEED_EMAIL / SEED_PASSWORD from .env now.
```

If the user already exists (e.g., you re-run it), the script exits safely:

```
User 'admin' already exists — nothing to do.
```

---

**18.3 — Remove the seed vars from `.env`.**

After a successful run, blank out or delete the three lines in `.env`:

```
SEED_USERNAME=
SEED_EMAIL=
SEED_PASSWORD=
```

This prevents the credentials from sitting in the file unnecessarily.

---

**18.4 — Verify the superuser can log in and hit a protected endpoint:**

```bash
python -c "
import requests

BASE = 'http://localhost:8000'

r = requests.post(f'{BASE}/auth/login', data={
    'username': 'admin',
    'password': '<your seed password>'
})
print('Login status:', r.status_code)   # expect 200
token = r.json()['access_token']

r = requests.get(f'{BASE}/auth/me', headers={'Authorization': f'Bearer {token}'})
print('/me body    :', r.json())
print('is_superuser:', r.json()['is_superuser'])   # expect True
"
```

Expected output:

```
Login status: 200
/me body    : {'id': ..., 'username': 'admin', 'email': 'admin@example.com', 'is_active': True, 'is_superuser': True, 'created_at': '...'}
is_superuser: True
```

---

Once verified, Step 18 is complete. The auth module implementation is finished. Move on to Step 19.

---

## Step 19: Commit the Auth Module

This step stages and commits all remaining auth module files.

> **Warning — do NOT stage `config.yaml` as-is.** It currently contains live Azure OpenAI API keys in plaintext. Blank them out before committing (step 19.1 below).

---

**19.1 — Remove the API keys from `config.yaml`.**

Open `config.yaml` and replace every `api_key:` value that has a real key with an empty string. The committed file should look like:

```yaml
    fast:
      api_key: ""
    smart:
      api_key: ""
    embedding:
      api_key: ""
```

Also blank out any `host`, `username`, `password` fields under `datasources` if they contain real values. The file is safe to commit once all secrets are empty strings.

---

**19.2 — Stage the files:**

```bash
git add docs/auth-module-implementation.md
git add src/ai_agentic_chatbot/auth/dependencies.py
git add src/ai_agentic_chatbot/auth/router.py
git add src/ai_agentic_chatbot/auth/seed.py
git add src/ai_agentic_chatbot/server.py
git add config.yaml
git add certs/
```

---

**19.3 — Commit:**

```bash
git commit -m "feat: complete user authentication module with JWT-secured APIs

- Add FastAPI dependencies: get_auth_db (session lifecycle) and
  get_current_user (JWT Bearer validation with 401 on failure)
- Add auth router with POST /auth/register, POST /auth/login,
  GET /auth/me; login uses OAuth2PasswordRequestForm (form data)
- Protect POST /stream, GET /schemaJson, GET /schemaText,
  GET /ingest with get_current_user dependency
- Add seed script (auth/seed.py) to create the first superuser
  from SEED_* env vars; idempotent and exits safely if user exists
- Complete implementation guide (steps 13–18) in docs/
- Add config.yaml with model and datasource configuration (secrets blanked)"
```

---

**19.4 — Verify the commit:**

```bash
git log --oneline -3
git show --stat HEAD
```

You should see the new commit at the top listing all the staged files.

---

Once verified, Step 19 is complete. Move on to Step 20.

---

## Step 20: Add `RefreshToken` ORM Model

**This step was completed automatically.** `auth/models.py` was updated with two changes:

**New imports added:**
```python
import uuid
from sqlalchemy import ForeignKey          # added
from sqlalchemy.dialects.postgresql import UUID  # added
```

**New class added after `User`:**

```python
class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    family_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)
    used: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    revoked: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
```

**Column-by-column rationale:**

| Column | Type | Purpose |
|---|---|---|
| `id` | UUID PK | Random UUID4 — avoids sequential ID enumeration attacks |
| `user_id` | BigInt FK → users.id | `CASCADE DELETE` so tokens are cleaned up when a user is deleted; indexed for family revocation queries |
| `token_hash` | String(64), unique | SHA-256 hex digest of the raw token (always 64 chars); unique index enables O(log n) lookup |
| `family_id` | UUID, indexed | Groups all tokens from one login session; one `UPDATE WHERE family_id = ?` revokes the whole family on replay detection |
| `used` | Boolean | `True` once this token has been rotated away — distinguishes normal rotation from security revocation |
| `revoked` | Boolean | `True` when force-revoked (logout, replay detected, admin action) — checked independently of `used` |
| `expires_at` | DateTime(tz) | Hard expiry; enables periodic cleanup `DELETE FROM refresh_tokens WHERE expires_at < NOW()` |
| `created_at` | DateTime(tz) | Audit trail; server-default so the application never needs to set it |

---

**20.1 — Verify the model imports correctly:**

```bash
python -c "
from ai_agentic_chatbot.auth.models import User, RefreshToken
import inspect
print('User table        :', User.__tablename__)
print('RefreshToken table:', RefreshToken.__tablename__)
print('RefreshToken cols :', [c.name for c in RefreshToken.__table__.columns])
print('Step 20 OK')
"
```

Expected output:

```
User table        : users
RefreshToken table: refresh_tokens
RefreshToken cols : ['id', 'user_id', 'token_hash', 'family_id', 'used', 'revoked', 'expires_at', 'created_at']
Step 20 OK
```

---

Once verified, Step 20 is complete. Move on to Step 21.

---

## Step 21: Generate and Apply the `refresh_tokens` Migration

**No change to `alembic/env.py` is needed.** Line 20 already imports `ai_agentic_chatbot.auth.models`, which contains both `User` and `RefreshToken`. The `include_object` filter in `env.py` automatically picks up any model registered with `Base` — `RefreshToken` is included the moment it was added to `models.py`.

---

**21.1 — Generate the migration:**

```bash
alembic revision --autogenerate -m "create refresh_tokens table"
```

This creates a new file in `alembic/versions/`.

---

**21.2 — Inspect the generated file before applying.**

Open the new file in `alembic/versions/` and confirm:

- `upgrade()` contains only `op.create_table('refresh_tokens', ...)` with all 8 columns plus index creation
- `downgrade()` contains only `op.drop_index(...)` and `op.drop_table('refresh_tokens')`
- There are **no** `op.drop_table` calls for `users`, `orders`, `sales`, `customer`, or any other existing table

If you see unexpected drops, stop and do not proceed.

---

**21.3 — Apply the migration:**

```bash
alembic upgrade head
```

Expected output:

```
INFO  [alembic.runtime.migration] Running upgrade <prev_rev> -> <new_rev>, create refresh_tokens table
```

---

**21.4 — Verify the table was created in the database:**

```bash
python -c "
from dotenv import load_dotenv; load_dotenv()
from ai_agentic_chatbot.infrastructure.datasource.datasource_init import initialize_datasources
from ai_agentic_chatbot.infrastructure.datasource.factory import get_engine
from sqlalchemy import text
initialize_datasources()
engine = get_engine('postgresql.primary')
with engine.connect() as conn:
    result = conn.execute(text(\"SELECT column_name, data_type FROM information_schema.columns WHERE table_name='refresh_tokens' ORDER BY ordinal_position\"))
    for row in result:
        print(f'  {row[0]:15s} {row[1]}')
print('Step 21 OK')
"
```

Expected output:

```
  id              uuid
  user_id         bigint
  token_hash      character varying
  family_id       uuid
  used            boolean
  revoked         boolean
  expires_at      timestamp with time zone
  created_at      timestamp with time zone
Step 21 OK
```

---

Once all 8 columns appear, Step 21 is complete. Move on to Step 22.

---

## Step 22: Add `generate_refresh_token()` and `hash_token()` to `jwt_utils.py`

**This step was completed automatically.** Two additions were made to `src/ai_agentic_chatbot/auth/jwt_utils.py`:

**New imports at the top:**
```python
import hashlib
import secrets
```

**Two new functions added after `decode_access_token`:**

```python
def generate_refresh_token() -> str:
    return secrets.token_urlsafe(64)


def hash_token(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()
```

**Why these two functions:**

- `generate_refresh_token()` — `secrets.token_urlsafe(64)` produces 64 bytes (512 bits) of cryptographically secure random data, URL-safe base64-encoded. This is the raw token sent to the client. Never stored in the DB.
- `hash_token()` — SHA-256 of the raw token, returned as a 64-character hex digest. This is what gets stored in `refresh_tokens.token_hash`. If the DB is exfiltrated, attackers get hashes, not usable tokens. SHA-256 (without salt) is safe here because the input has 512 bits of entropy — rainbow tables are infeasible.

**Why not JWT for refresh tokens:** The refresh token is always validated against the DB, so embedding claims in a JWT adds nothing. An opaque random string is simpler, smaller, and cannot be decoded by the client.

---

**22.1 — Verify both functions work correctly:**

```bash
python -c "
from ai_agentic_chatbot.auth.jwt_utils import generate_refresh_token, hash_token

raw = generate_refresh_token()
print('Raw token length  :', len(raw))
print('Raw token sample  :', raw[:30], '...')

hashed = hash_token(raw)
print('Hash length       :', len(hashed))
print('Hash sample       :', hashed[:30], '...')

# Same input always produces the same hash
assert hash_token(raw) == hashed, 'Hash is not deterministic'

# Different tokens produce different hashes
raw2 = generate_refresh_token()
assert hash_token(raw2) != hashed, 'Collision detected'

print('Step 22 OK')
"
```

Expected output:

```
Raw token length  : 86
Raw token sample  : <random base64 string> ...
Hash length       : 64
Hash sample       : <hex string> ...
Step 22 OK
```

> Note: raw token length is 86 characters (64 bytes → base64 encoding adds ~33% overhead).

---

Once verified, Step 22 is complete. Move on to Step 23.

---

## Step 23: Update Pydantic Schemas

**This step was completed automatically.** Three changes were made to `src/ai_agentic_chatbot/auth/schemas.py`:

**1 — `Token` schema updated** — `refresh_token` field added:

```python
class Token(BaseModel):
    access_token: str
    refresh_token: str | None = None   # ← added
    token_type: str = "bearer"
```

`refresh_token` is `None` by default so the existing `/auth/me` response (which returns a `Token`-derived object) is unaffected. It will be populated by the modified `login` endpoint (Step 26) and the new `refresh` endpoint.

**2 — `RefreshRequest` schema added:**

```python
class RefreshRequest(BaseModel):
    refresh_token: str
```

Request body for `POST /auth/refresh`. The raw opaque token string sent by the client.

**3 — `LogoutRequest` schema added:**

```python
class LogoutRequest(BaseModel):
    refresh_token: str
```

Request body for `POST /auth/logout`. Identical structure to `RefreshRequest` — kept as a separate class so each endpoint's intent is explicit and they can diverge independently in future.

---

**23.1 — Verify all schemas load correctly:**

```bash
python -c "
from ai_agentic_chatbot.auth.schemas import Token, RefreshRequest, LogoutRequest, TokenData

print('Token fields       :', list(Token.model_fields.keys()))
print('RefreshRequest     :', list(RefreshRequest.model_fields.keys()))
print('LogoutRequest      :', list(LogoutRequest.model_fields.keys()))

# Token.refresh_token should be optional (default None)
t = Token(access_token='abc')
assert t.refresh_token is None, 'refresh_token should default to None'
assert t.token_type == 'bearer'

# Token with refresh_token
t2 = Token(access_token='abc', refresh_token='xyz')
assert t2.refresh_token == 'xyz'

print('Step 23 OK')
"
```

Expected output:

```
Token fields       : ['access_token', 'refresh_token', 'token_type']
RefreshRequest     : ['refresh_token']
LogoutRequest      : ['refresh_token']
Step 23 OK
```

---

Once verified, Step 23 is complete. Move on to Step 24.

---

## Step 24: Add Refresh Token CRUD to `repository.py`

**This step was completed automatically.** Five new functions were added to `src/ai_agentic_chatbot/auth/repository.py`, along with updated imports:

**New imports:**
```python
import uuid
from datetime import datetime
from sqlalchemy import select, update      # update added
from ai_agentic_chatbot.auth.models import User, RefreshToken  # RefreshToken added
```

**New functions:**

```python
def create_refresh_token(
    session: Session,
    *,
    user_id: int,
    family_id: uuid.UUID,
    token_hash: str,
    expires_at: datetime,
) -> RefreshToken:
    rt = RefreshToken(user_id=user_id, family_id=family_id, token_hash=token_hash, expires_at=expires_at)
    session.add(rt)
    session.commit()
    session.refresh(rt)
    return rt


def get_refresh_token_by_hash(session: Session, token_hash: str) -> RefreshToken | None:
    return session.execute(
        select(RefreshToken).where(RefreshToken.token_hash == token_hash)
    ).scalar_one_or_none()


def mark_token_used(session: Session, token_id: uuid.UUID) -> None:
    session.execute(update(RefreshToken).where(RefreshToken.id == token_id).values(used=True))
    session.commit()


def revoke_family(session: Session, family_id: uuid.UUID) -> None:
    session.execute(update(RefreshToken).where(RefreshToken.family_id == family_id).values(revoked=True))
    session.commit()


def revoke_token(session: Session, token_id: uuid.UUID) -> None:
    session.execute(update(RefreshToken).where(RefreshToken.id == token_id).values(revoked=True))
    session.commit()
```

**What each function does:**

| Function | Called by | Purpose |
|---|---|---|
| `create_refresh_token` | `login` endpoint | Stores a new hashed token row with its family and expiry |
| `get_refresh_token_by_hash` | `refresh` + `logout` endpoints | Single indexed lookup — finds a token by its SHA-256 hash |
| `mark_token_used` | `refresh` endpoint | Marks the incoming token as used before issuing the new one |
| `revoke_family` | `refresh` endpoint (replay detected) | Revokes every token in a family — forces full re-login |
| `revoke_token` | `logout` endpoint | Revokes a single token cleanly on user-initiated logout |

---

**24.1 — Verify all functions import correctly:**

```bash
python -c "
from ai_agentic_chatbot.auth.repository import (
    create_refresh_token, get_refresh_token_by_hash,
    mark_token_used, revoke_family, revoke_token
)
import inspect
fns = [create_refresh_token, get_refresh_token_by_hash, mark_token_used, revoke_family, revoke_token]
for fn in fns:
    print(f'{fn.__name__:30s} params: {list(inspect.signature(fn).parameters.keys())}')
print('Step 24 OK')
"
```

Expected output:

```
create_refresh_token           params: ['session', 'user_id', 'family_id', 'token_hash', 'expires_at']
get_refresh_token_by_hash      params: ['session', 'token_hash']
mark_token_used                params: ['session', 'token_id']
revoke_family                  params: ['session', 'family_id']
revoke_token                   params: ['session', 'token_id']
Step 24 OK
```

---

Once verified, Step 24 is complete. Move on to Step 25.

---

## Step 25: Add `JWT_REFRESH_TOKEN_EXPIRE_DAYS` to `.env`

This step adds the refresh token expiry configuration to your environment files.

---

**25.1 — Open `.env`** and add this line in the `# JWT Authentication` block:

```
JWT_REFRESH_TOKEN_EXPIRE_DAYS=7
```

The block should look like this after the change:

```
# JWT Authentication
JWT_SECRET_KEY=<your key>
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=60
JWT_REFRESH_TOKEN_EXPIRE_DAYS=7
```

---

**25.2 — Open `.env.example`** and add the same line (no value needed in the template):

```
JWT_REFRESH_TOKEN_EXPIRE_DAYS=7
```

---

**25.3 — Verify the var loads correctly:**

```bash
python -c "
from dotenv import load_dotenv; load_dotenv()
import os
val = os.environ.get('JWT_REFRESH_TOKEN_EXPIRE_DAYS', 'MISSING')
print('JWT_REFRESH_TOKEN_EXPIRE_DAYS:', val)
assert val != 'MISSING', 'Variable not found in .env'
assert int(val) == 7, f'Expected 7, got {val}'
print('Step 25 OK')
"
```

Expected output:

```
JWT_REFRESH_TOKEN_EXPIRE_DAYS: 7
Step 25 OK
```

---

Once verified, Step 25 is complete. Move on to Step 26.

---

## Step 26: Update Auth Router — Login, Refresh, Logout

**This step was completed automatically.** `src/ai_agentic_chatbot/auth/router.py` was fully rewritten with the following changes:

**New imports added:**
```python
import os, uuid
from datetime import datetime, timedelta, timezone
from fastapi import Response
from ai_agentic_chatbot.auth.jwt_utils import generate_refresh_token, hash_token
from ai_agentic_chatbot.auth.repository import (
    create_refresh_token, get_refresh_token_by_hash,
    get_user_by_id, mark_token_used, revoke_family, revoke_token,
)
from ai_agentic_chatbot.auth.schemas import LogoutRequest, RefreshRequest
```

**Module-level constant:**
```python
_REFRESH_EXPIRE_DAYS = int(os.environ.get("JWT_REFRESH_TOKEN_EXPIRE_DAYS", "7"))
```

---

**`POST /auth/login` — modified:**

After authenticating, now also generates a refresh token row and returns it alongside the access token:

```python
raw_rt = generate_refresh_token()
create_refresh_token(
    db,
    user_id=user.id,
    family_id=uuid.uuid4(),          # new family per login
    token_hash=hash_token(raw_rt),
    expires_at=datetime.now(timezone.utc) + timedelta(days=_REFRESH_EXPIRE_DAYS),
)
return Token(access_token=access_token, refresh_token=raw_rt)
```

---

**`POST /auth/refresh` — new endpoint:**

Full validation + rotation logic:

1. Hash the incoming token → look up by `token_hash`
2. Not found → 401
3. `revoked == True` → 401 "Token has been revoked"
4. `used == True` → `revoke_family()` → 401 "Token reuse detected" *(replay attack)*
5. `expires_at < now()` → 401 "Refresh token expired"
6. `mark_token_used()` on the old token
7. Look up user via `get_user_by_id()` — 401 if deleted or inactive
8. `create_refresh_token()` with the **same `family_id`** (maintains the session chain)
9. Return new `{ access_token, refresh_token }`

---

**`POST /auth/logout` — new endpoint:**

```python
rt = get_refresh_token_by_hash(db, hash_token(body.refresh_token))
if rt is not None and not rt.revoked:
    revoke_token(db, rt.id)
return Response(status_code=status.HTTP_204_NO_CONTENT)
```

Always returns **HTTP 204** — never reveals whether the token existed (prevents an attacker from probing for valid tokens).

---

**`get_user_by_id` also added to `repository.py`** (needed by `/refresh` to load the user from the token's `user_id`):

```python
def get_user_by_id(session: Session, user_id: int) -> User | None:
    return session.execute(select(User).where(User.id == user_id)).scalar_one_or_none()
```

---

**26.1 — Verify the router imports and all 5 routes are registered:**

```bash
python -c "
from dotenv import load_dotenv; load_dotenv()
from ai_agentic_chatbot.auth.router import router
print('Router prefix:', router.prefix)
routes = [(sorted(r.methods), r.path) for r in router.routes]
print('Routes:')
for methods, path in routes:
    print(' ', methods, path)
print('Step 26 OK')
"
```

Expected output:

```
Router prefix: /auth
Routes:
  ['POST'] /register
  ['POST'] /login
  ['POST'] /refresh
  ['POST'] /logout
  ['GET'] /me
Step 26 OK
```

---

Once verified, Step 26 is complete. Move on to Step 27.

---

## Step 27: End-to-End Live Test — Refresh Token Flow

This step starts the server and exercises the full refresh token lifecycle: login → refresh → logout → replay detection.

---

**27.1 — Start the server** (separate terminal):

```bash
poetry run uvicorn ai_agentic_chatbot.server:app --host 0.0.0.0 --port 8000 --reload
```

Wait for `Application startup complete.`

---

**27.2 — Run the full end-to-end test script** (server must be running):

```bash
python -c "
import requests

BASE = 'http://localhost:8000'
USERNAME = 'admin'
PASSWORD = '<your admin password>'

# --- Login ---
r = requests.post(f'{BASE}/auth/login', data={'username': USERNAME, 'password': PASSWORD})
print('Login status          :', r.status_code)          # expect 200
assert r.status_code == 200
body = r.json()
access_token  = body['access_token']
refresh_token = body['refresh_token']
print('Access token received :', access_token[:30], '...')
print('Refresh token received:', refresh_token[:30], '...')

# --- /me with access token ---
r = requests.get(f'{BASE}/auth/me', headers={'Authorization': f'Bearer {access_token}'})
print('/me status            :', r.status_code)          # expect 200
assert r.status_code == 200

# --- Refresh: get new token pair ---
r = requests.post(f'{BASE}/auth/refresh', json={'refresh_token': refresh_token})
print('Refresh status        :', r.status_code)          # expect 200
assert r.status_code == 200
body2 = r.json()
new_access_token  = body2['access_token']
new_refresh_token = body2['refresh_token']
print('New access token      :', new_access_token[:30], '...')
print('New refresh token     :', new_refresh_token[:30], '...')

# --- /me with new access token ---
r = requests.get(f'{BASE}/auth/me', headers={'Authorization': f'Bearer {new_access_token}'})
print('/me with new token    :', r.status_code)          # expect 200
assert r.status_code == 200

# --- Replay: reuse the OLD refresh token (expect 401 + family revoked) ---
r = requests.post(f'{BASE}/auth/refresh', json={'refresh_token': refresh_token})
print('Replay attack status  :', r.status_code)          # expect 401
print('Replay detail         :', r.json().get('detail'))
assert r.status_code == 401

# --- Login again to get a fresh token for logout test ---
r = requests.post(f'{BASE}/auth/login', data={'username': USERNAME, 'password': PASSWORD})
fresh_rt = r.json()['refresh_token']

# --- Logout ---
r = requests.post(f'{BASE}/auth/logout', json={'refresh_token': fresh_rt})
print('Logout status         :', r.status_code)          # expect 204
assert r.status_code == 204

# --- Refresh after logout (expect 401) ---
r = requests.post(f'{BASE}/auth/refresh', json={'refresh_token': fresh_rt})
print('Post-logout refresh   :', r.status_code)          # expect 401
print('Post-logout detail    :', r.json().get('detail'))
assert r.status_code == 401

print()
print('All checks passed.')
"
```

Expected output:

```
Login status          : 200
Access token received : eyJhbGciOiJIUzI1NiIsInR5cCI6...
Refresh token received: <random string> ...
/me status            : 200
Refresh status        : 200
New access token      : eyJhbGciOiJIUzI1NiIsInR5cCI6...
New refresh token     : <random string> ...
/me with new token    : 200
Replay attack status  : 401
Replay detail         : Token reuse detected — all sessions revoked
Logout status         : 204
Post-logout refresh   : 401
Post-logout detail    : Token has been revoked

All checks passed.
```

---

Once all assertions pass, Step 27 is complete. The refresh token system is fully implemented and live.

---

## Feature 1 — Self-Service Password Update (`PATCH /auth/password`)

This feature lets an authenticated user change their own password by supplying their current password alongside the desired new one. The endpoint validates the current password with bcrypt, rejects passwords shorter than 8 characters, and persists the new bcrypt hash — all without requiring superuser privileges.

---

### Step 1: Edit `auth/schemas.py`

**1.1 — Open** `src/ai_agentic_chatbot/auth/schemas.py`.

**1.2 — Find** the existing import line at the top of the file:

```python
from pydantic import BaseModel, ConfigDict
```

**Replace it with** (add `Field`):

```python
from pydantic import BaseModel, ConfigDict, Field
```

**1.3 — Find** the last class in the file:

```python
class LogoutRequest(BaseModel):
    refresh_token: str
```

**Append** the two new models immediately after it:

```python
class PasswordUpdateRequest(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=8)


class PasswordUpdateResponse(BaseModel):
    message: str
```

---

### Step 2: Edit `auth/repository.py`

**2.1 — Open** `src/ai_agentic_chatbot/auth/repository.py`.

**2.2 — Scroll to the very end of the file** (after `revoke_token`).

**2.3 — Append** the following function:

```python
def update_user_password(db: Session, user: User, new_hashed: str) -> User:
    user.hashed_password = new_hashed
    db.commit()
    db.refresh(user)
    return user
```

> No new imports are needed — `Session` and `User` are already imported at the top of the file.

---

### Step 3: Edit `auth/service.py`

**3.1 — Open** `src/ai_agentic_chatbot/auth/service.py`.

**3.2 — Scroll to the very end of the file** (after `create_user_account`).

**3.3 — Append** the following function:

```python
def change_password(
    db: Session,
    current_user: User,
    current_password: str,
    new_password: str,
) -> User:
    from .password import verify_password, hash_password
    from .repository import update_user_password
    from fastapi import HTTPException
    if not verify_password(current_password, current_user.hashed_password):
        raise HTTPException(status_code=400, detail="Current password is incorrect")
    new_hashed = hash_password(new_password)
    return update_user_password(db, current_user, new_hashed)
```

> The three `from` imports are intentionally local to this function — they avoid circular imports and keep the module-level import list unchanged.

---

### Step 4: Edit `auth/router.py`

**4.1 — Open** `src/ai_agentic_chatbot/auth/router.py`.

**4.2 — Find** the schemas import line:

```python
from ai_agentic_chatbot.auth.schemas import LogoutRequest, RefreshRequest, Token, UserCreate, UserResponse
```

**Replace it with** (add `PasswordUpdateRequest` and `PasswordUpdateResponse`):

```python
from ai_agentic_chatbot.auth.schemas import LogoutRequest, PasswordUpdateRequest, PasswordUpdateResponse, RefreshRequest, Token, UserCreate, UserResponse
```

**4.3 — Find** the service import line:

```python
from ai_agentic_chatbot.auth.service import authenticate_user, create_user_account
```

**Replace it with** (add `change_password`):

```python
from ai_agentic_chatbot.auth.service import authenticate_user, change_password, create_user_account
```

**4.4 — Find** the end of the `GET /me` route handler:

```python
def me(current_user: User = Depends(get_current_user)):
    return UserResponse.model_validate(current_user)
```

**Append** the new route immediately after it:

```python
@router.patch("/password", response_model=PasswordUpdateResponse)
def update_password(
    payload: PasswordUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_auth_db),
):
    change_password(db, current_user, payload.current_password, payload.new_password)
    return PasswordUpdateResponse(message="Password updated successfully")
```

---

### Manual Test

Ensure the server is running (`uvicorn` or however you start it locally), then run:

```bash
python -c "
import requests

BASE = 'http://localhost:8000'
USERNAME = 'admin'
OLD_PASSWORD = '<your current password>'
NEW_PASSWORD = 'newpassword99'

# --- Step 1: Login with current credentials ---
r = requests.post(f'{BASE}/auth/login', data={'username': USERNAME, 'password': OLD_PASSWORD})
print('Login status         :', r.status_code)           # expect 200
assert r.status_code == 200
access_token = r.json()['access_token']
print('Access token         :', access_token[:30], '...')

headers = {'Authorization': f'Bearer {access_token}'}

# --- Step 2: Change password ---
r = requests.patch(
    f'{BASE}/auth/password',
    json={'current_password': OLD_PASSWORD, 'new_password': NEW_PASSWORD},
    headers=headers,
)
print('PATCH /password      :', r.status_code)           # expect 200
print('Response body        :', r.json())
assert r.status_code == 200
assert r.json()['message'] == 'Password updated successfully'

# --- Step 3: Login with the new password ---
r = requests.post(f'{BASE}/auth/login', data={'username': USERNAME, 'password': NEW_PASSWORD})
print('Login (new password) :', r.status_code)           # expect 200
assert r.status_code == 200

# --- Step 4: Confirm old password is rejected ---
r = requests.post(f'{BASE}/auth/login', data={'username': USERNAME, 'password': OLD_PASSWORD})
print('Login (old password) :', r.status_code)           # expect 401
assert r.status_code == 401

# --- Step 5: Confirm wrong current_password is rejected ---
r2 = requests.post(f'{BASE}/auth/login', data={'username': USERNAME, 'password': NEW_PASSWORD})
token2 = r2.json()['access_token']
r = requests.patch(
    f'{BASE}/auth/password',
    json={'current_password': 'wrongpassword', 'new_password': 'doesnotmatter'},
    headers={'Authorization': f'Bearer {token2}'},
)
print('Wrong current pwd    :', r.status_code)           # expect 400
assert r.status_code == 400

# --- Step 6: Confirm short new_password is rejected ---
r = requests.patch(
    f'{BASE}/auth/password',
    json={'current_password': NEW_PASSWORD, 'new_password': 'short'},
    headers={'Authorization': f'Bearer {token2}'},
)
print('Short new password   :', r.status_code)           # expect 422
assert r.status_code == 422

print()
print('All checks passed.')
"
```

Expected output:

```
Login status         : 200
Access token         : eyJhbGciOiJIUzI1NiIsInR5cCI6...
PATCH /password      : 200
Response body        : {'message': 'Password updated successfully'}
Login (new password) : 200
Login (old password) : 401
Wrong current pwd    : 400
Short new password   : 422

All checks passed.
```

---

Once all assertions pass, Feature 1 is complete. Authenticated users can now change their own password via `PATCH /auth/password`.