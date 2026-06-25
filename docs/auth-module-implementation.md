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