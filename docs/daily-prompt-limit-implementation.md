# Feature 3 — Configurable Daily Prompt Limit

## Overview

Feature 3 lets a superuser set a per-user daily prompt cap.

- `daily_prompt_limit = 0` means **unlimited** — the check is skipped entirely for those users.
- Any value `> 0` is enforced: once the user has sent that many prompts in the current UTC day, the next `POST /stream` call returns **HTTP 429**.
- The limit check fires **before** `create_prompt_log` is called, so rejected prompts are **not** counted against the user's total.
- The superuser sets limits via `PATCH /auth/users/{username}/limit`. Non-superusers receive HTTP 403.

---

## Dependency Note

Feature 3 requires **Feature 2 (prompt logging)** to already be applied. Specifically:

- The `prompt_logs` table must exist in the database (created by the Feature 2 Alembic migration).
- The `PromptLog` ORM model and `create_prompt_log` repository function must already be present in the codebase.

Do not proceed with Feature 3 until `alembic upgrade head` has been run for the Feature 2 migration.

---

## Step 1: Edit `auth/models.py`

**1.1 — Open** `src/ai_agentic_chatbot/auth/models.py`.

**1.2 — Find** the SQLAlchemy import line:

```python
from sqlalchemy import BigInteger, Boolean, DateTime, ForeignKey, String, Text, func
```

**Replace it with** (add `Integer`):

```python
from sqlalchemy import BigInteger, Boolean, DateTime, ForeignKey, Integer, String, Text, func
```

**1.3 — Find** the `updated_at` column in the `User` class:

```python
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True, onupdate=func.now()
    )
```

**Append** the new column immediately after it (before the blank line that separates `User` from `RefreshToken`):

```python
    daily_prompt_limit: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
```

After both edits the `User` class should end like this:

```python
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True, onupdate=func.now()
    )
    daily_prompt_limit: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")


class RefreshToken(Base):
```

**1.4 — Verify:**

```bash
python -c "
from ai_agentic_chatbot.auth.models import User
cols = [c.name for c in User.__table__.columns]
print('User columns:', cols)
assert 'daily_prompt_limit' in cols
print('Step 1 OK')
"
```

Expected output:

```
User columns: ['id', 'username', 'email', 'hashed_password', 'is_active', 'is_superuser', 'created_at', 'updated_at', 'daily_prompt_limit']
Step 1 OK
```

---

## Step 2: Run the Alembic Migration

**2.1 — Generate the migration file.** Run this from the project root with the virtualenv active:

```bash
alembic revision --autogenerate -m "add daily_prompt_limit to users"
```

This creates a new file in `alembic/versions/` named something like:
`xxxxxxxxxxxx_add_daily_prompt_limit_to_users.py`

---

**2.2 — Inspect the generated file** before applying it. Open the file in `alembic/versions/` and confirm the `upgrade()` function contains **only**:

```python
op.add_column('users', sa.Column('daily_prompt_limit', sa.Integer(), server_default='0', nullable=False))
```

And the `downgrade()` function contains **only**:

```python
op.drop_column('users', 'daily_prompt_limit')
```

There must be **no** `op.drop_table`, `op.drop_column` for any other column, or any other destructive operation. If you see unexpected changes, stop and do not proceed.

---

**2.3 — Apply the migration:**

```bash
alembic upgrade head
```

Expected output:

```
INFO  [alembic.runtime.migration] Running upgrade <prev_rev> -> <new_rev>, add daily_prompt_limit to users
```

---

**2.4 — Verify the column was added:**

```bash
python -c "
from dotenv import load_dotenv; load_dotenv()
from ai_agentic_chatbot.infrastructure.datasource.datasource_init import initialize_datasources
from ai_agentic_chatbot.infrastructure.datasource.factory import get_engine
from sqlalchemy import text
initialize_datasources()
engine = get_engine('postgresql.primary')
with engine.connect() as conn:
    result = conn.execute(text(\"SELECT column_name, data_type, column_default FROM information_schema.columns WHERE table_name='users' AND column_name='daily_prompt_limit'\"))
    for row in result:
        print(f'column: {row[0]}, type: {row[1]}, default: {row[2]}')
print('Step 2 OK')
"
```

Expected output:

```
column: daily_prompt_limit, type: integer, default: 0
Step 2 OK
```

---

Once you see the column listed, Step 2 is complete. Move on to Step 3.

---

## Step 3: Edit `auth/schemas.py`

**3.1 — Open** `src/ai_agentic_chatbot/auth/schemas.py`.

**3.2 — Find** the `UserResponse` class body:

```python
    id: int
    username: str
    email: str
    is_active: bool
    is_superuser: bool
    created_at: datetime
```

**Replace it with** (add `daily_prompt_limit` between `is_superuser` and `created_at`):

```python
    id: int
    username: str
    email: str
    is_active: bool
    is_superuser: bool
    daily_prompt_limit: int = 0
    created_at: datetime
```

**3.3 — Find** the last class in the file:

```python
class PasswordUpdateResponse(BaseModel):
    message: str
```

**Append** the new schema immediately after it:

```python
class PromptLimitUpdateRequest(BaseModel):
    daily_prompt_limit: int = Field(..., ge=0)
```

> `Field(..., ge=0)` means the field is required (`...`) and must be greater than or equal to 0. `Field` is already imported at the top of the file.

**3.4 — Verify:**

```bash
python -c "
from ai_agentic_chatbot.auth.schemas import UserResponse, PromptLimitUpdateRequest
print('UserResponse fields:', list(UserResponse.model_fields.keys()))
assert 'daily_prompt_limit' in UserResponse.model_fields

print('PromptLimitUpdateRequest fields:', list(PromptLimitUpdateRequest.model_fields.keys()))
assert 'daily_prompt_limit' in PromptLimitUpdateRequest.model_fields

import pydantic
try:
    PromptLimitUpdateRequest(daily_prompt_limit=-1)
    print('ERROR: should have raised')
except pydantic.ValidationError:
    print('Negative value correctly rejected')

print('Step 3 OK')
"
```

Expected output:

```
UserResponse fields: ['id', 'username', 'email', 'is_active', 'is_superuser', 'daily_prompt_limit', 'created_at']
PromptLimitUpdateRequest fields: ['daily_prompt_limit']
Negative value correctly rejected
Step 3 OK
```

---

## Step 4: Edit `auth/repository.py`

**4.1 — Open** `src/ai_agentic_chatbot/auth/repository.py`.

**4.2 — Find** the datetime import line at the top:

```python
from datetime import datetime
```

**Replace it with** (add `date`, `time`, `timezone`):

```python
from datetime import date, datetime, time, timezone
```

**4.3 — Find** the SQLAlchemy import line:

```python
from sqlalchemy import select, update
```

**Replace it with** (add `func`):

```python
from sqlalchemy import func, select, update
```

**4.4 — Scroll to the very end of the file** (after `create_prompt_log`).

**Append** the two new functions:

```python
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
```

**4.5 — Verify:**

```bash
python -c "
from ai_agentic_chatbot.auth.repository import count_prompts_today, update_user_prompt_limit
import inspect
print('count_prompts_today params  :', list(inspect.signature(count_prompts_today).parameters.keys()))
print('update_user_prompt_limit params:', list(inspect.signature(update_user_prompt_limit).parameters.keys()))
print('Step 4 OK')
"
```

Expected output:

```
count_prompts_today params  : ['db', 'user_id']
update_user_prompt_limit params: ['db', 'user', 'limit']
Step 4 OK
```

---

## Step 5: Edit `server.py`

**5.1 — Open** `src/ai_agentic_chatbot/server.py`.

**5.2 — Find** the repository import line:

```python
from ai_agentic_chatbot.auth.repository import create_prompt_log
```

**Replace it with** (add `count_prompts_today`):

```python
from ai_agentic_chatbot.auth.repository import count_prompts_today, create_prompt_log
```

**5.3 — Find** this block inside the `stream_endpoint` function body:

```python
        if not messages:
            raise HTTPException(status_code=400, detail="messages cannot be empty")

        create_prompt_log(
            db,
            user_id=current_user.id,
            thread_id=thread_id,
            prompt_text=messages[-1].content,
        )
```

**Replace it with** (insert the limit check between the empty-messages guard and `create_prompt_log`):

```python
        if not messages:
            raise HTTPException(status_code=400, detail="messages cannot be empty")

        if current_user.daily_prompt_limit > 0:
            used_today = count_prompts_today(db, current_user.id)
            if used_today >= current_user.daily_prompt_limit:
                raise HTTPException(status_code=429, detail="Daily prompt limit reached")

        create_prompt_log(
            db,
            user_id=current_user.id,
            thread_id=thread_id,
            prompt_text=messages[-1].content,
        )
```

**5.4 — Verify:**

```bash
python -c "
from dotenv import load_dotenv; load_dotenv()
from ai_agentic_chatbot.server import app
stream_route = next(r for r in app.routes if hasattr(r, 'path') and r.path == '/stream')
print('Stream route found:', stream_route.path)
import inspect, ai_agentic_chatbot.server as srv
src = inspect.getsource(srv.stream_endpoint)
assert 'count_prompts_today' in src, 'count_prompts_today not found in stream_endpoint'
assert 'daily_prompt_limit' in src
assert '429' in src
print('Step 5 OK')
"
```

Expected output:

```
Stream route found: /stream
Step 5 OK
```

---

## Step 6: Edit `auth/router.py`

**6.1 — Open** `src/ai_agentic_chatbot/auth/router.py`.

**6.2 — Find** the repository import block:

```python
from ai_agentic_chatbot.auth.repository import (
    create_refresh_token,
    get_refresh_token_by_hash,
    get_user_by_id,
    mark_token_used,
    revoke_family,
    revoke_token,
)
```

**Replace it with** (add `get_user_by_username` and `update_user_prompt_limit`):

```python
from ai_agentic_chatbot.auth.repository import (
    create_refresh_token,
    get_refresh_token_by_hash,
    get_user_by_id,
    get_user_by_username,
    mark_token_used,
    revoke_family,
    revoke_token,
    update_user_prompt_limit,
)
```

**6.3 — Find** the schemas import line:

```python
from ai_agentic_chatbot.auth.schemas import LogoutRequest, PasswordUpdateRequest, PasswordUpdateResponse, RefreshRequest, Token, UserCreate, UserResponse
```

**Replace it with** (add `PromptLimitUpdateRequest`):

```python
from ai_agentic_chatbot.auth.schemas import LogoutRequest, PasswordUpdateRequest, PasswordUpdateResponse, PromptLimitUpdateRequest, RefreshRequest, Token, UserCreate, UserResponse
```

**6.4 — Scroll to the very end of the file** (after `PATCH /password`).

**Append** the new route:

```python
@router.patch("/users/{username}/limit", response_model=UserResponse)
def set_prompt_limit(
    username: str,
    payload: PromptLimitUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_auth_db),
):
    if not current_user.is_superuser:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Superuser access required")
    target_user = get_user_by_username(db, username)
    if target_user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    updated = update_user_prompt_limit(db, target_user, payload.daily_prompt_limit)
    return UserResponse.model_validate(updated)
```

**6.5 — Verify:**

```bash
python -c "
from dotenv import load_dotenv; load_dotenv()
from ai_agentic_chatbot.auth.router import router
routes = [(sorted(r.methods), r.path) for r in router.routes]
print('Auth routes:')
for methods, path in routes:
    print(' ', methods, path)
assert any(path == '/users/{username}/limit' for _, path in routes), 'limit route not found'
print('Step 6 OK')
"
```

Expected output:

```
Auth routes:
  ['POST'] /register
  ['POST'] /login
  ['POST'] /refresh
  ['POST'] /logout
  ['GET'] /me
  ['PATCH'] /password
  ['PATCH'] /users/{username}/limit
Step 6 OK
```

---

## Logic Explanation

### The `daily_prompt_limit = 0` sentinel

A value of `0` means **unlimited**. The check in `stream_endpoint` is:

```python
if current_user.daily_prompt_limit > 0:
    ...
```

This means users created before Feature 3 was deployed are unaffected — their `daily_prompt_limit` defaults to `0` via `server_default="0"` on the column, so the check is never entered.

### UTC midnight as the day boundary

`count_prompts_today` computes the start of the current day in UTC:

```python
today_start = datetime.combine(date.today(), time.min).replace(tzinfo=timezone.utc)
```

`time.min` is `00:00:00`. The query then counts all `PromptLog` rows for the user where `created_at >= today_start`. Because `prompt_logs.created_at` is stored with timezone (`DateTime(timezone=True)`), the comparison is timezone-aware on both sides. The day resets at 00:00 UTC regardless of the server's local timezone.

### Rejected prompts are not counted

The check runs before `create_prompt_log` is called:

```python
if current_user.daily_prompt_limit > 0:
    used_today = count_prompts_today(db, current_user.id)
    if used_today >= current_user.daily_prompt_limit:
        raise HTTPException(status_code=429, detail="Daily prompt limit reached")  # exits here

create_prompt_log(...)  # only reached if the check passed
```

A rejected request raises `HTTPException` immediately, so no log row is written. The user's count does not increase on a 429 response.

---

## Verification — End-to-End Test Script

Start the server in a separate terminal before running this script:

```bash
python -m uvicorn ai_agentic_chatbot.server:app --host 0.0.0.0 --port 8000 --reload
```

Then run (replacing the placeholder values with real credentials):

```bash
python -c "
import requests

BASE = 'http://localhost:8000'
SUPERUSER_USERNAME = 'admin'
SUPERUSER_PASSWORD = '<superuser password>'
TARGET_USERNAME = '<a non-superuser username>'
TARGET_PASSWORD = '<that user password>'

# --- Step 1: Superuser logs in ---
r = requests.post(f'{BASE}/auth/login', data={'username': SUPERUSER_USERNAME, 'password': SUPERUSER_PASSWORD})
assert r.status_code == 200, f'Superuser login failed: {r.status_code}'
su_token = r.json()['access_token']
print('Superuser login OK')

# --- Step 2: Set target user limit to 2 ---
r = requests.patch(
    f'{BASE}/auth/users/{TARGET_USERNAME}/limit',
    json={'daily_prompt_limit': 2},
    headers={'Authorization': f'Bearer {su_token}'},
)
print('Set limit status    :', r.status_code)   # expect 200
assert r.status_code == 200
body = r.json()
print('daily_prompt_limit  :', body['daily_prompt_limit'])
assert body['daily_prompt_limit'] == 2

# --- Step 3: Target user logs in ---
r = requests.post(f'{BASE}/auth/login', data={'username': TARGET_USERNAME, 'password': TARGET_PASSWORD})
assert r.status_code == 200, f'Target login failed: {r.status_code}'
tgt_token = r.json()['access_token']
tgt_headers = {'Authorization': f'Bearer {tgt_token}'}
print('Target user login OK')

# --- Step 4: First prompt — expect 200 ---
r = requests.post(f'{BASE}/stream',
    json={'thread_id': 'test-limit-thread', 'messages': [{'role': 'user', 'content': 'hello'}]},
    headers=tgt_headers,
)
print('Prompt 1 status     :', r.status_code)   # expect 200
assert r.status_code == 200

# --- Step 5: Second prompt — expect 200 ---
r = requests.post(f'{BASE}/stream',
    json={'thread_id': 'test-limit-thread', 'messages': [{'role': 'user', 'content': 'hello again'}]},
    headers=tgt_headers,
)
print('Prompt 2 status     :', r.status_code)   # expect 200
assert r.status_code == 200

# --- Step 6: Third prompt — expect 429 ---
r = requests.post(f'{BASE}/stream',
    json={'thread_id': 'test-limit-thread', 'messages': [{'role': 'user', 'content': 'one more'}]},
    headers=tgt_headers,
)
print('Prompt 3 status     :', r.status_code)   # expect 429
print('Prompt 3 detail     :', r.json().get('detail'))
assert r.status_code == 429
assert r.json()['detail'] == 'Daily prompt limit reached'

# --- Step 7: Superuser sets limit back to 0 (unlimited) ---
r = requests.patch(
    f'{BASE}/auth/users/{TARGET_USERNAME}/limit',
    json={'daily_prompt_limit': 0},
    headers={'Authorization': f'Bearer {su_token}'},
)
print('Reset limit status  :', r.status_code)   # expect 200
assert r.status_code == 200
assert r.json()['daily_prompt_limit'] == 0

# --- Step 8: Target user can send again ---
r = requests.post(f'{BASE}/stream',
    json={'thread_id': 'test-limit-thread', 'messages': [{'role': 'user', 'content': 'now unlimited'}]},
    headers=tgt_headers,
)
print('Prompt after reset  :', r.status_code)   # expect 200
assert r.status_code == 200

print()
print('All checks passed.')
"
```

Expected output:

```
Superuser login OK
Set limit status    : 200
daily_prompt_limit  : 2
Target user login OK
Prompt 1 status     : 200
Prompt 2 status     : 200
Prompt 3 status     : 429
Prompt 3 detail     : Daily prompt limit reached
Reset limit status  : 200
Prompt after reset  : 200

All checks passed.
```

---

## Additional Verification — Non-Superuser Cannot Set Limits

```bash
python -c "
import requests

BASE = 'http://localhost:8000'
REGULAR_USERNAME = '<a non-superuser username>'
REGULAR_PASSWORD = '<that user password>'

r = requests.post(f'{BASE}/auth/login', data={'username': REGULAR_USERNAME, 'password': REGULAR_PASSWORD})
token = r.json()['access_token']

r = requests.patch(
    f'{BASE}/auth/users/{REGULAR_USERNAME}/limit',
    json={'daily_prompt_limit': 5},
    headers={'Authorization': f'Bearer {token}'},
)
print('Non-superuser set limit:', r.status_code)  # expect 403
print('Detail                 :', r.json().get('detail'))
assert r.status_code == 403
print('403 check passed.')
"
```

---

## Route Summary

| Method | Path | Auth | Who can call | Description |
|--------|------|------|--------------|-------------|
| `PATCH` | `/auth/users/{username}/limit` | Bearer JWT | Superuser only | Set `daily_prompt_limit` for any user. `0` = unlimited. Returns the updated `UserResponse`. |
| `POST` | `/stream` | Bearer JWT | Any active user | Returns HTTP 429 if the user has a non-zero limit and has already reached it today (UTC). |

**Changed response schemas:**

| Schema | Change |
|--------|--------|
| `UserResponse` | Added `daily_prompt_limit: int` field. All endpoints returning `UserResponse` (`/auth/register`, `/auth/me`, `PATCH /auth/password`, `PATCH /auth/users/{username}/limit`) now include this field. |