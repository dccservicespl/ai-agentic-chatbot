# Feature 2 — Save User Prompts to the Database

## Overview

Every time a user sends a message via `POST /stream`, the raw prompt text is saved to a `prompt_logs` table in PostgreSQL before the agent workflow starts. Each row records the authenticated user's ID, the conversation thread ID, the prompt text, and the timestamp. This provides a full audit trail of every question asked through the chatbot, enabling usage analytics, debugging, and compliance review without touching the agent's own checkpoint state.

---

## Step 1: Add the `PromptLog` ORM Model to `auth/models.py`

**1.1 — Open** `src/ai_agentic_chatbot/auth/models.py`.

**1.2 — Scroll to the very end of the file** (after the closing line of the `RefreshToken` class).

**1.3 — Append** the following class:

```python
class PromptLog(Base):
    __tablename__ = "prompt_logs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    thread_id: Mapped[str] = mapped_column(String(255), nullable=False)
    prompt_text: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
```

> No new imports are needed. `BigInteger`, `DateTime`, `ForeignKey`, `String`, `Text`, `func`, `Mapped`, `mapped_column`, and `datetime` are all already imported at the top of the file.

---

**1.4 — Verify the model loads correctly:**

```bash
python -c "
from ai_agentic_chatbot.auth.models import PromptLog
print('Table name :', PromptLog.__tablename__)
print('Columns    :', [c.name for c in PromptLog.__table__.columns])
print('Step 1 OK')
"
```

Expected output:

```
Table name : prompt_logs
Columns    : ['id', 'user_id', 'thread_id', 'prompt_text', 'created_at']
Step 1 OK
```

---

Once verified, Step 1 is complete. Move on to Step 2.

---

## Step 2: Update `auth/repository.py`

This step adds `PromptLog` to the existing models import and appends the `create_prompt_log` repository function.

---

**2.1 — Open** `src/ai_agentic_chatbot/auth/repository.py`.

**2.2 — Find** the existing models import line near the top of the file:

```python
from ai_agentic_chatbot.auth.models import User, RefreshToken
```

**Replace it with** (add `PromptLog`):

```python
from ai_agentic_chatbot.auth.models import User, RefreshToken, PromptLog
```

---

**2.3 — Scroll to the very end of the file** (after `update_user_password`).

**2.4 — Append** the following function:

```python
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
```

> The keyword-only arguments (`*`) prevent accidentally passing values as positional args. The function commits immediately and refreshes so the returned `PromptLog` has its `id` and `created_at` populated from the database.

---

**2.5 — Verify the function imports and has the correct signature:**

```bash
python -c "
from ai_agentic_chatbot.auth.repository import create_prompt_log
import inspect
print('create_prompt_log params:', list(inspect.signature(create_prompt_log).parameters.keys()))
print('Step 2 OK')
"
```

Expected output:

```
create_prompt_log params: ['db', 'user_id', 'thread_id', 'prompt_text']
Step 2 OK
```

---

Once verified, Step 2 is complete. Move on to Step 3.

---

## Step 3: Run the Alembic Migration

This step generates and applies the database migration to create the `prompt_logs` table.

> `alembic/env.py` needs **no changes**. Line 20 already contains `import ai_agentic_chatbot.auth.models  # noqa: F401`, which imports the entire models module. Because `PromptLog` was added to that same file in Step 1, Alembic's `autogenerate` will detect it automatically when the migration is generated.

---

**3.1 — Generate the migration file.** Run this from the project root:

```bash
alembic revision --autogenerate -m "create prompt_logs table"
```

This creates a new file inside `alembic/versions/` named something like:
`xxxxxxxxxxxx_create_prompt_logs_table.py`

---

**3.2 — Inspect the generated file before applying.**

Open the file in `alembic/versions/` and confirm the `upgrade()` function contains only:

- `op.create_table('prompt_logs', ...)` with all 5 columns
- `op.create_index(...)` for the `user_id` index

And the `downgrade()` function contains:

- `op.drop_index(...)` for the `user_id` index
- `op.drop_table('prompt_logs')`

There should be **no** `op.drop_table` calls for `users`, `refresh_tokens`, `orders`, `customer`, or any other existing table. If you see any unexpected drops, stop and do not proceed.

---

**3.3 — Apply the migration:**

```bash
alembic upgrade head
```

Expected output:

```
INFO  [alembic.runtime.migration] Running upgrade <prev_rev> -> <new_rev>, create prompt_logs table
```

---

**3.4 — Verify the table was created in the database:**

```bash
python -c "
from dotenv import load_dotenv; load_dotenv()
from ai_agentic_chatbot.infrastructure.datasource.datasource_init import initialize_datasources
from ai_agentic_chatbot.infrastructure.datasource.factory import get_engine
from sqlalchemy import text
initialize_datasources()
engine = get_engine('postgresql.primary')
with engine.connect() as conn:
    result = conn.execute(text(\"SELECT column_name, data_type FROM information_schema.columns WHERE table_name='prompt_logs' ORDER BY ordinal_position\"))
    for row in result:
        print(f'  {row[0]:15s} {row[1]}')
print('Step 3 OK')
"
```

Expected output:

```
  id              bigint
  user_id         bigint
  thread_id       character varying
  prompt_text     text
  created_at      timestamp with time zone
Step 3 OK
```

---

Once all 5 columns appear, Step 3 is complete. Move on to Step 4.

---

## Step 4: Update `server.py` — Hook into `/stream`

This step wires `create_prompt_log` into the `/stream` endpoint so every incoming message is saved before the agent runs.

---

**4.1 — Open** `src/ai_agentic_chatbot/server.py`.

**4.2 — Find** the existing auth dependencies import line (around line 39):

```python
from ai_agentic_chatbot.auth.dependencies import get_current_user
```

**Replace it with** (add `get_auth_db`):

```python
from ai_agentic_chatbot.auth.dependencies import get_auth_db, get_current_user
```

---

**4.3 — Find** the `User` model import on the next line:

```python
from ai_agentic_chatbot.auth.models import User
```

**Add one new line immediately after it:**

```python
from ai_agentic_chatbot.auth.repository import create_prompt_log
```

After both changes the import block should look like this:

```python
from ai_agentic_chatbot.auth.dependencies import get_auth_db, get_current_user
from ai_agentic_chatbot.auth.models import User
from ai_agentic_chatbot.auth.repository import create_prompt_log
```

---

**4.4 — Find** the `stream_endpoint` function signature:

```python
async def stream_endpoint(stream_request: StreamRequest, current_user: User = Depends(get_current_user)):
```

**Replace it with** (add the `db` parameter):

```python
async def stream_endpoint(
    stream_request: StreamRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_auth_db),
):
```

> `Session` is already imported at line 9 of `server.py` — no new import is needed for it.

---

**4.5 — Find** this block inside the `stream_endpoint` body (a few lines below the signature):

```python
        if not messages:
            raise HTTPException(status_code=400, detail="messages cannot be empty")

        config = {"configurable": {"thread_id": thread_id}}
        inputs = {"messages": [HumanMessage(content=messages[-1].content)]}
```

**Replace it with** (insert the `create_prompt_log` call between the validation check and the config block):

```python
        if not messages:
            raise HTTPException(status_code=400, detail="messages cannot be empty")

        create_prompt_log(
            db,
            user_id=current_user.id,
            thread_id=thread_id,
            prompt_text=messages[-1].content,
        )

        config = {"configurable": {"thread_id": thread_id}}
        inputs = {"messages": [HumanMessage(content=messages[-1].content)]}
```

---

**4.6 — Verify the server imports load correctly:**

```bash
python -c "
from dotenv import load_dotenv; load_dotenv()
from ai_agentic_chatbot.server import app
stream_route = next(r for r in app.routes if hasattr(r, 'path') and r.path == '/stream')
import inspect
sig = inspect.signature(stream_route.endpoint)
print('stream_endpoint params:', list(sig.parameters.keys()))
print('Step 4 OK')
"
```

Expected output:

```
stream_endpoint params: ['stream_request', 'current_user', 'db']
Step 4 OK
```

---

Once verified, Step 4 is complete. The feature is fully implemented.

---

## Verification — End-to-End Test

Run this script after starting the server to confirm that a message sent via `/stream` is persisted to `prompt_logs`.

> Prerequisites: server running (`uvicorn ai_agentic_chatbot.server:app --port 8000 --reload`), a valid user account, and `psycopg2` installed.

```python
import os
import requests
import psycopg2
from dotenv import load_dotenv

load_dotenv()

BASE = "http://localhost:8000"
USERNAME = "<your username>"
PASSWORD = "<your password>"
THREAD_ID = "test-prompt-log-thread"

# Step 1 — Login and obtain a token
r = requests.post(f"{BASE}/auth/login", data={"username": USERNAME, "password": PASSWORD})
assert r.status_code == 200, f"Login failed: {r.text}"
token = r.json()["access_token"]
print("Login OK, token:", token[:30], "...")

# Step 2 — Send a message to /stream and consume the SSE response
headers = {"Authorization": f"Bearer {token}"}
payload = {
    "thread_id": THREAD_ID,
    "messages": [{"role": "user", "content": "How many orders were placed last month?"}],
}
with requests.post(f"{BASE}/stream", json=payload, headers=headers, stream=True) as resp:
    assert resp.status_code == 200, f"/stream failed: {resp.text}"
    for chunk in resp.iter_content(chunk_size=None):
        pass  # consume the full stream
print("/stream call completed")

# Step 3 — Query prompt_logs directly via psycopg2
conn = psycopg2.connect(
    host=os.environ["POSTGRESQL_HOST"],
    port=os.environ.get("POSTGRESQL_PORT", "5432"),
    dbname=os.environ["POSTGRESQL_DB"],
    user=os.environ["POSTGRESQL_USER"],
    password=os.environ["POSTGRESQL_PASSWORD"],
)
cur = conn.cursor()
cur.execute(
    "SELECT id, user_id, thread_id, prompt_text, created_at "
    "FROM prompt_logs WHERE thread_id = %s ORDER BY created_at DESC LIMIT 1",
    (THREAD_ID,),
)
row = cur.fetchone()
cur.close()
conn.close()

if row:
    print(f"Row found — id={row[0]}, user_id={row[1]}, thread_id={row[2]}")
    print(f"  prompt_text : {row[3]}")
    print(f"  created_at  : {row[4]}")
    print("Verification passed.")
else:
    print("No row found in prompt_logs — check the migration was applied and the endpoint ran without error.")
```

Alternatively, open your PostgreSQL client (e.g., pgAdmin, DBeaver, or `psql`) and run:

```sql
SELECT id, user_id, thread_id, prompt_text, created_at
FROM prompt_logs
ORDER BY created_at DESC
LIMIT 10;
```

You should see one row per `/stream` call, with `prompt_text` matching the last message in the request body.

---

## Schema Reference

| Column | Type | Nullable | Default | Notes |
|---|---|---|---|---|
| `id` | `BIGINT` | NOT NULL | autoincrement | Primary key |
| `user_id` | `BIGINT` | NOT NULL | — | FK → `users.id` CASCADE DELETE; indexed |
| `thread_id` | `VARCHAR(255)` | NOT NULL | — | LangGraph conversation thread identifier |
| `prompt_text` | `TEXT` | NOT NULL | — | The raw user message text |
| `created_at` | `TIMESTAMPTZ` | NOT NULL | `NOW()` | Set by the database at insert time |

Foreign key: `prompt_logs.user_id` references `users.id` with `ON DELETE CASCADE` — deleting a user automatically removes all their prompt log rows.

Index: `ix_prompt_logs_user_id` on `(user_id)` — supports fast per-user log queries.