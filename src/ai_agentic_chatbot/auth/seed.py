"""One-time script to create the first superuser account.

Run from the project root:
    python -m ai_agentic_chatbot.auth.seed

Reads SEED_USERNAME, SEED_EMAIL, SEED_PASSWORD from the environment (.env).
Remove or blank those vars from .env after the first successful run.
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    username = os.environ.get("SEED_USERNAME", "").strip()
    email = os.environ.get("SEED_EMAIL", "").strip()
    password = os.environ.get("SEED_PASSWORD", "").strip()

    missing = [name for name, val in [("SEED_USERNAME", username), ("SEED_EMAIL", email), ("SEED_PASSWORD", password)] if not val]
    if missing:
        print(f"ERROR: missing required env vars: {', '.join(missing)}")
        print("Set them in .env and re-run.")
        sys.exit(1)

    from ai_agentic_chatbot.infrastructure.datasource.datasource_init import initialize_datasources
    from ai_agentic_chatbot.infrastructure.datasource.factory import get_session
    from ai_agentic_chatbot.auth.repository import get_user_by_username, get_user_by_email, create_user
    from ai_agentic_chatbot.auth.password import hash_password

    initialize_datasources()
    session = get_session("postgresql.primary")

    try:
        if get_user_by_username(session, username) is not None:
            print(f"User '{username}' already exists — nothing to do.")
            return
        if get_user_by_email(session, email) is not None:
            print(f"Email '{email}' already registered — nothing to do.")
            return

        user = create_user(
            session,
            username=username,
            email=email,
            hashed_password=hash_password(password),
            is_superuser=True,
        )
        print(f"Superuser created: id={user.id}  username={user.username}  email={user.email}")
        print("Remove SEED_USERNAME / SEED_EMAIL / SEED_PASSWORD from .env now.")
    finally:
        session.close()


if __name__ == "__main__":
    main()