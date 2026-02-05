import os
from dotenv import load_dotenv

load_dotenv()


def get_db_connection_string() -> str:
    user = os.getenv("POSTGRESQL_USER")
    password = (os.getenv("POSTGRESQL_PASSWORD"))
    host = os.getenv("POSTGRESQL_HOST")
    port = os.getenv("POSTGRESQL_PORT", "5432")
    db = os.getenv("POSTGRESQL_DB")

    return f"postgresql+psycopg://{user}:{password}@{host}:{port}/{db}"
