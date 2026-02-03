from ai_agentic_chatbot.infrastructure.datasource import get_session


def get_db_session():
    return {
        "mysql": get_session("mysql.primary"),
        "postgresql": get_session("postgresql.primary"),
    }
