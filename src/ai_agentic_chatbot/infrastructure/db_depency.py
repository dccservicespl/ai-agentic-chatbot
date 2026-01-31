from ai_agentic_chatbot.infrastructure.datasource import get_session


def get_db_session():
    return get_session("primary")
