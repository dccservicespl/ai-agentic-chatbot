from fastapi import APIRouter
from pydantic import BaseModel

from ai_agentic_chatbot.services.orchestration import ChatOrchestrator

router = APIRouter(prefix="/api")
orchestrator = ChatOrchestrator()

class ChatRequest(BaseModel):
    question: str

@router.post("/chat/query", tags=["Chat API"])
def chat_query(request: ChatRequest):
    return orchestrator.process(request.question)