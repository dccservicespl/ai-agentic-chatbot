from fastapi import FastAPI
from ai_agentic_chatbot.controller.chat import router
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(
    title="AI Chat Application",
    version="1.0.0",
    description="Agent enabled AI ChatBot application",
)
app.include_router(router)


@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "UP"}
