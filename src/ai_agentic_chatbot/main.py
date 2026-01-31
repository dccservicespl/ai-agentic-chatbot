from fastapi import FastAPI, Depends, HTTPException
from ai_agentic_chatbot.controller.chat import router
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from ai_agentic_chatbot.infrastructure.db_depency import get_db

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


@app.get("/db-health")
async def db_health(db: AsyncSession = Depends(get_db)):
    try:
        await db.execute(text("SELECT 1"))
        return {"database": "UP"}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))
