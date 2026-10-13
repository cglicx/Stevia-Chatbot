from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from stevia_bot.chatbot import SteviaChatbot

app = FastAPI(title="Stevia Chatbot API")
chatbot = SteviaChatbot()


class ChatTurn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    sensor_data: dict[str, Any] | None = None
    history: list[ChatTurn] | None = None


@app.post("/api/chat")
def chat(payload: ChatRequest) -> dict[str, Any]:
    question = payload.message.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Please enter a question.")

    history = (
        [{"role": turn.role, "content": turn.content} for turn in payload.history]
        if payload.history is not None
        else None
    )
    return chatbot.answer(
        question,
        sensor_data=payload.sensor_data,
        history=history,
    )