from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from stevia_bot.chatbot import SteviaChatbot
from stevia_bot.device_control import DeviceControlError, send_device_command

app = FastAPI(title="Stevia Chatbot API")
chatbot = SteviaChatbot()


class ChatTurn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    sensor_data: dict[str, Any] | None = None
    history: list[ChatTurn] | None = None


class DeviceCommandRequest(BaseModel):
    device: Literal["light", "fan", "pump"]
    action: Literal["ON", "OFF"]
    confirmed: bool


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
    result = chatbot.answer(
        question,
        sensor_data=payload.sensor_data,
        history=history,
    )
    if result.get("execute_device_command") is True:
        command = result["device_command"]
        try:
            device_result = send_device_command(
                command["device"],
                command["action"],
            )
        except DeviceControlError as error:
            raise HTTPException(
                status_code=error.status_code,
                detail=str(error),
            ) from error
        result["answer"] = device_result["message"]
        result["device_result"] = device_result
        result.pop("execute_device_command")
    return result


@app.post("/api/device-command")
def device_command(payload: DeviceCommandRequest) -> dict[str, Any]:
    if not payload.confirmed:
        raise HTTPException(
            status_code=400,
            detail="Device command confirmation is required.",
        )
    try:
        return send_device_command(payload.device, payload.action)
    except DeviceControlError as error:
        raise HTTPException(
            status_code=error.status_code,
            detail=str(error),
        ) from error