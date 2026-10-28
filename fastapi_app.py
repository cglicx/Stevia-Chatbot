from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from stevia_bot.chatbot import SteviaChatbot
from stevia_bot.device_control import DeviceControlError, send_device_command

app = FastAPI(title="Stevia Chatbot API")
chatbot = SteviaChatbot()


class ChatTurn(BaseModel):
    role: str
    content: str


class PendingDeviceCommand(BaseModel):
    device: Literal["light", "fan", "pump"]
    action: Literal["ON", "OFF"]


class ChatRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    message: str
    sensor_data: dict[str, Any] | None = None
    history: list[ChatTurn] | None = None
    pending_device_command: PendingDeviceCommand | None = Field(
        default=None,
        alias="pendingDeviceCommand",
    )


class DeviceCommandRequest(PendingDeviceCommand):
    confirmed: bool


@app.post("/api/chat")
def chat(payload: ChatRequest) -> dict[str, Any]:
    question = payload.message.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Please enter a question.")

    history = (
        [
            {
                "role": (
                    "assistant"
                    if turn.role.casefold() in {"bot", "ai"}
                    else turn.role
                ),
                "content": turn.content,
            }
            for turn in payload.history
        ]
        if payload.history is not None
        else None
    )
    result = chatbot.answer(
        question,
        sensor_data=payload.sensor_data,
        history=history,
        pending_device_context=(
            payload.pending_device_command.model_dump()
            if payload.pending_device_command is not None
            else None
        ),
    )
    if result.get("confirmation", {}).get("required") is True:
        quick_replies = [
            {"text": option["label"], "value": option["value"]}
            for option in result["confirmation"]["options"]
        ]
        result["quick_replies"] = quick_replies
        result["buttons"] = quick_replies
        result["deviceCommand"] = result["device_command"]
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