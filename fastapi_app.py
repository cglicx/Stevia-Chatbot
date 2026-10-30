from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from stevia_bot.backend_sensor_data import SensorBackendError, fetch_live_sensor_context
from stevia_bot.chatbot import SteviaChatbot
from stevia_bot.device_control import DeviceControlError, send_device_command

app = FastAPI(title="Stevia Chatbot API")
chatbot = SteviaChatbot()


class ChatTurn(BaseModel):
    role: str
    content: str


class PendingDeviceCommand(BaseModel):
    device: Literal["light", "pump"]
    action: Literal["ON", "OFF"]


class ChatRequest(BaseModel):
    message: str
    sensor_data: dict[str, Any] | None = None
    sensor_records: list[dict[str, Any]] | None = None
    sensor_records_total: int | None = None
    history: list[ChatTurn] | None = None
    pending_device_command: PendingDeviceCommand | None = None


class DeviceCommandRequest(PendingDeviceCommand):
    confirmed: bool


@app.post("/api/chat")
def chat(payload: ChatRequest) -> dict[str, Any]:
    question = payload.message.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Please enter a question.")

    sensor_data = payload.sensor_data
    sensor_records = payload.sensor_records
    sensor_records_total = payload.sensor_records_total
    if sensor_data is None or sensor_records is None:
        try:
            live_data, live_records, live_total = fetch_live_sensor_context()
        except SensorBackendError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        if sensor_data is None:
            sensor_data = live_data
        if sensor_records is None:
            sensor_records = live_records
        if sensor_records_total is None:
            sensor_records_total = live_total

    history = (
        [{"role": turn.role, "content": turn.content} for turn in payload.history]
        if payload.history is not None
        else None
    )
    result = chatbot.answer(
        question,
        sensor_data=sensor_data,
        sensor_records=sensor_records,
        sensor_records_total=sensor_records_total,
        history=history,
        pending_device_context=(
            payload.pending_device_command.model_dump()
            if payload.pending_device_command is not None
            else None
        ),
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