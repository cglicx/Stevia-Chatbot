import hmac
import os
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from stevia_bot.chatbot import SteviaChatbot
from stevia_bot.device_control import DeviceControlError, send_device_command

ENVIRONMENT = os.getenv("ENVIRONMENT", "development").strip().lower()
CHATBOT_API_KEY = os.getenv("CHATBOT_API_KEY", "").strip()
if ENVIRONMENT == "production" and len(CHATBOT_API_KEY) < 32:
    raise RuntimeError(
        "Production requires CHATBOT_API_KEY to contain at least 32 characters."
    )

cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "").split(",")
    if origin.strip()
]

app = FastAPI(
    title="Stevia Chatbot API",
    docs_url=None if ENVIRONMENT == "production" else "/docs",
    redoc_url=None if ENVIRONMENT == "production" else "/redoc",
    openapi_url=None if ENVIRONMENT == "production" else "/openapi.json",
)
if cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
    )
chatbot = SteviaChatbot()


@app.middleware("http")
async def require_api_key(request: Request, call_next):
    if request.url.path.startswith("/api/") and request.method != "OPTIONS":
        if not CHATBOT_API_KEY:
            if ENVIRONMENT == "production":
                return JSONResponse(
                    status_code=503,
                    content={"detail": "API authentication is not configured."},
                )
        else:
            authorization = request.headers.get("Authorization", "")
            origin = request.headers.get("Origin")
            same_origin = origin == f"{request.url.scheme}://{request.url.netloc}"
            if not authorization and (
                same_origin or origin in cors_origins
            ):
                return await call_next(request)

            scheme, _, supplied_key = authorization.partition(" ")
            if (
                scheme.casefold() != "bearer"
                or not supplied_key
                or not hmac.compare_digest(
                    supplied_key.encode("utf-8"),
                    CHATBOT_API_KEY.encode("utf-8"),
                )
            ):
                return JSONResponse(
                    status_code=401,
                    content={"detail": "A valid bearer token is required."},
                    headers={"WWW-Authenticate": "Bearer"},
                )
    return await call_next(request)


class ChatTurn(BaseModel):
    role: str
    content: str


class PendingDeviceCommand(BaseModel):
    device: Literal["light", "pump"]
    action: Literal["ON", "OFF"]


class ChatRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    message: str = Field(max_length=2000)
    sensor_data: dict[str, Any] | None = None
    sensor_records: list[dict[str, Any]] | None = None
    sensor_records_total: int | None = None
    history: list[ChatTurn] | None = None
    pending_device_command: PendingDeviceCommand | None = None


class DeviceCommandRequest(PendingDeviceCommand):
    confirmed: bool


@app.get("/healthz")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


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
        sensor_records=payload.sensor_records,
        sensor_records_total=payload.sensor_records_total,
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