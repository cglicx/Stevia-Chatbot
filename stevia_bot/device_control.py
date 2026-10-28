import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from typing import Any


class DeviceControlError(Exception):
    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.status_code = status_code


def send_device_command(device: str, action: str) -> dict[str, Any]:
    endpoint = os.getenv("DEVICE_CONTROL_API_URL", "").strip()
    if not endpoint:
        raise DeviceControlError(
            "Device control is not configured. Set DEVICE_CONTROL_API_URL.",
            status_code=503,
        )

    body = json.dumps({"device": device, "action": action}).encode("utf-8")
    request = Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=10) as response:
            response_body = response.read()
    except HTTPError as error:
        raise DeviceControlError(
            f"Device backend rejected the command (HTTP {error.code})."
        ) from error
    except (TimeoutError, URLError) as error:
        raise DeviceControlError("Unable to reach the device backend.") from error

    try:
        backend_result = json.loads(response_body)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DeviceControlError(
            "Device backend returned an invalid response."
        ) from error
    if not isinstance(backend_result, dict):
        raise DeviceControlError("Device backend returned an invalid response.")
    if backend_result.get("success") is not True:
        message = backend_result.get("message")
        if not isinstance(message, str) or not message:
            message = "Device backend could not execute the command."
        raise DeviceControlError(message)

    return {
        "success": True,
        "device": device,
        "action": action,
        "message": backend_result.get("message", "Device command sent."),
    }
