import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from typing import Any

BACKEND_API_BASE_URL = os.getenv(
    "CAPSTONE_BACKEND_API_BASE_URL",
    "http://192.168.1.20/Capstone/Backend/index.php?url=",
)
DEVICE_CONTROL_API_URL = f"{BACKEND_API_BASE_URL}device-control"
DEVICE_STATUS_API_URL = f"{BACKEND_API_BASE_URL}device-status"


class DeviceControlError(Exception):
    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.status_code = status_code


def send_device_command(device: str, action: str) -> dict[str, Any]:
    if device not in {"light", "pump"}:
        raise DeviceControlError(
            "Unsupported device; the connected ESP32 supports only light and pump.",
            400,
        )
    if action not in {"ON", "OFF"}:
        raise DeviceControlError("Action must be ON or OFF.", 400)

    body = json.dumps({"device": device, "action": action}).encode("utf-8")
    request = Request(
        DEVICE_CONTROL_API_URL,
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

    device_labels = {"light": "lights", "pump": "pump"}
    label = device_labels[device]
    device_subject = f"The {label}"
    state_verb = "are" if device == "light" else "is"
    try:
        status_request = Request(DEVICE_STATUS_API_URL, method="GET")
        with urlopen(status_request, timeout=10) as response:
            status_result = json.loads(response.read())
    except (HTTPError, TimeoutError, URLError, UnicodeDecodeError, json.JSONDecodeError):
        status_result = None

    reported_state = None
    if isinstance(status_result, dict) and status_result.get("success") is True:
        devices = status_result.get("devices")
        if isinstance(devices, list):
            reported_device = next(
                (
                    item
                    for item in devices
                    if isinstance(item, dict) and item.get("device_name") == device
                ),
                None,
            )
            if reported_device is not None:
                state = reported_device.get("actual_state")
                if isinstance(state, str) and state.upper() in {"ON", "OFF"}:
                    reported_state = state.upper()

    if reported_state == action:
        message = (
            f"{device_subject} {state_verb} {reported_state} "
            "according to the latest device report."
        )
    elif reported_state is not None:
        message = (
            f"The {label} command was accepted, but its latest reported state is "
            f"{reported_state}."
        )
    elif status_result is None:
        message = (
            f"The {label} command was accepted, but I couldn't verify its current state."
        )
    else:
        message = (
            f"The {label} command was accepted, but its current state has not been "
            "reported yet."
        )

    return {
        "success": True,
        "device": device,
        "action": action,
        "reported_state": reported_state,
        "message": message,
    }
