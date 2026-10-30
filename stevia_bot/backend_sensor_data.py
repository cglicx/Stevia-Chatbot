import json
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .device_control import BACKEND_API_BASE_URL


class SensorBackendError(Exception):
    pass


def _get_backend_data(
    path: str, params: Optional[dict[str, int]] = None
) -> dict[str, Any]:
    url = f"{BACKEND_API_BASE_URL}{path}"
    if params:
        url += f"&{urlencode(params)}"

    request = Request(
        url,
        headers={"Accept": "application/json"},
        method="GET",
    )
    try:
        with urlopen(request, timeout=5) as response:
            body = response.read()
    except HTTPError as error:
        raise SensorBackendError(
            f"The sensor backend returned HTTP {error.code}."
        ) from error
    except (TimeoutError, URLError) as error:
        raise SensorBackendError("The sensor backend is unavailable.") from error

    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SensorBackendError(
            "The sensor backend returned invalid JSON."
        ) from error

    if (
        not isinstance(payload, dict)
        or payload.get("success") is not True
        or not isinstance(payload.get("data"), dict)
    ):
        raise SensorBackendError("The sensor backend returned an unexpected response.")

    return payload["data"]


def fetch_live_sensor_context() -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    dashboard = _get_backend_data("sensor/dashboard")
    history = _get_backend_data(
        "sensor/history", {"days": 30, "limit": 1000}
    )
    reading = dashboard.get("reading")
    records = history.get("records")
    total_records = history.get("total_records")

    if reading is not None and not isinstance(reading, dict):
        raise SensorBackendError("The sensor backend returned an invalid latest reading.")
    if not isinstance(records, list) or not isinstance(total_records, int):
        raise SensorBackendError("The sensor backend returned invalid sensor history.")

    required_fields = {
        "timestamp",
        "temperature",
        "humidity",
        "soil_moisture",
        "soil_ph",
        "light_intensity",
    }
    if reading is not None and (
        not required_fields.issubset(reading)
        or not isinstance(reading["timestamp"], str)
    ):
        raise SensorBackendError("The sensor backend returned an incomplete latest reading.")
    if any(
        not isinstance(record, dict)
        or not required_fields.issubset(record)
        or not isinstance(record.get("timestamp"), str)
        for record in records
    ):
        raise SensorBackendError("The sensor backend returned incomplete sensor history.")

    sensor_data: dict[str, Any] = {}
    if reading is not None:
        sensor_data = {
            "recorded_at": reading["timestamp"],
            "temperature": reading["temperature"],
            "humidity": reading["humidity"],
            "soil_moisture": reading["soil_moisture"],
            "water_level": reading["soil_moisture"],
            "soil_ph": reading["soil_ph"],
            "soil_pH": reading["soil_ph"],
            "light_intensity": reading["light_intensity"],
            "source": "capstone_db.sensor_readings",
        }

    sensor_records = [
        {
            "recorded_at": record["timestamp"],
            "temperature": record["temperature"],
            "humidity": record["humidity"],
            "soil_moisture": record["soil_moisture"],
            "water_level": record["soil_moisture"],
            "soil_ph": record["soil_ph"],
            "light_intensity": record["light_intensity"],
        }
        for record in records
        if isinstance(record, dict)
    ]

    return sensor_data, sensor_records, total_records
