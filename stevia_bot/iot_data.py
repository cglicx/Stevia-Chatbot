import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DATASET_PATH = Path(__file__).resolve().parent.parent / "IoTProcessed_Data.csv"
SENSOR_READINGS_PATH = Path(__file__).resolve().parent.parent / "sensor_readings"
_LOGGER = logging.getLogger(__name__)
_SENSOR_FILES = {
    "Environment Temperature.xlsx": ("temperature", "celsius"),
    "Environment Humidity.xlsx": ("humidity", "percent"),
    "Soil Moisture.xlsx": ("water_level", "percent"),
    "Soil pH.xlsx": ("soil_ph", "pH"),
    "Soil Temperature.xlsx": ("soil_temperature", "celsius"),
    "Environment Light Intensity.xlsx": ("light_intensity", "lux"),
    "Water TDS.xlsx": ("water_tds", "ppm"),
    "Solar Panel Battery Voltage.xlsx": ("battery_voltage", "volts"),
}


def _number(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value) or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _trend_direction(values: list[float] | np.ndarray) -> str | None:
    """Classify the direction of at least three recent sensor samples."""
    samples = np.asarray(values, dtype=float)
    samples = samples[np.isfinite(samples)]
    if samples.size < 3:
        return None

    slope = float(np.polyfit(np.arange(samples.size), samples, 1)[0])
    differences = np.diff(samples)
    tolerance = max(
        0.02 * max(abs(float(np.median(samples))), 1.0),
        float(np.std(differences)) * 0.35,
    )
    if slope > tolerance:
        return "rising"
    if slope < -tolerance:
        return "falling"
    return "stable"


def _format_source_timestamp(value: Any) -> str:
    if hasattr(value, "isoformat"):
        return value.isoformat(sep=" ")
    return str(value).strip().replace(" +0530", "").replace(" -0530", "")


def _read_excel_sensor_rows(file_path: Path) -> pd.DataFrame:
    frame = pd.read_excel(file_path, header=1, usecols=[0, 2])
    if frame.empty:
        return pd.DataFrame(columns=["timestamp", "source_timestamp", "value"])

    source_timestamps = frame.iloc[:, 0]
    timestamps = pd.to_datetime(source_timestamps, errors="coerce", utc=True, format="mixed")
    values = pd.to_numeric(frame.iloc[:, 1], errors="coerce")
    valid = timestamps.notna() & np.isfinite(values.to_numpy(dtype=float))
    return pd.DataFrame({
        "timestamp": timestamps[valid],
        "source_timestamp": source_timestamps[valid],
        "value": values[valid],
    }).sort_values("timestamp")


def _load_latest_from_excel() -> dict[str, Any]:
    """Load the latest sensor reading from Excel files in sensor_readings folder.
    
    Excel structure has title row (row 0), header row (row 1), then data rows (row 2+).
    Timestamp is in column 0, measurement value is in column 2.
    """
    if not SENSOR_READINGS_PATH.exists():
        return {}

    latest_reading: dict[str, Any] = {}
    sensor_trends: dict[str, str | None] = {}
    sensor_units: dict[str, str] = {}
    latest_global_timestamp: pd.Timestamp | None = None
    latest_display_timestamp = ""
    
    for file_name, (parameter_name, unit) in _SENSOR_FILES.items():
        file_path = SENSOR_READINGS_PATH / file_name
        if not file_path.exists():
            continue

        try:
            valid_rows = _read_excel_sensor_rows(file_path)
            if valid_rows.empty:
                continue

            latest_row = valid_rows.iloc[-1]
            latest_timestamp = latest_row["timestamp"]
            latest_reading[parameter_name] = float(latest_row["value"])
            sensor_units[parameter_name] = unit
            sensor_trends[parameter_name] = _trend_direction(valid_rows["value"].tail(5).to_numpy())
            if latest_global_timestamp is None or latest_timestamp > latest_global_timestamp:
                latest_global_timestamp = latest_timestamp
                latest_display_timestamp = _format_source_timestamp(latest_row["source_timestamp"])
        except Exception:
            _LOGGER.exception("Unable to load sensor readings from %s", file_path)

    if latest_global_timestamp:
        latest_reading["recorded_at"] = latest_display_timestamp
    if sensor_trends:
        latest_reading["sensor_trends"] = sensor_trends
    if sensor_units:
        latest_reading["sensor_units"] = sensor_units
    return latest_reading


def load_latest_reading() -> dict[str, Any]:
    """Load latest sensor reading from Excel folder, with CSV as fallback."""
    reading = _load_latest_from_excel()
    if reading:
        reading["source"] = "sensor_readings (Excel)"
        return reading

    if not DATASET_PATH.exists():
        return {}

    try:
        frame = pd.read_csv(DATASET_PATH, encoding="utf-8-sig")
        timestamps = pd.to_datetime(frame["date"], errors="coerce", utc=True, format="mixed")
        frame = frame.loc[timestamps.notna()].copy()
        frame["_timestamp"] = timestamps[timestamps.notna()]
        if frame.empty:
            return {}
        frame = frame.sort_values("_timestamp")
        latest_row = frame.iloc[-1]
    except (OSError, KeyError, ValueError, pd.errors.ParserError):
        _LOGGER.exception("Unable to load sensor readings from %s", DATASET_PATH)
        return {}

    sensor_trends = {
        parameter: _trend_direction(pd.to_numeric(frame[column], errors="coerce").dropna().tail(5).to_numpy())
        for parameter, column in {
            "temperature": "tempreature",
            "humidity": "humidity",
            "water_level": "water_level",
        }.items()
        if column in frame
    }
    reading = {
        "recorded_at": _format_source_timestamp(latest_row["date"]),
        "temperature": _number(latest_row.get("tempreature")),
        "humidity": _number(latest_row.get("humidity")),
        "water_level": _number(latest_row.get("water_level")),
        "nitrogen": _number(latest_row.get("N")),
        "phosphorus": _number(latest_row.get("P")),
        "potassium": _number(latest_row.get("K")),
        "fan_on": _number(latest_row.get("Fan_actuator_ON")) == 1,
        "watering_pump_on": _number(latest_row.get("Watering_plant_pump_ON")) == 1,
        "water_pump_on": _number(latest_row.get("Water_pump_actuator_ON")) == 1,
        "source": DATASET_PATH.name,
    }
    if sensor_trends:
        reading["sensor_trends"] = sensor_trends
    reading["sensor_units"] = {
        "temperature": "celsius",
        "humidity": "percent",
        "water_level": "percent",
    }
    return reading


def load_sensor_records() -> list[dict[str, Any]]:
    """Load timestamped sensor history from Excel files, with CSV as a fallback."""
    records_by_timestamp: dict[pd.Timestamp, dict[str, Any]] = {}

    if SENSOR_READINGS_PATH.exists():
        for file_name, (parameter_name, _) in _SENSOR_FILES.items():
            file_path = SENSOR_READINGS_PATH / file_name
            if not file_path.exists():
                continue

            try:
                valid_rows = _read_excel_sensor_rows(file_path)
                for row in valid_rows.itertuples(index=False):
                    record = records_by_timestamp.setdefault(
                        row.timestamp,
                        {"recorded_at": _format_source_timestamp(row.source_timestamp)},
                    )
                    record[parameter_name] = float(row.value)
            except Exception:
                _LOGGER.exception("Unable to load sensor records from %s", file_path)

    if records_by_timestamp:
        return [
            records_by_timestamp[timestamp]
            for timestamp in sorted(records_by_timestamp, reverse=True)
        ]

    if not DATASET_PATH.exists():
        return []

    try:
        frame = pd.read_csv(DATASET_PATH, encoding="utf-8-sig")
    except (OSError, pd.errors.ParserError):
        _LOGGER.exception("Unable to load sensor records from %s", DATASET_PATH)
        return []
    if "date" not in frame:
        _LOGGER.error("Sensor records CSV %s is missing the 'date' column", DATASET_PATH)
        return []
    timestamps = pd.to_datetime(frame["date"], errors="coerce", utc=True, format="mixed")
    valid_frame = frame.loc[timestamps.notna()].copy()
    valid_frame["_timestamp"] = timestamps[timestamps.notna()]
    if valid_frame.empty:
        return []

    column_map = {
        "tempreature": "temperature",
        "humidity": "humidity",
        "water_level": "water_level",
        "N": "nitrogen",
        "P": "phosphorus",
        "K": "potassium",
        "Fan_actuator_ON": "fan_on",
        "Watering_plant_pump_ON": "watering_pump_on",
        "Water_pump_actuator_ON": "water_pump_on",
    }
    records = []
    for _, row in valid_frame.sort_values("_timestamp", ascending=False).iterrows():
        record: dict[str, Any] = {"recorded_at": _format_source_timestamp(row["date"])}
        for column, parameter_name in column_map.items():
            if column not in valid_frame:
                continue
            value = _number(row[column])
            if value is not None and np.isfinite(value):
                record[parameter_name] = (
                    value == 1 if parameter_name.endswith("_on") else value
                )
        records.append(record)
    return records


LATEST_READING = load_latest_reading()
