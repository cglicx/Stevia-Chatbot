from flask import Flask, jsonify, render_template, request

from stevia_bot.backend_sensor_data import SensorBackendError, fetch_live_sensor_context
from stevia_bot.chatbot import SteviaChatbot
from stevia_bot.device_control import DeviceControlError, send_device_command
from stevia_bot.knowledge_base import SUGGESTED_QUESTIONS, WELCOME_MESSAGE

app = Flask(__name__)
chatbot = SteviaChatbot()


@app.get("/")
def index():
    return render_template(
        "index.html",
        welcome_message=WELCOME_MESSAGE,
        suggested_questions=SUGGESTED_QUESTIONS,
    )


@app.post("/api/chat")
def chat():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        payload = {}
    question = str(payload.get("message", "")).strip()
    history = payload.get("history")
    pending_device_command = payload.get(
        "pending_device_command",
        payload.get("pendingDeviceCommand"),
    )

    if not question:
        return jsonify({"error": "Please enter a question."}), 400
    if pending_device_command is not None and (
        not isinstance(pending_device_command, dict)
        or not isinstance(pending_device_command.get("device"), str)
        or pending_device_command.get("device") not in {"light", "pump"}
        or not isinstance(pending_device_command.get("action"), str)
        or pending_device_command.get("action") not in {"ON", "OFF"}
    ):
        return jsonify({"error": "Invalid pending device command."}), 400
    if isinstance(history, list):
        history = [
            {
                **turn,
                "role": (
                    "assistant"
                    if isinstance(turn.get("role"), str)
                    and turn["role"].casefold() in {"bot", "ai"}
                    else turn.get("role")
                ),
            }
            if isinstance(turn, dict)
            else turn
            for turn in history
        ]

    try:
        sensor_data, sensor_records, total_records = fetch_live_sensor_context()
    except SensorBackendError as error:
        return jsonify({"error": str(error)}), 502

    result = chatbot.answer(
        question,
        sensor_data=sensor_data,
        sensor_records=sensor_records,
        sensor_records_total=total_records,
        history=history,
        pending_device_context=pending_device_command,
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
            return jsonify({"error": str(error)}), error.status_code
        result["answer"] = device_result["message"]
        result["device_result"] = device_result
        result.pop("execute_device_command")
    return jsonify(result)


@app.post("/api/device-command")
def device_command():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or payload.get("confirmed") is not True:
        return jsonify({"error": "Device command confirmation is required."}), 400

    device = payload.get("device")
    action = payload.get("action")
    if (
        not isinstance(device, str)
        or device not in {"light", "pump"}
        or not isinstance(action, str)
        or action not in {"ON", "OFF"}
    ):
        return jsonify({"error": "Invalid device command."}), 400

    try:
        return jsonify(send_device_command(device, action))
    except DeviceControlError as error:
        return jsonify({"error": str(error)}), error.status_code


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
