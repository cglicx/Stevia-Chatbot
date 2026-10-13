from flask import Flask, jsonify, render_template, request

from stevia_bot.chatbot import SteviaChatbot
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
    payload = request.get_json(silent=True) or {}
    question = str(payload.get("message", "")).strip()
    sensor_data = payload.get("sensor_data")
    history = payload.get("history")

    if not question:
        return jsonify({"error": "Please enter a question."}), 400

    result = chatbot.answer(question, sensor_data=sensor_data, history=history)
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
