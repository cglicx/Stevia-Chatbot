import unittest
from unittest.mock import patch

from app import app
from stevia_bot.chatbot import SteviaChatbot


class ChatApiTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_unqualified_latest_readings_asks_how_many(self):
        response = self.client.post(
            "/api/chat",
            json={"message": "What are the latest sensor readings?"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["intent"], "sensor_records")
        self.assertEqual(payload["source"], "sensor-record confirmation")
        self.assertIn("How many latest sensor readings", payload["answer"])
        self.assertIn("sensor_data", payload)

    def test_sensor_records_request_confirms_then_returns_selected_records(self):
        request_text = "Show my Stevia records"
        prompt_response = self.client.post(
            "/api/chat",
            json={"message": request_text},
        )
        prompt = prompt_response.get_json()

        with patch("stevia_bot.chatbot.load_sensor_records", return_value=[
            {"recorded_at": "2024-09-27 13:28:26", "temperature": 26.7},
            {"recorded_at": "2024-09-27 13:28:13", "temperature": 26.6},
        ]):
            response = self.client.post(
                "/api/chat",
                json={
                    "message": "1",
                    "history": [
                        {"role": "user", "content": request_text},
                        {"role": "assistant", "content": prompt["answer"]},
                    ],
                },
            )

        self.assertEqual(prompt_response.status_code, 200)
        self.assertIn("How many latest sensor readings", prompt["answer"])
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["intent"], "sensor_records")
        self.assertIn("1 sensor reading (records 1-1), newest first", payload["answer"])
        self.assertEqual(payload["sensor_records"], [
            {"recorded_at": "2024-09-27 13:28:26", "temperature": 26.7},
        ])

    def test_counted_sensor_request_and_follow_ups_preserve_context(self):
        records = [
            {"recorded_at": f"2024-09-27 13:{minute}:26", "temperature": value}
            for minute, value in (
                ("28", 26.8),
                ("27", 26.7),
                ("26", 26.6),
                ("25", 26.5),
                ("24", 26.4),
                ("23", 26.3),
            )
        ]
        first_question = "Can you send me the 3 latest sensor readings?"
        conversation = [
            {"role": "user", "content": "What is Stevia?"},
            {"role": "assistant", "content": "Stevia is a sweet herb."},
            {"role": "user", "content": first_question},
        ]

        with patch("stevia_bot.chatbot.load_sensor_records", return_value=records):
            first_response = self.client.post(
                "/api/chat",
                json={"message": first_question, "history": conversation},
            )
            first_payload = first_response.get_json()
            conversation.extend([
                {"role": "assistant", "content": first_payload["answer"]},
                {"role": "user", "content": "show me the next three"},
            ])
            next_response = self.client.post(
                "/api/chat",
                json={"message": "show me the next three", "history": conversation},
            )
            next_payload = next_response.get_json()
            conversation.extend([
                {"role": "assistant", "content": next_payload["answer"]},
                {"role": "user", "content": "put them in a table"},
            ])
            table_response = self.client.post(
                "/api/chat",
                json={"message": "put them in a table", "history": conversation},
            )
            table_payload = table_response.get_json()

        self.assertEqual(first_response.status_code, 200)
        self.assertEqual(first_payload["sensor_records"], records[:3])
        self.assertIn("3 sensor readings (records 1-3), newest first", first_payload["answer"])
        self.assertEqual(next_response.status_code, 200)
        self.assertEqual(next_payload["sensor_records"], records[3:])
        self.assertIn("records 4-6", next_payload["answer"])
        self.assertEqual(table_response.status_code, 200)
        self.assertEqual(table_payload["sensor_records"], records[3:])
        self.assertIn("records 4-6", table_payload["answer"])

    def test_at_least_count_answers_readings_confirmation_not_a_new_topic(self):
        records = [
            {"recorded_at": f"2024-09-27 13:{minute}:26", "temperature": value}
            for minute, value in (("28", 26.8), ("27", 26.7), ("26", 26.6), ("25", 26.5))
        ]
        first_question = "Can you send me the latest sensor readings?"
        prompt = self.client.post(
            "/api/chat",
            json={"message": first_question},
        ).get_json()
        conversation = [
            {"role": "user", "content": first_question},
            {"role": "assistant", "content": prompt["answer"]},
            {"role": "user", "content": "atleast 3"},
        ]

        with patch("stevia_bot.chatbot.load_sensor_records", return_value=records):
            response = self.client.post(
                "/api/chat",
                json={"message": "atleast 3", "history": conversation},
            )

        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["intent"], "sensor_records")
        self.assertEqual(payload["sensor_records"], records[:3])

    def test_new_topic_can_interrupt_sensor_count_confirmation(self):
        chatbot = SteviaChatbot()
        prompt = chatbot.answer("Show me the latest sensor readings.")
        with patch.object(chatbot, "_model_scores", return_value={}), \
             patch.object(chatbot, "_llm_answer", return_value=None):
            answer = chatbot.answer(
                "What soil pH is best?",
                history=[
                    {"role": "user", "content": "Show me the latest sensor readings."},
                    {"role": "assistant", "content": prompt["answer"]},
                ],
            )

        self.assertEqual(answer["intent"], "soil")

    def test_blank_question_returns_bad_request(self):
        response = self.client.post("/api/chat", json={"message": "  "})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json(), {"error": "Please enter a question."})

    def test_non_object_json_returns_bad_request(self):
        response = self.client.post("/api/chat", json=["not", "an", "object"])

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json(), {"error": "Please enter a question."})

    def test_device_command_requires_explicit_confirmation(self):
        response = self.client.post(
            "/api/device-command",
            json={"device": "fan", "action": "ON", "confirmed": False},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.get_json(),
            {"error": "Device command confirmation is required."},
        )

    def test_confirmed_device_command_is_forwarded_to_backend(self):
        with patch(
            "app.send_device_command",
            return_value={
                "success": True,
                "device": "fan",
                "action": "ON",
                "message": "Device command saved.",
            },
        ) as send_command:
            response = self.client.post(
                "/api/device-command",
                json={"device": "fan", "action": "ON", "confirmed": True},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["success"], True)
        send_command.assert_called_once_with("fan", "ON")

    def test_invalid_device_command_is_rejected(self):
        response = self.client.post(
            "/api/device-command",
            json={"device": "sprinkler", "action": "ON", "confirmed": True},
        )

        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
