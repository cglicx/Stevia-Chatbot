import unittest
from unittest.mock import patch

from app import app


class ChatApiTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_current_reading_request_returns_sensor_data(self):
        response = self.client.post(
            "/api/chat",
            json={"message": "What are the latest sensor readings?"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["intent"], "sensor")
        self.assertIn("answer", payload)
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
        self.assertIn("specific number of records or the entire set", prompt["answer"])
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["intent"], "sensor_records")
        self.assertIn("1 timestamped record, newest first", payload["answer"])
        self.assertIn("temperature 26.7 C", payload["answer"])
        self.assertNotIn("2024-09-27 13:28:13", payload["answer"])
        self.assertEqual(payload["sensor_records"], [
            {"recorded_at": "2024-09-27 13:28:26", "temperature": 26.7},
        ])

    def test_blank_question_returns_bad_request(self):
        response = self.client.post("/api/chat", json={"message": "  "})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json(), {"error": "Please enter a question."})

    def test_non_object_json_returns_bad_request(self):
        response = self.client.post("/api/chat", json=["not", "an", "object"])

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json(), {"error": "Please enter a question."})


if __name__ == "__main__":
    unittest.main()
