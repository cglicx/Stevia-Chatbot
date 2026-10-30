import unittest
from unittest.mock import patch

from app import app
from stevia_bot.backend_sensor_data import SensorBackendError
from stevia_bot.chatbot import SteviaChatbot


class ChatApiTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.live_sensor_context = patch(
            "app.fetch_live_sensor_context",
            return_value=({}, [], 0),
        )
        self.live_sensor_context.start()
        self.addCleanup(self.live_sensor_context.stop)

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

        records = [
            {"recorded_at": "2024-09-27 13:28:26", "temperature": 26.7},
            {"recorded_at": "2024-09-27 13:28:13", "temperature": 26.6},
        ]
        with patch("app.fetch_live_sensor_context", return_value=({}, records, 2)):
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

    def test_database_sensor_context_and_history_override_local_files(self):
        records = [
            {
                "recorded_at": "2026-10-30 10:00:00",
                "temperature": 27.1,
                "humidity": 70.0,
                "soil_moisture": 48.0,
                "water_level": 48.0,
                "soil_ph": 6.8,
                "light_intensity": 9200.0,
            },
            {
                "recorded_at": "2026-10-30 09:59:30",
                "temperature": 26.9,
                "humidity": 71.0,
                "soil_moisture": 49.0,
                "water_level": 49.0,
                "soil_ph": 6.8,
                "light_intensity": 9100.0,
            },
        ]

        with patch("stevia_bot.chatbot.LATEST_READING", {"temperature": 12}), \
             patch(
                 "app.fetch_live_sensor_context",
                 return_value=(
                     {
                         "recorded_at": records[0]["recorded_at"],
                         "temperature": 27.1,
                         "humidity": 70.0,
                         "soil_moisture": 48.0,
                         "water_level": 48.0,
                         "soil_ph": 6.8,
                         "light_intensity": 9200.0,
                     },
                     records,
                     2,
                 ),
             ), \
             patch("stevia_bot.chatbot.load_sensor_records", side_effect=AssertionError):
            response = self.client.post(
                "/api/chat",
                json={
                    "message": "Can you send me the 2 latest sensor readings?",
                    "sensor_data": {
                        "recorded_at": records[0]["recorded_at"],
                        "temperature": 27.1,
                        "humidity": 70.0,
                        "soil_moisture": 48.0,
                        "water_level": 48.0,
                        "soil_ph": 6.8,
                        "light_intensity": 9200.0,
                    },
                    "sensor_records": records,
                    "sensor_records_total": 2,
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["sensor_data"]["temperature"], 27.1)
        self.assertEqual(payload["sensor_records"], records)
        self.assertIn("Showing 2 sensor readings", payload["answer"])

    def test_sensor_history_response_discloses_truncated_database_history(self):
        answer = SteviaChatbot._sensor_records_summary(
            [{"recorded_at": "2026-10-30 10:00:00"}],
            selection="all",
            total_records=1200,
        )

        self.assertIn("latest 1 of 1200 sensor readings", answer)

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

        with patch(
            "app.fetch_live_sensor_context",
            return_value=({}, records, len(records)),
        ):
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

        with patch(
            "app.fetch_live_sensor_context",
            return_value=({}, records, len(records)),
        ):
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

    def test_sensor_backend_failure_is_returned_instead_of_using_local_files(self):
        with patch(
            "app.fetch_live_sensor_context",
            side_effect=SensorBackendError("The sensor backend is unavailable."),
        ):
            response = self.client.post(
                "/api/chat",
                json={"message": "What is my latest temperature?"},
            )

        self.assertEqual(response.status_code, 502)
        self.assertEqual(
            response.get_json(),
            {"error": "The sensor backend is unavailable."},
        )

    def test_non_object_json_returns_bad_request(self):
        response = self.client.post("/api/chat", json=["not", "an", "object"])

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json(), {"error": "Please enter a question."})

    def test_device_command_requires_explicit_confirmation(self):
        response = self.client.post(
            "/api/device-command",
            json={"device": "pump", "action": "ON", "confirmed": False},
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
                "device": "pump",
                "action": "ON",
                "message": "Device command saved.",
            },
        ) as send_command:
            response = self.client.post(
                "/api/device-command",
                json={"device": "pump", "action": "ON", "confirmed": True},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["success"], True)
        send_command.assert_called_once_with("pump", "ON")

    def test_chat_yes_reply_executes_explicit_pending_device_context(self):
        with patch(
            "app.send_device_command",
            return_value={
                "success": True,
                "device": "pump",
                "action": "ON",
                "reported_state": "ON",
                "message": "The pump is ON according to the latest device report.",
            },
        ) as send_command:
            response = self.client.post(
                "/api/chat",
                json={
                    "message": "Yes",
                    "pending_device_command": {"device": "pump", "action": "ON"},
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["intent"], "device_control")
        self.assertEqual(
            response.get_json()["answer"],
            "The pump is ON according to the latest device report.",
        )
        send_command.assert_called_once_with("pump", "ON")

    def test_invalid_device_command_is_rejected(self):
        response = self.client.post(
            "/api/device-command",
            json={"device": "sprinkler", "action": "ON", "confirmed": True},
        )

        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
