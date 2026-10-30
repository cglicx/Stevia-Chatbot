import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from fastapi_app import app


class FastApiDeviceCommandTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_check_is_available_without_api_authentication(self):
        response = self.client.get("/healthz")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_api_requires_configured_bearer_token(self):
        with patch("fastapi_app.CHATBOT_API_KEY", "a" * 32):
            unauthorized = self.client.post(
                "/api/chat",
                json={"message": "How do I water Stevia?"},
            )
            authorized = self.client.post(
                "/api/chat",
                headers={"Authorization": f"Bearer {'a' * 32}"},
                json={"message": "How do I water Stevia?"},
            )

        self.assertEqual(unauthorized.status_code, 401)
        self.assertEqual(authorized.status_code, 200)

    def test_allowed_browser_origin_can_call_api_without_bearer_token(self):
        with (
            patch("fastapi_app.CHATBOT_API_KEY", "a" * 32),
            patch("fastapi_app.cors_origins", ["https://chat.example.com"]),
        ):
            response = self.client.post(
                "/api/chat",
                headers={"Origin": "https://chat.example.com"},
                json={"message": "How do I water Stevia?"},
            )

        self.assertEqual(response.status_code, 200)

    def test_device_request_returns_mobile_confirmation_options(self):
        response = self.client.post(
            "/api/chat",
            json={"message": "open sprinkler"},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["device_command"], {"device": "pump", "action": "ON"})
        self.assertEqual(
            [option["label"] for option in payload["confirmation"]["options"]],
            ["Yes", "No"],
        )
        self.assertEqual(
            payload["quick_replies"],
            [{"text": "Yes", "value": "yes"}, {"text": "No", "value": "no"}],
        )
        self.assertEqual(payload["buttons"], payload["quick_replies"])
        self.assertEqual(payload["deviceCommand"], payload["device_command"])

    def test_chat_yes_reply_executes_pending_device_command_and_confirms_state(self):
        with patch(
            "fastapi_app.send_device_command",
            return_value={
                "success": True,
                "device": "fan",
                "action": "ON",
                "reported_state": "ON",
                "message": "The fan is ON according to the latest device report.",
            },
        ) as send_command:
            response = self.client.post(
                "/api/chat",
                json={
                    "message": "Yes",
                    "history": [
                        {"role": "user", "content": "open fan"},
                        {
                            "role": "assistant",
                            "content": "Would you like me to turn on the fan?",
                        },
                        {"role": "user", "content": "Yes"},
                    ],
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["intent"], "device_control")
        self.assertEqual(
            payload["answer"],
            "The fan is ON according to the latest device report.",
        )
        self.assertEqual(payload["device_result"]["reported_state"], "ON")
        self.assertNotIn("execute_device_command", payload)
        send_command.assert_called_once_with("fan", "ON")

    def test_chat_no_reply_cancels_pending_device_command(self):
        with patch("fastapi_app.send_device_command") as send_command:
            response = self.client.post(
                "/api/chat",
                json={
                    "message": "No",
                    "history": [
                        {"role": "user", "content": "open fan"},
                        {
                            "role": "assistant",
                            "content": "Would you like me to turn on the fan?",
                        },
                        {"role": "user", "content": "No"},
                    ],
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn("Cancelled", response.json()["answer"])
        send_command.assert_not_called()

    def test_chat_yes_reply_executes_device_with_explicit_context(self):
        with patch(
            "fastapi_app.send_device_command",
            return_value={
                "success": True,
                "device": "light",
                "action": "ON",
                "reported_state": "ON",
                "message": "The lights are ON according to the latest device report.",
            },
        ) as send_command:
            response = self.client.post(
                "/api/chat",
                json={
                    "message": "Yes",
                    "pending_device_command": {"device": "light", "action": "ON"},
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["answer"],
            "The lights are ON according to the latest device report.",
        )
        send_command.assert_called_once_with("light", "ON")

    def test_chat_no_reply_with_explicit_context_does_not_execute_device(self):
        with patch("fastapi_app.send_device_command") as send_command:
            response = self.client.post(
                "/api/chat",
                json={
                    "message": "No",
                    "pending_device_command": {"device": "fan", "action": "ON"},
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn("Cancelled", response.json()["answer"])
        send_command.assert_not_called()

    def test_chat_yes_accepts_camel_case_pending_context_and_punctuation(self):
        with patch(
            "fastapi_app.send_device_command",
            return_value={
                "success": True,
                "device": "fan",
                "action": "ON",
                "reported_state": "ON",
                "message": "The fan is ON according to the latest device report.",
            },
        ) as send_command:
            response = self.client.post(
                "/api/chat",
                json={
                    "message": "Yes!",
                    "pendingDeviceCommand": {"device": "fan", "action": "ON"},
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["intent"], "device_control")
        send_command.assert_called_once_with("fan", "ON")

    def test_chat_yes_recognizes_bot_role_in_history(self):
        with patch(
            "fastapi_app.send_device_command",
            return_value={
                "success": True,
                "device": "fan",
                "action": "ON",
                "reported_state": "ON",
                "message": "The fan is ON according to the latest device report.",
            },
        ) as send_command:
            response = self.client.post(
                "/api/chat",
                json={
                    "message": "Yes",
                    "history": [
                        {"role": "user", "content": "open fan"},
                        {
                            "role": "bot",
                            "content": "Would you like me to turn on the fan?",
                        },
                    ],
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["intent"], "device_control")
        send_command.assert_called_once_with("fan", "ON")

    def test_device_command_is_forwarded_only_after_confirmation(self):
        with patch(
            "fastapi_app.send_device_command",
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
        send_command.assert_called_once_with("fan", "ON")

    def test_unconfirmed_mobile_command_is_not_forwarded(self):
        with patch("fastapi_app.send_device_command") as send_command:
            response = self.client.post(
                "/api/device-command",
                json={"device": "fan", "action": "ON", "confirmed": False},
            )

        self.assertEqual(response.status_code, 400)
        send_command.assert_not_called()

    def test_device_command_reports_missing_backend_configuration(self):
        with patch("stevia_bot.device_control.DEVICE_CONTROL_API_URL", None):
            response = self.client.post(
                "/api/device-command",
                json={"device": "fan", "action": "ON", "confirmed": True},
            )

        self.assertEqual(response.status_code, 503)


if __name__ == "__main__":
    unittest.main()
