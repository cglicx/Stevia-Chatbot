import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from fastapi_app import app


class FastApiDeviceCommandTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

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


if __name__ == "__main__":
    unittest.main()
