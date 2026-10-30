import json
import unittest
from unittest.mock import MagicMock, patch

from stevia_bot.device_control import (
    DEVICE_CONTROL_API_URL,
    DEVICE_STATUS_API_URL,
    DeviceControlError,
    send_device_command,
)


class DeviceControlTests(unittest.TestCase):
    @staticmethod
    def backend_response(payload):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(payload).encode(
            "utf-8"
        )
        return response

    def test_command_is_posted_to_configured_backend(self):
        command_response = self.backend_response({
            "success": True,
            "message": "Device command saved.",
        })
        status_response = self.backend_response({
            "success": True,
            "devices": [
                {"device_name": "pump", "desired_state": "ON", "actual_state": "ON"},
            ],
        })
        with patch(
            "stevia_bot.device_control.urlopen",
            side_effect=[command_response, status_response],
        ) as urlopen:
            result = send_device_command("pump", "ON")

        request = urlopen.call_args_list[0].args[0]
        self.assertEqual(request.full_url, DEVICE_CONTROL_API_URL)
        self.assertEqual(
            request.full_url,
            "http://192.168.1.20/Capstone/Backend/index.php?url=device-control",
        )
        self.assertEqual(request.method, "POST")
        self.assertEqual(
            json.loads(request.data),
            {"device": "pump", "action": "ON"},
        )
        status_request = urlopen.call_args_list[1].args[0]
        self.assertEqual(status_request.full_url, DEVICE_STATUS_API_URL)
        self.assertEqual(status_request.method, "GET")
        self.assertEqual(result["reported_state"], "ON")
        self.assertEqual(
            result["message"],
            "The pump is ON according to the latest device report.",
        )

    def test_command_reports_when_device_has_not_reached_requested_state(self):
        command_response = self.backend_response({"success": True})
        status_response = self.backend_response({
            "success": True,
            "devices": [
                {"device_name": "pump", "desired_state": "ON", "actual_state": "OFF"},
            ],
        })
        with patch(
            "stevia_bot.device_control.urlopen",
            side_effect=[command_response, status_response],
        ):
            result = send_device_command("pump", "ON")

        self.assertEqual(result["reported_state"], "OFF")
        self.assertIn("latest reported state is OFF", result["message"])

    def test_confirmed_light_state_uses_plural_response(self):
        command_response = self.backend_response({"success": True})
        status_response = self.backend_response({
            "success": True,
            "devices": [
                {"device_name": "light", "desired_state": "ON", "actual_state": "ON"},
            ],
        })
        with patch(
            "stevia_bot.device_control.urlopen",
            side_effect=[command_response, status_response],
        ):
            result = send_device_command("light", "ON")

        self.assertEqual(
            result["message"],
            "The lights are ON according to the latest device report.",
        )

    def test_unsupported_device_is_rejected_without_a_backend_request(self):
        with patch("stevia_bot.device_control.urlopen") as urlopen:
            with self.assertRaisesRegex(DeviceControlError, "Unsupported device"):
                send_device_command("fan", "ON")
        urlopen.assert_not_called()

    def test_backend_failure_is_not_reported_as_success(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = (
            b'{"success": false, "message": "Device not found."}'
        )
        with patch("stevia_bot.device_control.urlopen", return_value=response):
            with self.assertRaisesRegex(DeviceControlError, "Device not found"):
                send_device_command("pump", "ON")


if __name__ == "__main__":
    unittest.main()
