import json
import unittest
from unittest.mock import MagicMock, patch

from stevia_bot.device_control import (
    DeviceControlError,
    send_device_command,
)


class DeviceControlTests(unittest.TestCase):
    control_url = "https://devices.example.test/api/device-control"
    status_url = "https://devices.example.test/api/device-status"

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
                {"device_name": "fan", "desired_state": "ON", "actual_state": "ON"},
                {"device_name": "light", "desired_state": "OFF", "actual_state": "OFF"},
            ],
        })
        with (
            patch(
                "stevia_bot.device_control.DEVICE_CONTROL_API_URL",
                self.control_url,
            ),
            patch(
                "stevia_bot.device_control.DEVICE_STATUS_API_URL",
                self.status_url,
            ),
            patch(
                "stevia_bot.device_control.urlopen",
                side_effect=[command_response, status_response],
            ) as urlopen,
        ):
            result = send_device_command("fan", "ON")

        request = urlopen.call_args_list[0].args[0]
        self.assertEqual(request.full_url, self.control_url)
        self.assertEqual(request.full_url, self.control_url)
        self.assertEqual(request.method, "POST")
        self.assertEqual(
            json.loads(request.data),
            {"device": "fan", "action": "ON"},
        )
        status_request = urlopen.call_args_list[1].args[0]
        self.assertEqual(status_request.full_url, self.status_url)
        self.assertEqual(status_request.method, "GET")
        self.assertEqual(result["reported_state"], "ON")
        self.assertEqual(
            result["message"],
            "The fan is ON according to the latest device report.",
        )

    def test_command_reports_when_device_has_not_reached_requested_state(self):
        command_response = self.backend_response({"success": True})
        status_response = self.backend_response({
            "success": True,
            "devices": [
                {"device_name": "fan", "desired_state": "ON", "actual_state": "OFF"},
            ],
        })
        with (
            patch("stevia_bot.device_control.DEVICE_CONTROL_API_URL", self.control_url),
            patch("stevia_bot.device_control.DEVICE_STATUS_API_URL", self.status_url),
            patch(
                "stevia_bot.device_control.urlopen",
                side_effect=[command_response, status_response],
            ),
        ):
            result = send_device_command("fan", "ON")

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
        with (
            patch("stevia_bot.device_control.DEVICE_CONTROL_API_URL", self.control_url),
            patch("stevia_bot.device_control.DEVICE_STATUS_API_URL", self.status_url),
            patch(
                "stevia_bot.device_control.urlopen",
                side_effect=[command_response, status_response],
            ),
        ):
            result = send_device_command("light", "ON")

        self.assertEqual(
            result["message"],
            "The lights are ON according to the latest device report.",
        )

    def test_backend_failure_is_not_reported_as_success(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = (
            b'{"success": false, "message": "Device not found."}'
        )
        with (
            patch("stevia_bot.device_control.DEVICE_CONTROL_API_URL", self.control_url),
            patch("stevia_bot.device_control.urlopen", return_value=response),
        ):
            with self.assertRaisesRegex(DeviceControlError, "Device not found"):
                send_device_command("fan", "ON")

    def test_device_control_requires_a_configured_backend(self):
        with patch("stevia_bot.device_control.DEVICE_CONTROL_API_URL", None):
            with self.assertRaisesRegex(DeviceControlError, "not configured") as error:
                send_device_command("fan", "ON")
        self.assertEqual(error.exception.status_code, 503)

    def test_backend_token_is_forwarded_to_control_and_status_endpoints(self):
        command_response = self.backend_response({"success": True})
        status_response = self.backend_response({
            "success": True,
            "devices": [{"device_name": "fan", "actual_state": "ON"}],
        })
        with (
            patch("stevia_bot.device_control.DEVICE_CONTROL_API_URL", self.control_url),
            patch("stevia_bot.device_control.DEVICE_STATUS_API_URL", self.status_url),
            patch("stevia_bot.device_control.DEVICE_BACKEND_TOKEN", "backend-secret"),
            patch(
                "stevia_bot.device_control.urlopen",
                side_effect=[command_response, status_response],
            ) as urlopen,
        ):
            send_device_command("fan", "ON")

        self.assertEqual(
            urlopen.call_args_list[0].args[0].get_header("Authorization"),
            "Bearer backend-secret",
        )
        self.assertEqual(
            urlopen.call_args_list[1].args[0].get_header("Authorization"),
            "Bearer backend-secret",
        )


if __name__ == "__main__":
    unittest.main()
