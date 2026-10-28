import json
import os
import unittest
from unittest.mock import MagicMock, patch

from stevia_bot.device_control import DeviceControlError, send_device_command


class DeviceControlTests(unittest.TestCase):
    def test_command_is_posted_to_configured_backend(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps({
            "success": True,
            "message": "Device command saved.",
        }).encode("utf-8")
        with patch.dict(os.environ, {"DEVICE_CONTROL_API_URL": "https://device.test/command"}), \
             patch("stevia_bot.device_control.urlopen", return_value=response) as urlopen:
            result = send_device_command("fan", "ON")

        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "https://device.test/command")
        self.assertEqual(request.method, "POST")
        self.assertEqual(
            json.loads(request.data),
            {"device": "fan", "action": "ON"},
        )
        self.assertEqual(result["message"], "Device command saved.")

    def test_missing_backend_url_is_reported(self):
        with patch.dict(os.environ, {"DEVICE_CONTROL_API_URL": ""}):
            with self.assertRaisesRegex(DeviceControlError, "not configured"):
                send_device_command("fan", "ON")

    def test_backend_failure_is_not_reported_as_success(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = (
            b'{"success": false, "message": "Device not found."}'
        )
        with patch.dict(os.environ, {"DEVICE_CONTROL_API_URL": "https://device.test/command"}), \
             patch("stevia_bot.device_control.urlopen", return_value=response):
            with self.assertRaisesRegex(DeviceControlError, "Device not found"):
                send_device_command("fan", "ON")


if __name__ == "__main__":
    unittest.main()
