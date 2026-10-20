import unittest
from unittest.mock import patch

import pandas as pd

from stevia_bot import iot_data
from stevia_bot.chatbot import SteviaChatbot
from stevia_bot.iot_data import _trend_direction


class SensorTrendTests(unittest.TestCase):
    def test_trend_direction_uses_recent_samples(self):
        self.assertEqual(_trend_direction([20, 21, 22]), "rising")
        self.assertEqual(_trend_direction([22, 21, 20]), "falling")
        self.assertEqual(_trend_direction([20, 20, 20]), "stable")
        self.assertIsNone(_trend_direction([20, 21]))

    def test_reading_summary_mentions_only_directional_trends(self):
        summary = SteviaChatbot._current_reading_summary({
            "recorded_at": "2024-09-27 13:28:26",
            "temperature": 29,
            "humidity": 72,
            "water_level": 40,
            "sensor_trends": {
                "temperature": "rising",
                "humidity": "stable",
                "water_level": "falling",
            },
        })
        self.assertIn("temperature has been rising", summary)
        self.assertIn("soil moisture has been falling", summary)
        self.assertNotIn("humidity has been stable", summary)

    def test_topic_sensor_note_includes_matching_trend(self):
        note = SteviaChatbot._sensor_note({
            "temperature": 32,
            "sensor_trends": {"temperature": "rising"},
        }, "temperature")
        self.assertIn("Recent readings show this value is rising", note)

    def test_llm_reply_does_not_claim_bert_is_loaded(self):
        chatbot = SteviaChatbot()
        with patch("stevia_bot.chatbot.LATEST_READING", {}):
            with patch.object(chatbot, "_llm_answer", return_value="Keep the plant warm."):
                with patch.object(chatbot, "_select_intent", return_value=("temperature", "test")):
                    response = chatbot.answer("What temperature is best?")
        self.assertFalse(response["model_loaded"])

    def test_csv_fallback_returns_latest_reading_and_trends(self):
        class FakePath:
            def __init__(self, name, exists):
                self.name = name
                self._exists = exists

            def exists(self):
                return self._exists

        frame = pd.DataFrame({
            "date": ["2024-09-27 13:26:26", "2024-09-27 13:27:26", "2024-09-27 13:28:26"],
            "tempreature": [21, 24, 27],
            "humidity": [70, 70, 70],
            "water_level": [60, 50, 40],
        })
        with patch.object(iot_data, "SENSOR_READINGS_PATH", FakePath("sensor_readings", False)), \
             patch.object(iot_data, "DATASET_PATH", FakePath("IoTProcessed_Data.csv", True)), \
             patch.object(iot_data.pd, "read_csv", return_value=frame):
            reading = iot_data.load_latest_reading()

        self.assertEqual(reading["recorded_at"], "2024-09-27 13:28:26")
        self.assertEqual(reading["sensor_trends"]["temperature"], "rising")
        self.assertEqual(reading["sensor_trends"]["water_level"], "falling")

    def test_excel_history_combines_sensor_values_by_timestamp(self):
        class FakeFile:
            def __init__(self, name, exists):
                self.name = name
                self._exists = exists

            def exists(self):
                return self._exists

        class FakeFolder:
            def exists(self):
                return True

            def __truediv__(self, name):
                return FakeFile(name, name in {
                    "Environment Temperature.xlsx",
                    "Environment Humidity.xlsx",
                })

        class FakeDatasetPath:
            name = "IoTProcessed_Data.csv"

            def exists(self):
                return False

        frames = {
            "Environment Temperature.xlsx": pd.DataFrame({
                "timestamp": ["2024-09-27 13:28:13 +0530", "2024-09-27 13:28:26 +0530"],
                "value": [26.6, 26.7],
            }),
            "Environment Humidity.xlsx": pd.DataFrame({
                "timestamp": ["2024-09-27 13:28:13 +0530"],
                "value": [95.0],
            }),
        }

        def read_excel(path, header, usecols):
            self.assertEqual(header, 1)
            self.assertEqual(usecols, [0, 2])
            return frames[path.name]

        with patch.object(iot_data, "SENSOR_READINGS_PATH", FakeFolder()), \
             patch.object(iot_data, "DATASET_PATH", FakeDatasetPath()), \
             patch.object(iot_data.pd, "read_excel", side_effect=read_excel):
            records = iot_data.load_sensor_records()

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["recorded_at"], "2024-09-27 13:28:26")
        self.assertEqual(records[0]["temperature"], 26.7)
        self.assertEqual(records[1]["temperature"], 26.6)
        self.assertEqual(records[1]["humidity"], 95.0)

    def test_csv_history_fallback_returns_all_timestamped_rows(self):
        class FakePath:
            def __init__(self, exists):
                self._exists = exists

            def exists(self):
                return self._exists

        frame = pd.DataFrame({
            "date": ["2024-09-27 13:27:26", "2024-09-27 13:28:26"],
            "tempreature": [26.5, 26.7],
            "humidity": [94, 95],
            "water_level": [63, 64],
            "Fan_actuator_ON": [0, 1],
        })
        with patch.object(iot_data, "SENSOR_READINGS_PATH", FakePath(False)), \
             patch.object(iot_data, "DATASET_PATH", FakePath(True)), \
             patch.object(iot_data.pd, "read_csv", return_value=frame):
            records = iot_data.load_sensor_records()

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["recorded_at"], "2024-09-27 13:28:26")
        self.assertEqual(records[0]["temperature"], 26.7)
        self.assertTrue(records[0]["fan_on"])


if __name__ == "__main__":
    unittest.main()
