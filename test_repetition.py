import unittest

from stevia_bot.chatbot import SteviaChatbot


class RepetitionTests(unittest.TestCase):
    def test_sensor_summary_uses_a_different_variant_when_available(self):
        previous_answer = (
            "Latest snapshot (2024-09-27 13:28:26): temperature 26.7 C, humidity 95.0%, "
            "and water level 64.05%."
        )
        summary = SteviaChatbot._current_reading_summary(
            {
                "recorded_at": "2024-09-27 13:28:26",
                "temperature": 26.7,
                "humidity": 95.0,
                "water_level": 64.05,
            },
            [{"role": "assistant", "content": previous_answer}],
        )

        self.assertNotEqual(summary, previous_answer)
        self.assertIn("2024-09-27 13:28:26", summary)

    def test_watering_frequency_prefers_frequency_answer(self):
        answer = SteviaChatbot._choose_response(
            "watering",
            "How often should I water it?",
            history=[],
        )

        self.assertIn("There is no fixed weekly schedule", answer)

    def test_repeated_answer_is_detected_after_whitespace_normalization(self):
        self.assertTrue(
            SteviaChatbot._is_duplicate_response(
                "Keep the soil evenly moist.",
                {"Keep  the\nsoil evenly moist."},
            )
        )


if __name__ == "__main__":
    unittest.main()
