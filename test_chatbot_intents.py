import unittest
from unittest.mock import patch

from stevia_bot.chatbot import SteviaChatbot


class ChatbotIntentTests(unittest.TestCase):
    def test_unrelated_question_gets_out_of_scope_response(self):
        chatbot = SteviaChatbot()
        with patch.object(chatbot, "_model_scores", side_effect=AssertionError("classifier should not handle unrelated questions")):
            result = chatbot.answer("What is a banana?")

        self.assertEqual(
            result["answer"],
            "Sorry, your question is outside my scope. I can only answer questions related to stevia.",
        )
        self.assertEqual(result["intent"], "out_of_scope")

    def test_stevia_question_is_not_rejected_by_scope_check(self):
        chatbot = SteviaChatbot()
        with patch.object(chatbot, "_model_scores", return_value={}):
            result = chatbot.answer("What is Stevia?")

        self.assertNotEqual(result["intent"], "out_of_scope")

    def test_short_follow_up_uses_in_scope_conversation_history(self):
        chatbot = SteviaChatbot()
        with patch.object(chatbot, "_model_scores", return_value={}), \
             patch.object(chatbot, "_llm_answer", return_value=None):
            result = chatbot.answer(
                "And indoors?",
                history=[
                    {"role": "user", "content": "What temperature is best for Stevia?"},
                    {"role": "assistant", "content": "Stevia grows best in warm conditions."},
                ],
            )

        self.assertNotEqual(result["intent"], "out_of_scope")

    def test_leaf_loss_uses_problems_classifier_score(self):
        chatbot = SteviaChatbot()
        with patch.object(chatbot, "_model_scores", return_value={"problems": 0.9}):
            intent, _ = chatbot._select_intent("Why is my stevia losing leaves?")

        self.assertEqual(intent, "problems")

    def test_stress_signs_use_problems_classifier_score(self):
        chatbot = SteviaChatbot()
        with patch.object(chatbot, "_model_scores", return_value={"problems": 0.9}):
            intent, _ = chatbot._select_intent("What are common signs of stress in Stevia?")

        self.assertEqual(intent, "problems")

    def test_sensor_question_returns_only_current_snapshot(self):
        chatbot = SteviaChatbot()
        readings = {
            "recorded_at": "2024-09-27 13:28:26",
            "temperature": 26.7,
            "humidity": 95.0,
            "water_level": 64.05,
        }
        with patch("stevia_bot.chatbot.LATEST_READING", readings), \
             patch.object(chatbot, "_llm_answer", side_effect=AssertionError("LLM should not handle readings")), \
             patch.object(chatbot, "_model_scores", side_effect=AssertionError("classifier should not handle readings")):
            result = chatbot.answer("What is the latest sensor reading?")

        self.assertEqual(result["intent"], "sensor")
        self.assertEqual(result["topic"], "Current sensor readings")
        self.assertIn("temperature 26.7 C", result["answer"])
        self.assertIn("humidity 95.0%", result["answer"])
        self.assertIn("water level 64.05%", result["answer"])
        self.assertNotIn("soil-moisture classifications", result["answer"])

    def test_weekly_watering_advice_uses_reading_without_inventing_thresholds(self):
        chatbot = SteviaChatbot()
        readings = {
            "water_level": 64.05,
            "sensor_units": {"water_level": "percent"},
            "sensor_trends": {"water_level": "stable"},
        }
        with patch("stevia_bot.chatbot.LATEST_READING", readings), \
             patch.object(chatbot, "_select_intent", return_value=("watering", "test")):
            result = chatbot.answer("How often should I water my stevia plant each week?")

        self.assertIn("2-3 waterings per week", result["answer"])
        self.assertIn("64.05 percent", result["answer"])
        self.assertIn("no numeric dry/moist thresholds", result["answer"])
        self.assertIn("water only when that layer is dry", result["answer"])
        self.assertEqual(result["source"], "rule-based SPROUT sensor comparison")

    def test_current_temperature_question_gets_parameter_comparison(self):
        chatbot = SteviaChatbot()
        with patch("stevia_bot.chatbot.LATEST_READING", {"temperature": 26.7}), \
             patch.object(chatbot, "_select_intent", return_value=("temperature", "test")):
            result = chatbot.answer("What is the current temperature?")

        self.assertIn("18-28 celsius", result["answer"])
        self.assertIn("within the target range", result["answer"])
        self.assertNotIn("Latest snapshot", result["answer"])

    def test_light_lux_is_not_mistaken_for_direct_sun_hours(self):
        note = SteviaChatbot._sensor_note({
            "light_intensity": 43.74,
            "sensor_units": {"light_intensity": "lux"},
        }, "sunlight")

        self.assertIn("6-8 hours of direct sun per day", note)
        self.assertIn("43.74 lux", note)
        self.assertIn("cannot establish whether the target is met", note)

    def test_sensor_field_aliases_compare_humidity_and_soil_ph(self):
        humidity_note = SteviaChatbot._sensor_note({"humidity": 95.0}, "humidity")
        soil_note = SteviaChatbot._sensor_note({"soil_ph": 6.1}, "soil")

        self.assertIn("95 percent, above the target range", humidity_note)
        self.assertIn("Improve ventilation", humidity_note)
        self.assertIn("6.1 pH, below the target range", soil_note)
        self.assertIn("raise pH gradually", soil_note)

    def test_humidity_question_does_not_crash_chat_response(self):
        chatbot = SteviaChatbot()
        with patch("stevia_bot.chatbot.LATEST_READING", {"humidity": 55.0}), \
             patch.object(chatbot, "_llm_answer", return_value=None):
            result = chatbot.answer("Should I mist Stevia leaves to raise humidity?")

        self.assertEqual(result["intent"], "humidity")
        self.assertIn("below the target range", result["answer"])
        self.assertIn("Activate misting/humidification", result["answer"])


if __name__ == "__main__":
    unittest.main()