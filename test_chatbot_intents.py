import unittest
from types import SimpleNamespace
from unittest.mock import patch

from stevia_bot.chatbot import SteviaChatbot
from train_intent_classifier import split_indices


class ChatbotIntentTests(unittest.TestCase):
    def test_training_split_keeps_similar_phrasing_in_one_partition(self):
        texts = [
            "What was the highest humidity today?",
            "What was the lowest humidity today?",
            "How high was the temperature yesterday?",
            "Tell me the humidity level right now.",
            "Can you give me my plant's humidity history?",
            "Show the latest humidity reading.",
            "Did the air get too humid overnight?",
            "Is the humidity within the ideal range?",
            "How much light does Stevia need?",
            "Should I move Stevia into more sunlight?",
            "Does Stevia need shade in the afternoon?",
            "Where should I place my plant for sunlight?",
            "How many hours of light should my Stevia get?",
            "Is a bright window enough for my Stevia?",
            "Can too much sun harm Stevia leaves?",
            "Does Stevia grow well indoors?",
        ]
        labels = ["humidity"] * 8 + ["sunlight"] * 8

        splits = split_indices(texts, labels, seed=42)
        assignments = {
            index: split_name
            for split_name, indices in splits.items()
            for index in indices
        }

        self.assertEqual(set(assignments), set(range(len(texts))))
        self.assertEqual(len(assignments), len(texts))
        self.assertEqual(assignments[0], assignments[1])
        for label in set(labels):
            for split_name in ("train", "validation", "test"):
                self.assertTrue(
                    any(assignments[index] == split_name for index, value in enumerate(labels) if value == label)
                )

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

    def test_topic_change_uses_the_new_question_topic(self):
        chatbot = SteviaChatbot()
        with patch.object(chatbot, "_model_scores", return_value={}), \
             patch.object(chatbot, "_llm_answer", return_value=None):
            result = chatbot.answer(
                "How often should I water it?",
                history=[
                    {"role": "user", "content": "What temperature is best for Stevia?"},
                    {"role": "assistant", "content": "Keep Stevia warm."},
                ],
            )

        self.assertEqual(result["intent"], "watering")

    def test_user_correction_routes_to_corrected_topic(self):
        chatbot = SteviaChatbot()
        chatbot.model = SimpleNamespace(
            config=SimpleNamespace(intent_confidence_threshold=0.8)
        )
        with patch.object(
             chatbot,
             "_model_scores",
             return_value={"humidity": 0.95, "soil": 0.05},
        ), \
             patch.object(chatbot, "_llm_answer", return_value=None):
            result = chatbot.answer("I meant soil pH, not humidity.")

        self.assertEqual(result["intent"], "soil")

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

    def test_high_confidence_model_intent_overrides_conflicting_keywords(self):
        chatbot = SteviaChatbot()
        chatbot.model = SimpleNamespace(
            config=SimpleNamespace(intent_confidence_threshold=0.8)
        )
        with patch.object(
            chatbot,
            "_model_scores",
            return_value={"watering": 0.96, "growing": 0.04},
        ):
            intent, source = chatbot._select_intent("How do I care for my plant?")

        self.assertEqual(intent, "watering")
        self.assertEqual(source, "fine-tuned classifier")

    def test_low_confidence_model_defers_to_keyword_routing(self):
        chatbot = SteviaChatbot()
        chatbot.model = SimpleNamespace(
            config=SimpleNamespace(intent_confidence_threshold=0.8)
        )
        with patch.object(
            chatbot,
            "_model_scores",
            return_value={"watering": 0.55, "growing": 0.45},
        ):
            intent, _ = chatbot._select_intent("How do I care for my plant?")

        self.assertEqual(intent, "growing")

    def test_stress_signs_use_problems_classifier_score(self):
        chatbot = SteviaChatbot()
        with patch.object(chatbot, "_model_scores", return_value={"problems": 0.9}):
            intent, _ = chatbot._select_intent("What are common signs of stress in Stevia?")

        self.assertEqual(intent, "problems")

    def test_model_scores_use_saved_confidence_temperature(self):
        import torch

        chatbot = SteviaChatbot()

        class FakeConfig:
            id2label = {0: "general", 1: "watering"}
            confidence_temperature = 2.0

        class FakeModel:
            config = FakeConfig()

            def __call__(self, **kwargs):
                return type("Output", (), {"logits": torch.tensor([[2.0, 0.0]])})()

        class FakeTokenizer:
            def __call__(self, *args, **kwargs):
                return {}

        chatbot.model = FakeModel()
        chatbot.tokenizer = FakeTokenizer()
        chatbot._model_loaded = True
        scores = chatbot._model_scores("How often should I water?")

        self.assertAlmostEqual(
            scores["general"],
            torch.softmax(torch.tensor([1.0, 0.0]), dim=-1)[0].item(),
        )

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

    def test_records_request_confirms_selection_before_returning_history(self):
        chatbot = SteviaChatbot()
        records = [
            {"recorded_at": "2024-09-27 13:28:26", "temperature": 26.7, "water_tds": 118.11},
            {"recorded_at": "2024-09-27 13:28:13", "temperature": 26.7, "water_tds": 119.91},
        ]
        with patch("stevia_bot.chatbot.load_sensor_records", return_value=records), \
             patch.object(chatbot, "_llm_answer", side_effect=AssertionError("LLM should not handle records")), \
             patch.object(chatbot, "_model_scores", side_effect=AssertionError("classifier should not handle records")):
            prompt = chatbot.answer("Can you provide the records of my stevia?")
            result = chatbot.answer(
                "1",
                history=[
                    {"role": "user", "content": "Can you provide the records of my stevia?"},
                    {"role": "assistant", "content": prompt["answer"]},
                ],
            )

        self.assertIn("specific number of records or the entire set", prompt["answer"])
        self.assertEqual(prompt["source"], "sensor-record confirmation")
        self.assertEqual(result["intent"], "sensor_records")
        self.assertEqual(result["topic"], "Historical sensor records")
        self.assertIn("1 timestamped record, newest first", result["answer"])
        self.assertIn("2024-09-27 13:28:26: temperature 26.7 C, water TDS 118.11 ppm", result["answer"])
        self.assertNotIn("2024-09-27 13:28:13", result["answer"])
        self.assertEqual(len(result["sensor_records"]), 1)
        self.assertEqual(result["sensor_records"][0]["temperature"], 26.7)

    def test_all_recorded_measurements_returns_full_history_after_selection(self):
        chatbot = SteviaChatbot()
        records = [
            {"recorded_at": "2024-09-27 13:28:26", "temperature": 26.7},
            {"recorded_at": "2024-09-27 13:28:13", "temperature": 26.6},
        ]
        with patch("stevia_bot.chatbot.load_sensor_records", return_value=[]), \
             patch.object(chatbot, "_model_scores", side_effect=AssertionError("classifier should not handle records")):
            prompt = chatbot.answer("Show all recorded measurements for my plant.")
        with patch("stevia_bot.chatbot.load_sensor_records", return_value=records):
            result = chatbot.answer(
                "the entire set",
                history=[
                    {"role": "user", "content": "Show all recorded measurements for my plant."},
                    {"role": "assistant", "content": prompt["answer"]},
                ],
            )

        self.assertEqual(result["intent"], "sensor_records")
        self.assertIn("2 timestamped records, newest first", result["answer"])
        self.assertIn("2024-09-27 13:28:13", result["answer"])
        self.assertEqual(len(result["sensor_records"]), 2)

    def test_invalid_record_selection_repeats_confirmation_without_loading(self):
        chatbot = SteviaChatbot()
        prompt = chatbot.answer("Show me the sensor records.")

        with patch("stevia_bot.chatbot.load_sensor_records") as load_records:
            result = chatbot.answer(
                "yes",
                history=[
                    {"role": "user", "content": "Show me the sensor records."},
                    {"role": "assistant", "content": prompt["answer"]},
                ],
            )

        self.assertEqual(result["answer"], chatbot.SENSOR_RECORDS_CONFIRMATION)
        load_records.assert_not_called()

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