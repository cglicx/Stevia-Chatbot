import os
import re
from pathlib import Path
from typing import Any

from .iot_data import LATEST_READING
from .knowledge_base import KNOWLEDGE_BASE, KNOWLEDGE_SOURCE, SPROUT_KNOWLEDGE, WELCOME_MESSAGE


class SteviaChatbot:
    """Answers Stevia questions with a fine-tuned BERT classifier and deterministic fallback."""

    MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "stevia-intent-bert"
    OUT_OF_SCOPE_RESPONSE = "Sorry, your question is outside my scope. I can only answer questions related to stevia."

    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.model_error = None
        self._model_loaded = False
        self.llm_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.llm_enabled = bool(os.getenv("OPENAI_API_KEY"))

    def _load_model(self):
        if self._model_loaded:
            return
        self._model_loaded = True
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_PATH)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_PATH)
            self.model.eval()
        except Exception as error:
            self.model_error = str(error)

    def _ensure_model(self):
        if not self._model_loaded:
            self._load_model()

    def _model_scores(self, question: str) -> dict[str, float]:
        self._ensure_model()
        if not self.model or not self.tokenizer:
            return {}

        import torch

        with torch.no_grad():
            inputs = self.tokenizer(question, truncation=True, max_length=64, return_tensors="pt")
            probabilities = torch.softmax(self.model(**inputs).logits[0], dim=-1)

        return {
            label: float(probabilities[index])
            for index, label in self.model.config.id2label.items()
        }

    def _keyword_scores(self, question: str) -> dict[str, float]:
        words = set(re.findall(r"[a-z]+", question.lower()))
        scores = {}
        for intent, item in KNOWLEDGE_BASE.items():
            matches = words.intersection(item["keywords"])
            # Count matching evidence directly. Normalizing by list length made
            # the generic Stevia topic beat specific topics such as watering.
            scores[intent] = float(len(matches))
            if intent == "general" and matches == {"stevia"}:
                scores[intent] = 0.0
        return scores

    def _select_intent(self, question: str, history: list[dict[str, Any]] | None = None) -> tuple[str, str]:
        recent_question = ""
        if isinstance(history, list):
            recent_user_messages = [
                item.get("content", "") for item in history
                if isinstance(item, dict) and item.get("role") == "user"
            ]
            if recent_user_messages:
                recent_question = recent_user_messages[-1]

        # Add the previous user turn only when the short follow-up has no
        # topic of its own, such as "And indoors?".
        words = re.findall(r"[a-z]+", question.lower())
        context_question = question
        current_scores = self._keyword_scores(question)
        has_current_topic = max(current_scores.values(), default=0.0) > 0
        if recent_question and len(words) <= 5 and not has_current_topic:
            context_question = f"{recent_question} {question}"

        keyword_scores = self._keyword_scores(context_question)
        model_scores = self._model_scores(context_question)

        # A specific care word such as "water" should beat the generic
        # "plant" match in questions like "Should I water the plant?".
        specific_keyword_score = max(
            (score for intent, score in keyword_scores.items() if intent != "general"),
            default=0.0,
        )
        if specific_keyword_score > 0:
            keyword_scores["general"] = 0.0

        # Keyword evidence keeps answers stable while BERT supplies semantic context.
        combined = {
            intent: keyword_scores[intent] + model_scores.get(intent, 0.0) * 0.35
            for intent in KNOWLEDGE_BASE
        }
        intent = max(combined, key=combined.get)
        if combined[intent] == 0:
            return "general", "fallback"
        source = "fine-tuned classifier + knowledge-base" if model_scores else "knowledge-base fallback"
        return intent, source

    def _build_llm_context(self, question: str, history: list[dict[str, Any]] | None, sensor_data: dict[str, Any]) -> str:
        context_lines = [
            "You are a helpful Stevia plant-care assistant.",
            "Use the uploaded SPROUT knowledge when it is relevant to watering, soil, light, temperature, humidity, or plant health.",
        ]

        if sensor_data:
            context_lines.append(f"Current sensor snapshot: {sensor_data}")

        if SPROUT_KNOWLEDGE:
            key_items = []
            for item in SPROUT_KNOWLEDGE[:5]:
                parameter = item.get("parameter", "")
                optimal = item.get("optimal_range")
                if parameter and optimal:
                    key_items.append(f"{parameter}: {optimal[0]}-{optimal[1]} {item.get('unit', '')}")
            if key_items:
                context_lines.append("Key SPROUT targets: " + "; ".join(key_items))

        if isinstance(history, list) and history:
            recent_turns = []
            for entry in history[-6:]:
                if isinstance(entry, dict):
                    role = entry.get("role", "user")
                    content = entry.get("content", "")
                    if content:
                        recent_turns.append(f"{role}: {content}")
            if recent_turns:
                context_lines.append("Recent conversation:\n" + "\n".join(recent_turns))

        context_lines.append(f"User question: {question}")
        return "\n".join(context_lines)

    def _llm_answer(self, question: str, history: list[dict[str, Any]] | None, sensor_data: dict[str, Any]) -> str | None:
        if not os.getenv("OPENAI_API_KEY"):
            return None

        try:
            from openai import OpenAI
        except Exception as error:
            self.model_error = str(error)
            return None

        try:
            client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            prompt = self._build_llm_context(question, history, sensor_data)
            completion = client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "Answer as a knowledgeable Stevia care assistant. Keep advice practical and concise. Use the sensor snapshot, recent sensor trends, and SPROUT targets when relevant. Do not infer a trend when none is provided.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.6,
                max_tokens=300,
            )
            content = completion.choices[0].message.content
            return content.strip() if content else None
        except Exception as error:
            self.model_error = str(error)
            return None

    def answer(
        self,
        question: str,
        sensor_data: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        manual_sensor_data = sensor_data if isinstance(sensor_data, dict) else {}
        combined_sensor_data = {**LATEST_READING, **manual_sensor_data}

        if self._is_social_only(question):
            social_reply = "Hi! Ask me about your Stevia plant care, watering, light, soil, or temperature."
            if any(word in question.lower() for word in ["thank", "thanks", "welcome"]):
                social_reply = "You’re welcome! Ask me anything about your Stevia care."
            return {
                "answer": social_reply,
                "topic": "About Stevia",
                "intent": "general",
                "source": "social",
                "model_loaded": self.model is not None,
                "sensor_data": combined_sensor_data,
                "knowledge_source": KNOWLEDGE_SOURCE if SPROUT_KNOWLEDGE else "built-in fallback",
            }

        if not self._is_in_scope(question, history):
            return {
                "answer": self.OUT_OF_SCOPE_RESPONSE,
                "topic": "Outside scope",
                "intent": "out_of_scope",
                "source": "scope validation",
                "model_loaded": self.model is not None,
                "sensor_data": combined_sensor_data,
                "knowledge_source": KNOWLEDGE_SOURCE if SPROUT_KNOWLEDGE else "built-in fallback",
            }

        if self._asks_for_current_readings(question):
            return {
                "answer": self._current_reading_summary(combined_sensor_data, history),
                "topic": "Current sensor readings",
                "intent": "sensor",
                "source": "sensor-data",
                "model_loaded": self.model is not None,
                "sensor_data": combined_sensor_data,
                "knowledge_source": KNOWLEDGE_SOURCE if SPROUT_KNOWLEDGE else "built-in fallback",
            }

        intent, source = self._select_intent(question, history=history)
        sensor_note = self._sensor_note(combined_sensor_data, intent)
        if sensor_note:
            return {
                "answer": sensor_note,
                "topic": KNOWLEDGE_BASE[intent]["title"],
                "intent": intent,
                "source": "rule-based SPROUT sensor comparison",
                "model_loaded": self.model is not None,
                "sensor_data": combined_sensor_data,
                "knowledge_source": KNOWLEDGE_SOURCE if SPROUT_KNOWLEDGE else "built-in fallback",
            }

        llm_response = self._llm_answer(question, history, combined_sensor_data)
        if llm_response:
            return {
                "answer": llm_response,
                "topic": KNOWLEDGE_BASE[intent]["title"],
                "intent": intent,
                "source": "llm (OpenAI)",
                "model_loaded": self.model is not None,
                "sensor_data": combined_sensor_data,
                "knowledge_source": KNOWLEDGE_SOURCE if SPROUT_KNOWLEDGE else "built-in fallback",
            }

        response = self._choose_response(intent, question, history)

        return {
            "answer": response,
            "topic": KNOWLEDGE_BASE[intent]["title"],
            "intent": intent,
            "source": source,
            "model_loaded": self.model is not None,
            "sensor_data": combined_sensor_data,
            "knowledge_source": KNOWLEDGE_SOURCE if SPROUT_KNOWLEDGE else "built-in fallback",
        }

    @staticmethod
    def _is_social_only(question: str) -> bool:
        words = set(re.findall(r"[a-z]+", question.lower()))
        if not words:
            return False

        social_words = {
            "hi", "hello", "hey", "greetings", "yo", "morning", "afternoon", "evening",
            "goodbye", "bye", "thanks", "thank", "thankyou", "thx", "welcome",
        }
        plant_words = {"stevia", "plant", "water", "soil", "sunlight", "humidity", "temperature", "light", "grow", "growing", "care"}

        has_social = bool(words.intersection(social_words))
        has_plant_topic = bool(words.intersection(plant_words))
        return has_social and not has_plant_topic

    def _is_in_scope(
        self,
        question: str,
        history: list[dict[str, Any]] | None = None,
    ) -> bool:
        if re.search(r"\bstevia\b", question, re.IGNORECASE):
            return True
        if self._asks_for_current_readings(question):
            return True
        if any(score > 0 for score in self._keyword_scores(question).values()):
            return True

        if not isinstance(history, list):
            return False

        previous_user_messages = [
            entry.get("content", "")
            for entry in history
            if isinstance(entry, dict)
            and entry.get("role") == "user"
            and entry.get("content", "").strip().lower() != question.strip().lower()
        ]
        return any(
            re.search(r"\bstevia\b", message, re.IGNORECASE)
            or any(score > 0 for score in self._keyword_scores(message).values())
            for message in previous_user_messages
        )

    @staticmethod
    def _asks_for_current_readings(question: str) -> bool:
        words = set(re.findall(r"[a-z]+", question.lower()))
        return bool(words.intersection({"sensor", "sensors", "reading", "readings", "measurement", "measurements", "snapshot"}))

    @staticmethod
    def _normalize_response(text: str | None) -> str:
        if text is None:
            return ""
        return re.sub(r"\s+", " ", str(text)).strip().lower()

    @classmethod
    def _is_duplicate_response(cls, candidate: str, previous_answers: set[str]) -> bool:
        candidate_key = cls._normalize_response(candidate)
        for previous in previous_answers:
            previous_key = cls._normalize_response(previous)
            if not previous_key:
                continue
            if candidate_key == previous_key or candidate_key in previous_key or previous_key in candidate_key:
                return True
        return False

    @staticmethod
    def _current_reading_summary(
        sensor_data: dict[str, Any],
        history: list[dict[str, Any]] | None = None,
    ) -> str:
        if not sensor_data:
            return "No IoTProcessed_Data reading is available yet."
        recorded_at = sensor_data.get("recorded_at", "unknown time")
        temperature = sensor_data.get("temperature", "unknown")
        humidity = sensor_data.get("humidity", "unknown")
        water_level = sensor_data.get("water_level", sensor_data.get("soil_moisture", "unknown"))
        previous_answers = {
            entry.get("content") for entry in (history or [])
            if isinstance(entry, dict) and entry.get("role") == "assistant"
        }
        summaries = [
            (
                f"Latest snapshot ({recorded_at}): temperature {temperature} C, humidity {humidity}%, "
                f"and water level {water_level}%."
            ),
            (
                f"Current readings at {recorded_at}: {temperature} C with {humidity}% humidity, and the water level is {water_level}%."
            ),
            (
                f"Sensor check from {recorded_at}: climate is {temperature} C and {humidity}% humidity; water status is {water_level}%."
            ),
            (
                f"The newest recorded data is dated {recorded_at}. Temperature is {temperature} C, humidity is {humidity}%, "
                f"and the water level is {water_level}%."
            ),
            (
                f"Latest conditions: {temperature} C, {humidity}% humidity, and {water_level}% soil moisture."
            ),
        ]
        trend_phrases = []
        trend_names = {
            "temperature": "temperature",
            "humidity": "humidity",
            "water_level": "soil moisture",
        }
        trends = sensor_data.get("sensor_trends", {})
        for key, label in trend_names.items():
            direction = trends.get(key) if isinstance(trends, dict) else None
            if direction in {"rising", "falling"}:
                trend_phrases.append(f"{label} has been {direction} across recent readings")
        if trend_phrases:
            summaries = [f"{summary} Recent trend: {', '.join(trend_phrases)}." for summary in summaries]
        for summary in summaries:
            if not SteviaChatbot._is_duplicate_response(summary, previous_answers):
                return summary
        return summaries[0]

    @staticmethod
    def _choose_response(
        intent: str,
        question: str,
        history: list[dict[str, Any]] | None,
    ) -> str:
        item = KNOWLEDGE_BASE[intent]
        responses = item.get("responses", [item["response"]])
        question_words = set(re.findall(r"[a-z]+", question.lower()))

        previous_answers = {
            entry.get("content") for entry in (history or [])
            if isinstance(entry, dict) and entry.get("role") == "assistant"
        }
        if intent == "watering" and question_words.intersection({"often", "week", "weekly", "frequently"}):
            frequency_response = next(
                (response for response in responses if "There is no fixed weekly schedule" in response),
                None,
            )
            if frequency_response and not SteviaChatbot._is_duplicate_response(frequency_response, previous_answers):
                return frequency_response

        preferred = 1 if intent == "watering" and question_words.intersection({"often", "week", "weekly", "frequently"}) else 0
        ordered_responses = responses[preferred:] + responses[:preferred]
        for response in ordered_responses:
            if not SteviaChatbot._is_duplicate_response(response, previous_answers):
                return response
        return ordered_responses[0]

    @staticmethod
    def _sensor_note(sensor_data: dict[str, Any] | None, intent: str) -> str:
        """Compare supported sensor readings with SPROUT's Stevia care targets."""
        if not isinstance(sensor_data, dict):
            return ""

        profiles = {
            "temperature": ("temperature", ("temperature",), "temperature", "temperature"),
            "humidity": ("relative_humidity", ("humidity", "relative_humidity"), "humidity", "humidity"),
            "sunlight": ("light_intensity", ("light_intensity",), "light", "light_intensity"),
            "watering": ("soil_moisture", ("water_level", "soil_moisture"), "soil moisture", "water_level"),
            "soil": ("soil_pH", ("soil_ph", "soil_pH"), "soil pH", "soil_ph"),
        }
        profile = profiles.get(intent)
        if not profile:
            return ""

        dataset_parameter, sensor_keys, label, trend_key = profile
        dataset_item = next(
            (item for item in SPROUT_KNOWLEDGE if item.get("parameter", "").lower() == dataset_parameter.lower()),
            None,
        )
        if not dataset_item:
            return ""

        sensor_key = next((key for key in sensor_keys if sensor_data.get(key) is not None), None)
        reading = sensor_data.get(sensor_key) if sensor_key else None
        if reading is None:
            return ""

        optimal_range = dataset_item.get("optimal_range")
        if optimal_range:
            unit = dataset_item.get("unit", "").replace("_", " ")
            target = f"{optimal_range[0]:g}-{optimal_range[1]:g} {unit}"
        else:
            target = "consistently moist, well-drained soil; let the top inch dry before rewatering"

        units = sensor_data.get("sensor_units", {})
        actual_unit = units.get(sensor_key) if isinstance(units, dict) else None

        if dataset_parameter == "soil_moisture":
            trend = SteviaChatbot._format_sensor_trend(sensor_data, trend_key, label)
            unit_text = f" {actual_unit}" if actual_unit else "%"
            return (
                f"SPROUT's Stevia moisture target is {target}. The current soil-moisture reading is "
                f"{reading}{unit_text}.{trend} SPROUT provides no numeric dry/moist thresholds or "
                "sensor calibration for this percentage, so it cannot reliably determine whether to irrigate. "
                "As a planning guide, check the top inch daily and expect roughly 2-3 waterings per week in "
                "warm weather, less in cool weather: water only when that layer is dry, wait if it is still "
                "moist, and stop watering and improve drainage if the soil is saturated."
            )

        if dataset_parameter == "light_intensity" and actual_unit and actual_unit.lower() not in {
            "hours_direct_sun", "hours", "hours per day",
        }:
            light_target = f"{optimal_range[0]:g}-{optimal_range[1]:g} hours"
            return (
                f"SPROUT's light target is {light_target} of direct sun per day. The current light sensor reads "
                f"{reading} {actual_unit}; lux measures light intensity, not daily direct-sun hours, so this "
                "reading cannot establish whether the target is met. Track direct-sun duration; if it is "
                "under 6 hours, gradually extend exposure or add supplemental light."
            )

        try:
            numeric_reading = float(reading)
        except (TypeError, ValueError):
            return f"The current {label} sensor value ({reading}) is not numeric, so it cannot be compared with the SPROUT target of {target}."

        classification = next(
            (item for item in dataset_item.get("classifications", [])
             if item.get("range")
             and item["range"][0] <= numeric_reading <= item["range"][1]),
            None,
        )
        low, high = optimal_range
        if numeric_reading < low:
            status = "below the target range"
            fallback_action = {
                "temperature": "Move the plant to a warmer protected location and shield it from cold.",
                "relative_humidity": "Raise humidity gradually while maintaining airflow.",
                "soil_pH": "Raise pH gradually with an appropriate amendment and retest the soil.",
                "light_intensity": "Increase direct-light exposure gradually, aiming for the target duration.",
            }.get(dataset_parameter, "Adjust conditions gradually toward the target range.")
        elif numeric_reading > high:
            status = "above the target range"
            fallback_action = {
                "temperature": "Provide afternoon shade and ventilation; check soil before increasing irrigation.",
                "relative_humidity": "Improve ventilation, reduce misting, and inspect leaves for fungal spots.",
                "soil_pH": "Lower pH gradually with an appropriate amendment and retest the soil.",
                "light_intensity": "Protect the plant from harsh afternoon sun if leaves show heat stress.",
            }.get(dataset_parameter, "Adjust conditions gradually toward the target range.")
        else:
            status = "within the target range"
            fallback_action = "Maintain current conditions and keep monitoring."

        unit = dataset_item.get("unit", "").replace("_", " ")
        note = f"SPROUT target for {label} is {target}. Current reading: {numeric_reading:g} {unit}, {status}."
        if classification:
            note += f" {classification['interpretation']} Recommended action: {classification['recommendation']}"
        else:
            note += f" Recommended action: {fallback_action}"
        note += SteviaChatbot._format_sensor_trend(sensor_data, trend_key, label)
        return note

    @staticmethod
    def _format_sensor_trend(sensor_data: dict[str, Any], key: str, label: str) -> str:
        trends = sensor_data.get("sensor_trends", {})
        direction = trends.get(key) if isinstance(trends, dict) else None
        if direction in {"rising", "falling"}:
            return f" Recent readings show this value is {direction}."
        if direction == "stable":
            return f" Recent {label} readings are stable."
        return ""
