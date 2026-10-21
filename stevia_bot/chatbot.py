import logging
import os
import re
from pathlib import Path
from typing import Any

from .iot_data import LATEST_READING, load_sensor_records
from .knowledge_base import KNOWLEDGE_BASE, KNOWLEDGE_SOURCE, SPROUT_KNOWLEDGE

_LOGGER = logging.getLogger(__name__)
_TYPOGRAPHY_KEYWORDS = {
    word.lower()
    for item in KNOWLEDGE_BASE.values()
    for keyword in item["keywords"]
    for word in keyword.split()
}
_TYPOGRAPHY_KEYWORDS.update({
    "sensor", "sensors", "reading", "readings", "measurement", "measurements",
    "snapshot", "record", "records", "recorded", "log", "logs", "history",
    "historical", "past", "previous",
    "give", "latest", "newest", "recent", "last", "send", "show", "get",
    "tell", "provide", "fetch", "retrieve", "next", "all", "entire",
})
_TYPOGRAPHY_ALIASES = {
    "stvia": "stevia",
}
_TYPOGRAPHY_TWO_EDIT_WORDS = {
    keyword for keyword in _TYPOGRAPHY_KEYWORDS if len(keyword) >= 7
} | {"give", "latest"}
_TYPOGRAPHY_COMMON_WORDS = {
    "what", "who", "when", "where", "why", "how", "is", "are", "the", "a",
    "an", "of", "to", "for", "me", "my", "it", "can", "you", "should",
    "does", "do", "this", "that", "best", "each", "week", "today", "now",
    "based", "on", "see",
}
_TYPOGRAPHY_CONFIRMATION_PREFIX = "Typo check: Did you mean"


class SteviaChatbot:
    """Answers Stevia questions with a fine-tuned BERT classifier and deterministic fallback."""

    MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "stevia-intent-bert"
    _TYPOGRAPHY_CONFIRMATION_PREFIX = _TYPOGRAPHY_CONFIRMATION_PREFIX
    OUT_OF_SCOPE_RESPONSE = "Sorry, your question is outside my scope. I can only answer questions related to stevia."
    SENSOR_RECORDS_CONFIRMATION = (
        "How many latest sensor readings would you like? Reply with a number (for example, 3) "
        'or "all".'
    )

    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.model_error = None
        self._model_loaded = False
        self.llm_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

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
            _LOGGER.warning("Unable to load the intent classifier", exc_info=True)

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
            temperature = max(float(getattr(self.model.config, "confidence_temperature", 1.0)), 1e-6)
            probabilities = torch.softmax(self.model(**inputs).logits[0] / temperature, dim=-1)

        return {
            label: float(probabilities[index])
            for index, label in self.model.config.id2label.items()
        }

    def _keyword_scores(self, question: str) -> dict[str, float]:
        words = set(re.findall(r"[a-z]+", question.lower()))
        negated_words = self._negated_words(question)
        scores = {}
        for intent, item in KNOWLEDGE_BASE.items():
            matches = words.intersection(item["keywords"]) - negated_words
            # Count matching evidence directly. Normalizing by list length made
            # the generic Stevia topic beat specific topics such as watering.
            scores[intent] = float(len(matches))
            if intent == "general" and matches == {"stevia"}:
                scores[intent] = 0.0
        return scores

    @staticmethod
    def _is_single_typo(word: str, candidate: str) -> bool:
        if abs(len(word) - len(candidate)) > 1 or word == candidate:
            return False

        if len(word) == len(candidate):
            differences = [
                index for index, (left, right) in enumerate(zip(word, candidate))
                if left != right
            ]
            if len(differences) == 1:
                return True
            return (
                len(differences) == 2
                and differences[1] == differences[0] + 1
                and word[differences[0]] == candidate[differences[1]]
                and word[differences[1]] == candidate[differences[0]]
            )

        shorter, longer = sorted((word, candidate), key=len)
        shorter_index = longer_index = 0
        skipped_character = False
        while shorter_index < len(shorter) and longer_index < len(longer):
            if shorter[shorter_index] == longer[longer_index]:
                shorter_index += 1
                longer_index += 1
            elif skipped_character:
                return False
            else:
                skipped_character = True
                longer_index += 1
        return True

    @classmethod
    def _correct_typographical_errors(cls, question: str) -> str:
        corrections = [
            correction
            for correction in cls._typographical_corrections(question)
            if correction[4] == 1
        ]
        return cls._apply_typographical_corrections(question, corrections)

    @classmethod
    def _apply_typographical_corrections(
        cls,
        question: str,
        corrections: list[tuple[int, int, str, str, int]],
    ) -> str:
        corrected = question
        for start, end, word, replacement, distance in reversed(corrections):
            corrected = corrected[:start] + cls._match_word_case(word, replacement) + corrected[end:]
        return corrected

    @staticmethod
    def _match_word_case(word: str, replacement: str) -> str:
        if word.isupper():
            return replacement.upper()
        if word.istitle():
            return replacement.title()
        return replacement

    @classmethod
    def _typographical_corrections(
        cls,
        question: str,
    ) -> list[tuple[int, int, str, str, int]]:
        corrections = []
        for match in re.finditer(r"[a-zA-Z]+", question):
            word = match.group()
            if len(word) < 4:
                continue
            if word.lower() in _TYPOGRAPHY_KEYWORDS or word.lower() in _TYPOGRAPHY_COMMON_WORDS:
                continue

            candidates = []
            alias = _TYPOGRAPHY_ALIASES.get(word.lower())
            if alias:
                candidates.append((alias, 1))
            for keyword in _TYPOGRAPHY_KEYWORDS:
                if len(keyword) < 4 or keyword == word.lower():
                    continue
                if cls._is_single_typo(word.lower(), keyword):
                    candidates.append((keyword, 1))
                elif keyword in _TYPOGRAPHY_TWO_EDIT_WORDS:
                    distance = cls._edit_distance(word.lower(), keyword, limit=2)
                    if distance == 2:
                        candidates.append((keyword, distance))

            if not candidates:
                continue
            best_distance = min(distance for _, distance in candidates)
            best_candidates = {
                candidate
                for candidate, distance in candidates
                if distance == best_distance
            }
            if len(best_candidates) > 1:
                best_candidates = cls._resolve_typo_candidate_tie(
                    question,
                    match.start(),
                    match.end(),
                    best_candidates,
                )
            if len(best_candidates) == 1:
                corrections.append((
                    match.start(),
                    match.end(),
                    word,
                    best_candidates.pop(),
                    best_distance,
                ))
        return corrections

    @staticmethod
    def _resolve_typo_candidate_tie(
        question: str,
        start: int,
        end: int,
        candidates: set[str],
    ) -> set[str]:
        before = question[:start].lower()
        after = question[end:].lower()
        following_word_match = re.match(r"\W*([a-z]+)", after)
        following_word = following_word_match.group(1) if following_word_match else ""
        preceding_words = re.findall(r"[a-z]+", before)
        if (
            len(preceding_words) >= 2
            and preceding_words[-2:] == ["can", "you"]
            and following_word == "me"
            and "give" in candidates
        ):
            return {"give"}

        has_count_before = bool(re.search(r"\b\d+\s+(?:[a-z]+\s+)?$", before))
        if (
            has_count_before
            and candidates.intersection({"reading", "readings", "record", "records"})
        ):
            plural_candidates = {candidate for candidate in candidates if candidate.endswith("s")}
            if plural_candidates:
                candidates = plural_candidates

        recency_candidates = candidates.intersection({"latest", "newest", "recent", "last"})
        follows_reading_term = following_word in {
            "reading", "readings", "sensor", "sensors", "record", "records",
        } or any(
            SteviaChatbot._is_single_typo(following_word, reading_term)
            or SteviaChatbot._edit_distance(following_word, reading_term, limit=2) <= 2
            for reading_term in ("reading", "readings")
        )
        if recency_candidates and follows_reading_term:
            if "latest" in recency_candidates:
                return {"latest"}
            if "newest" in recency_candidates:
                return {"newest"}
        return candidates

    @staticmethod
    def _edit_distance(left: str, right: str, limit: int) -> int:
        if abs(len(left) - len(right)) > limit:
            return limit + 1

        previous = list(range(len(right) + 1))
        for left_index, left_character in enumerate(left, start=1):
            current = [left_index]
            for right_index, right_character in enumerate(right, start=1):
                current.append(min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_character != right_character),
                ))
            if min(current) > limit:
                return limit + 1
            previous = current
        return previous[-1]

    @classmethod
    def _pending_typo_confirmation(
        cls,
        history: list[dict[str, Any]] | None,
    ) -> tuple[str, str] | None:
        if not isinstance(history, list):
            return None
        last_assistant_index = next(
            (
                index
                for index in range(len(history) - 1, -1, -1)
                if isinstance(history[index], dict)
                and history[index].get("role") == "assistant"
            ),
            None,
        )
        if last_assistant_index is None:
            return None

        last_assistant = str(history[last_assistant_index].get("content", ""))
        confirmation_pattern = (
            rf'{re.escape(cls._TYPOGRAPHY_CONFIRMATION_PREFIX)}: \[(.+)\]\? '
            r'Reply "yes" to use it, or "no" to leave the message unchanged\.'
        )
        confirmation_match = re.fullmatch(
            confirmation_pattern,
            last_assistant,
            re.IGNORECASE,
        )
        original_question = next(
            (
                str(entry.get("content", ""))
                for entry in reversed(history[:last_assistant_index])
                if isinstance(entry, dict) and entry.get("role") == "user"
            ),
            "",
        )
        if confirmation_match:
            prompt_index = last_assistant_index
            corrected_question = confirmation_match.group(1)
        else:
            prompt_index = last_assistant_index
            previous_user_message = next(
                (
                    str(entry.get("content", "")).strip().casefold()
                    for entry in reversed(history[:last_assistant_index])
                    if isinstance(entry, dict) and entry.get("role") == "user"
                ),
                "",
            )
            if previous_user_message not in {"no", "n", "nope", "not that"}:
                return None
            earlier_confirmation = next(
                (
                    str(entry.get("content", ""))
                    for entry in reversed(history[:last_assistant_index])
                    if isinstance(entry, dict)
                    and entry.get("role") == "assistant"
                    and re.fullmatch(
                        confirmation_pattern,
                        str(entry.get("content", "")),
                        re.IGNORECASE,
                    )
                ),
                None,
            )
            if earlier_confirmation is None:
                return None
            prompt_match = re.fullmatch(confirmation_pattern, earlier_confirmation, re.IGNORECASE)
            if prompt_match is None:
                return None
            corrected_question = prompt_match.group(1)

        if not original_question:
            return None
        return original_question, corrected_question

    @staticmethod
    def _negated_words(question: str) -> set[str]:
        return set(re.findall(
            r"\b(?:not|instead of|rather than)\s+(?:about\s+)?([a-z]+)\b",
            question.lower(),
        ))

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

        if model_scores and self.model is not None:
            model_intent = max(model_scores, key=model_scores.get)
            confidence_threshold = float(
                getattr(self.model.config, "intent_confidence_threshold", 1.0)
            )
            intent_keywords = {
                word
                for keyword in KNOWLEDGE_BASE.get(model_intent, {}).get("keywords", [])
                for word in keyword.lower().split()
            }
            negated_intent = bool(intent_keywords.intersection(self._negated_words(context_question)))
            if (
                model_intent in KNOWLEDGE_BASE
                and not negated_intent
                and model_scores[model_intent] >= confidence_threshold
            ):
                return model_intent, "fine-tuned classifier"

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
            _LOGGER.warning("OpenAI support is unavailable", exc_info=True)
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
            _LOGGER.warning("OpenAI response generation failed", exc_info=True)
            return None

    def answer(
        self,
        question: str,
        sensor_data: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        manual_sensor_data = sensor_data if isinstance(sensor_data, dict) else {}
        combined_sensor_data = {**LATEST_READING, **manual_sensor_data}
        history = self._history_before_current(question, history)
        pending_typo = self._pending_typo_confirmation(history)
        if pending_typo:
            original_question, corrected_question = pending_typo
            reply = question.strip().casefold()
            if reply in {"yes", "y", "correct", "that's right", "that is right"}:
                question = corrected_question
            elif reply in {"no", "n", "nope", "not that"}:
                return self._response(
                    "Okay. Please rephrase your request with the intended words.",
                    "Typo clarification",
                    "typo_clarification",
                    "typo confirmation",
                    combined_sensor_data,
                )
            else:
                question = self._correct_typographical_errors(question)
        corrections = self._typographical_corrections(question)
        requires_confirmation = (
            len(corrections) > 1
            or any(correction[4] > 1 for correction in corrections)
        )
        if requires_confirmation:
            suggested_question = self._apply_typographical_corrections(
                question,
                corrections,
            )
            return self._response(
                f"{self._TYPOGRAPHY_CONFIRMATION_PREFIX}: [{suggested_question}]? "
                'Reply "yes" to use it, or "no" to leave the message unchanged.',
                "Typo clarification",
                "typo_clarification",
                "typo confirmation",
                combined_sensor_data,
            )
        question = self._correct_typographical_errors(question)

        if self._is_social_only(question):
            social_reply = "Hi! Ask me about your Stevia plant care, watering, light, soil, or temperature."
            if any(word in question.lower() for word in ["thank", "thanks", "welcome"]):
                social_reply = "You’re welcome! Ask me anything about your Stevia care."
            return self._response(
                social_reply, "About Stevia", "general", "social", combined_sensor_data
            )

        if self._is_awaiting_sensor_records_selection(history):
            selection = self._sensor_records_selection(question)
            if selection is None and not self._is_new_topic_question(question):
                return self._response(
                    self.SENSOR_RECORDS_CONFIRMATION,
                    "Historical sensor records",
                    "sensor_records",
                    "sensor-record confirmation",
                    combined_sensor_data,
                )

            if selection is not None:
                records = load_sensor_records()
                selected_records = records if selection == "all" else records[:selection]
                return self._response(
                    self._sensor_records_summary(selected_records, selection),
                    "Historical sensor records",
                    "sensor_records",
                    "sensor-data",
                    combined_sensor_data,
                    sensor_records=selected_records,
                )

        if self._asks_for_multiple_latest_readings(question):
            selection = self._sensor_records_selection(question)
            records = load_sensor_records()
            selected_records = records[:selection] if isinstance(selection, int) else records
            return self._response(
                self._sensor_records_summary(selected_records, selection),
                "Historical sensor records",
                "sensor_records",
                "sensor-data",
                combined_sensor_data,
                sensor_records=selected_records,
            )

        follow_up = self._sensor_records_follow_up(question, history)
        if follow_up is not None:
            limit, offset = follow_up
            records = load_sensor_records()
            selected_records = records[offset:] if limit == "all" else records[offset:offset + limit]
            return self._response(
                self._sensor_records_summary(selected_records, limit, offset),
                "Historical sensor records",
                "sensor_records",
                "sensor-data",
                combined_sensor_data,
                sensor_records=selected_records,
            )

        if not self._is_in_scope(question, history):
            return self._response(
                self.OUT_OF_SCOPE_RESPONSE,
                "Outside scope",
                "out_of_scope",
                "scope validation",
                combined_sensor_data,
            )

        if self._asks_for_sensor_records(question):
            return self._response(
                self.SENSOR_RECORDS_CONFIRMATION,
                "Historical sensor records",
                "sensor_records",
                "sensor-record confirmation",
                combined_sensor_data,
            )

        if self._asks_for_latest_reading_count(question):
            return self._response(
                self.SENSOR_RECORDS_CONFIRMATION,
                "Historical sensor records",
                "sensor_records",
                "sensor-record confirmation",
                combined_sensor_data,
            )

        if self._asks_for_sensor_assessment(question):
            return self._response(
                self._sensor_reading_assessment(combined_sensor_data),
                "Sensor-based plant check",
                "problems",
                "rule-based sensor assessment",
                combined_sensor_data,
            )

        if self._asks_for_current_readings(question):
            return self._response(
                self._current_reading_summary(combined_sensor_data, history),
                "Current sensor readings",
                "sensor",
                "sensor-data",
                combined_sensor_data,
            )

        intent, source = self._select_intent(question, history=history)
        sensor_note = self._sensor_note(combined_sensor_data, intent)
        if sensor_note:
            return self._response(
                sensor_note,
                KNOWLEDGE_BASE[intent]["title"],
                intent,
                "rule-based SPROUT sensor comparison",
                combined_sensor_data,
            )

        llm_response = self._llm_answer(question, history, combined_sensor_data)
        if llm_response:
            return self._response(
                llm_response,
                KNOWLEDGE_BASE[intent]["title"],
                intent,
                "llm (OpenAI)",
                combined_sensor_data,
            )

        response = self._choose_response(intent, question, history)

        return self._response(
            response, KNOWLEDGE_BASE[intent]["title"], intent, source, combined_sensor_data
        )

    def _response(
        self,
        answer: str,
        topic: str,
        intent: str,
        source: str,
        sensor_data: dict[str, Any],
        sensor_records: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        response = {
            "answer": answer,
            "topic": topic,
            "intent": intent,
            "source": source,
            "model_loaded": self.model is not None,
            "sensor_data": sensor_data,
            "knowledge_source": KNOWLEDGE_SOURCE if SPROUT_KNOWLEDGE else "built-in fallback",
        }
        if sensor_records is not None:
            response["sensor_records"] = sensor_records
        return response

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

    @staticmethod
    def _history_before_current(
        question: str,
        history: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]] | None:
        if not isinstance(history, list) or not history:
            return history
        last_entry = history[-1]
        if (
            isinstance(last_entry, dict)
            and last_entry.get("role") == "user"
            and str(last_entry.get("content", "")).strip().casefold() == question.strip().casefold()
        ):
            return history[:-1]
        return history

    @classmethod
    def _replace_word(cls, text: str, word: str, replacement: str) -> str:
        return re.sub(
            rf"\b{re.escape(word)}\b",
            lambda match: cls._match_word_case(match.group(), replacement),
            text,
            count=1,
            flags=re.IGNORECASE,
        )

    def _is_new_topic_question(self, question: str) -> bool:
        words = re.findall(r"[a-z]+", question.lower())
        return (
            any(score > 0 for score in self._keyword_scores(question).values())
            or (
                len(words) > 2
                and words[0] in {"what", "how", "where", "why", "when", "who", "which"}
                and words[:2] != ["how", "many"]
            )
        )

    def _is_in_scope(
        self,
        question: str,
        history: list[dict[str, Any]] | None = None,
    ) -> bool:
        if re.search(r"\bstevia\b", question, re.IGNORECASE):
            return True
        if self._asks_for_current_readings(question):
            return True
        if self._asks_for_sensor_records(question):
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
    def _asks_for_sensor_assessment(question: str) -> bool:
        words = set(re.findall(r"[a-z]+", question.lower()))
        asks_about_health = bool(words.intersection({
            "problem", "problems", "issue", "issues", "wrong", "healthy", "health",
            "concern", "concerns", "trouble", "disease",
        }))
        refers_to_readings = bool(words.intersection({
            "reading", "readings", "sensor", "sensors", "data", "based",
        }))
        return asks_about_health and refers_to_readings

    @staticmethod
    def _asks_for_sensor_records(question: str) -> bool:
        words = set(re.findall(r"[a-z]+", question.lower()))
        record_terms = {"record", "records", "recorded", "log", "logs"}
        history_terms = {"history", "historical", "past", "previous"}
        sensor_terms = {"sensor", "sensors", "reading", "readings", "measurement", "measurements", "data"}
        return bool(words.intersection(record_terms)) or (
            bool(words.intersection(history_terms))
            and bool(words.intersection(sensor_terms))
        )

    @classmethod
    def _asks_for_multiple_latest_readings(cls, question: str) -> bool:
        words = set(re.findall(r"[a-z]+", question.lower()))
        recency_terms = {"latest", "newest", "recent", "last"}
        sensor_terms = {"sensor", "sensors", "reading", "readings", "measurement", "measurements"}
        selection = cls._sensor_records_selection(question)
        return (
            isinstance(selection, int)
            and selection > 1
            and bool(words.intersection(recency_terms))
            and bool(words.intersection(sensor_terms))
        )

    @classmethod
    def _asks_for_latest_reading_count(cls, question: str) -> bool:
        words = set(re.findall(r"[a-z]+", question.lower()))
        plural_reading_terms = {"sensors", "readings", "measurements", "data"}
        latest_terms = {"latest", "newest", "recent", "last"}
        return (
            bool(words.intersection(plural_reading_terms))
            and bool(words.intersection(latest_terms))
            and cls._sensor_records_selection(question) is None
        )

    @classmethod
    def _sensor_records_follow_up(
        cls,
        question: str,
        history: list[dict[str, Any]] | None,
    ) -> tuple[int | str, int] | None:
        if not isinstance(history, list):
            return None

        previous_user_message = next(
            (
                str(entry.get("content", ""))
                for entry in reversed(history)
                if isinstance(entry, dict) and entry.get("role") == "user"
            ),
            "",
        )
        previous_assistant_message = next(
            (
                str(entry.get("content", ""))
                for entry in reversed(history)
                if isinstance(entry, dict) and entry.get("role") == "assistant"
            ),
            "",
        )
        has_sensor_record_context = (
            cls._asks_for_current_readings(previous_user_message)
            or cls._asks_for_sensor_records(previous_user_message)
            or cls._asks_for_multiple_latest_readings(previous_user_message)
            or bool(re.search(
                r"\b(?:historical sensor records|showing (?:all )?\d+ sensor readings?)\b",
                previous_assistant_message,
                re.IGNORECASE,
            ))
        )
        if not has_sensor_record_context:
            return None

        words = set(re.findall(r"[a-z]+", question.lower()))
        is_next_page = "next" in words
        asks_for_latest = bool(words.intersection({"latest", "newest", "recent", "last"}))
        asks_for_table = bool(re.search(r"\btable\b", question, re.IGNORECASE))
        selection = cls._sensor_records_selection(question)
        asks_for_count = isinstance(selection, int)
        if not (is_next_page or asks_for_latest or asks_for_table or asks_for_count):
            return None

        previous_page = cls._previous_sensor_record_page(history)
        previous_count = (
            previous_page[1]
            if previous_page is not None
            else cls._previous_sensor_record_count(history)
        )
        requested_count = selection
        if is_next_page:
            limit = requested_count if isinstance(requested_count, int) else previous_count or 3
            offset = (
                previous_page[0] + previous_page[1]
                if previous_page is not None
                else previous_count or 0
            )
            return limit, offset
        if asks_for_count:
            return requested_count, 0
        if asks_for_latest:
            limit = requested_count if isinstance(requested_count, int) else previous_count or 3
            return limit, 0
        offset = previous_page[0] if previous_page is not None else 0
        return previous_count or 3, offset

    @classmethod
    def _previous_sensor_record_count(cls, history: list[dict[str, Any]]) -> int | None:
        previous_page = cls._previous_sensor_record_page(history)
        if previous_page is not None:
            return previous_page[1]

        for entry in reversed(history):
            if isinstance(entry, dict) and entry.get("role") == "user":
                selection = cls._sensor_records_selection(str(entry.get("content", "")))
                if isinstance(selection, int):
                    return selection
        return None

    @staticmethod
    def _previous_sensor_record_page(
        history: list[dict[str, Any]],
    ) -> tuple[int, int] | None:
        for entry in reversed(history):
            if not isinstance(entry, dict) or entry.get("role") != "assistant":
                continue
            content = str(entry.get("content", ""))
            match = re.search(
                r"\bshowing\s+(?:all\s+)?(\d+)\s+sensor\s+readings?"
                r"(?:\s+\(records\s+(\d+)-(\d+)\))?",
                content,
                re.IGNORECASE,
            )
            if match:
                count = int(match.group(1))
                offset = int(match.group(2)) - 1 if match.group(2) else 0
                return offset, count
            match = re.search(r"\((\d+)\s+timestamped\s+records?\b", content, re.IGNORECASE)
            if match:
                return 0, int(match.group(1))
        return None

    @classmethod
    def _is_awaiting_sensor_records_selection(
        cls,
        history: list[dict[str, Any]] | None,
    ) -> bool:
        if not isinstance(history, list):
            return False
        for entry in reversed(history):
            if isinstance(entry, dict) and entry.get("role") == "assistant":
                return entry.get("content") == cls.SENSOR_RECORDS_CONFIRMATION
        return False

    @staticmethod
    def _sensor_records_selection(question: str) -> int | str | None:
        normalized = question.lower()
        if re.search(r"\b(?:all|entire|whole|complete|full)\b", normalized):
            return "all"

        match = re.search(r"\b(\d+)\b", normalized)
        if match:
            count = int(match.group(1))
            return count if count > 0 else None
        number_words = {
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
        }
        for word, count in number_words.items():
            if re.search(rf"\b{word}\b", normalized):
                return count
        return None

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
    def _sensor_records_summary(
        records: list[dict[str, Any]],
        selection: int | str | None = None,
        offset: int = 0,
    ) -> str:
        if not records:
            return "No historical Stevia sensor records are available."

        reading_label = "reading" if len(records) == 1 else "readings"
        record_range = f"(records {offset + 1}-{offset + len(records)})"
        if selection == "all":
            return f"Showing all {len(records)} sensor {reading_label} {record_range}, newest first."
        if isinstance(selection, int) and len(records) < selection:
            requested_label = "reading" if selection == 1 else "readings"
            return (
                f"Showing {len(records)} of {selection} requested sensor {requested_label} "
                f"{record_range}, newest first."
            )
        return f"Showing {len(records)} sensor {reading_label} {record_range}, newest first."

    @staticmethod
    def _sensor_reading_assessment(sensor_data: dict[str, Any]) -> str:
        if not sensor_data:
            return (
                "I don't have current sensor readings to assess. Share the latest readings or "
                "describe the plant's symptoms, and I can help narrow down possible causes."
            )

        profiles = (
            ("temperature", "temperature", "temperature", "celsius", "C"),
            ("humidity", "relative_humidity", "humidity", "percent", "%"),
            ("soil_ph", "soil_pH", "soil pH", "pH", ""),
        )
        concerns = []
        in_range = []
        for sensor_key, parameter, label, unit, display_unit in profiles:
            value = sensor_data.get(sensor_key)
            if value is None:
                continue
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue

            target = next(
                (
                    item for item in SPROUT_KNOWLEDGE
                    if item.get("parameter", "").lower() == parameter.lower()
                ),
                None,
            )
            optimal_range = target.get("optimal_range") if target else None
            if not optimal_range:
                continue

            if numeric_value < optimal_range[0] or numeric_value > optimal_range[1]:
                direction = "below" if numeric_value < optimal_range[0] else "above"
                value_text = f"{numeric_value:g}{display_unit}" if display_unit == "%" else f"{numeric_value:g} {display_unit}".strip()
                concerns.append(
                    f"{label} {value_text} is {direction} the "
                    f"{optimal_range[0]:g}-{optimal_range[1]:g} {unit} target"
                )
            else:
                value_text = f"{numeric_value:g}{display_unit}" if display_unit == "%" else f"{numeric_value:g} {display_unit}".strip()
                in_range.append(f"{label} {value_text}")

        if concerns:
            assessment = "The readings flag conditions to check, not a diagnosis: " + "; ".join(concerns) + "."
            if any(concern.startswith("humidity ") for concern in concerns):
                assessment += " Improve airflow and inspect leaves for fungal spots."
            if any(concern.startswith("soil pH ") for concern in concerns):
                assessment += " Verify pH with a fresh soil test before adjusting."
            if in_range:
                assessment += f" Also within target: {', '.join(in_range)}."
        elif in_range:
            assessment = (
                f"Readings are within target: {', '.join(in_range)}; "
                "they don't indicate a clear problem."
            )
        else:
            assessment = (
                "The available readings can't establish whether the plant has a problem; "
                "some key values are missing or don't have calibrated diagnostic ranges."
            )

        if sensor_data.get("water_level", sensor_data.get("soil_moisture")) is not None:
            assessment += (
                " Soil-moisture percentage has no calibrated wet/dry threshold."
            )
        assessment += " Share visible symptoms for a better assessment."
        return assessment

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
