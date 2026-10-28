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
_TAGALOG_PHRASES = {
    "buksan ang": "open the",
    "buksan mo": "open",
    "buksan": "open",
    "paandarin ang": "start the",
    "paandarin mo": "start",
    "paandarin": "start",
    "sindihan": "turn on",
    "patayin ang": "turn off the",
    "patayin mo": "turn off",
    "patayin": "turn off",
    "i-on ang": "turn on the",
    "i-on mo": "turn on",
    "i-on": "turn on",
    "pa-on": "turn on",
    "paki-on": "turn on",
    "i-off ang": "turn off the",
    "i-off mo": "turn off",
    "i-off": "turn off",
    "pa-off": "turn off",
    "paki-off": "turn off",
    "isara ang": "close the",
    "isara mo": "close",
    "isara": "close",
    "ihinto ang": "stop the",
    "ihinto mo": "stop",
    "ihinto": "stop",
    "bentilador": "fan",
    "ilaw": "lights",
    "mga ilaw": "lights",
    "bomba": "pump",
    "pandilig": "sprinkler",
    "batay sa": "based on",
    "ayon sa": "based on",
    "based sa": "based on",
    "sa susunod na ilang araw": "in next few days",
    "sa mga susunod na araw": "in next few days",
    "gaano kadalas": "how often",
    "ilang beses": "how often",
    "need ko ba": "should I",
    "need ko": "need I",
    "dapat ko": "should I",
    "need ko ba i-water": "should I water",
    "need ko ba mag-water": "should I water",
    "dapat ko ba i-water": "should I water",
    "dapat ko ba mag-water": "should I water",
    "paano i-water": "how to water",
    "paano mag-water": "how to water",
    "paano i-grow": "how to grow",
    "paano mag-grow": "how to grow",
    "paano i-check": "how to check",
    "paano mag-fertilize": "how to fertilize",
    "paano mag-prune": "how to prune",
    "paano alagaan": "how to care",
    "okay ba": "is it okay",
    "safe ba": "is it safe",
    "dry na ba": "is dry",
    "basa pa ba": "is still moist",
    "yung soil": "the soil",
    "yung halaman": "the plant",
    "yung plant": "the plant",
    "my halaman": "my plant",
    "ano po ang": "what is",
    "ano ang": "what is",
    "ano yung": "what is",
    "paano po": "how",
    "paano ba": "how",
    "paano ko dapat diligan": "how should I water",
    "paano dapat diligan": "how should I water",
    "paano ko didiligan": "how should I water",
    "paano ko didiligan ito": "how should I water it",
    "paano diligan": "how to water",
    "paano magdilig": "how to water",
    "anong": "what",
    "ano": "what",
    "pinakamaganda": "best",
    "kailangan": "should",
    "pangangalaga": "care",
    "patubig": "irrigation",
    "mahalumigmig": "humid",
    "akin": "my",
    "aking": "my",
    "ko": "I",
    "mo": "you",
    "yung": "the",
    "ang": "",
    "ng": "",
    "sa": "in",
    "ba": "",
    "naman": "",
    "kasi": "because",
    "muna": "first",
    "lang": "only",
    "pa": "still",
    "na": "now",
    "ito": "it",
    "iyan": "that",
    "sakin": "for me",
    "nyo": "you",
    "siya": "it",
    "sila": "they",
    "kami": "we",
    "tayo": "we",
    "temp": "temperature",
    "i-water": "water",
    "mag-water": "water",
    "i-grow": "grow",
    "mag-grow": "grow",
    "mag-fertilize": "fertilize",
    "mag-prune": "prune",
    "mag-check": "check",
    "sikat ng araw": "sunlight",
    "lupa": "soil",
    "halaman": "plant",
    "diligan": "water",
    "didiligan": "water",
    "magdilig": "water",
    "pagdidilig": "watering",
    "tubig": "water",
    "tuyo": "dry",
    "basa": "moist",
    "liwanag": "light",
    "halumigmig": "humidity",
    "maalinsangan": "humid",
    "temperatura": "temperature",
    "init": "heat",
    "malamig": "cold",
    "araw-araw": "daily",
    "araw": "day",
    "rekomendasyon": "recommendation",
    "mairerekomenda": "recommend",
    "inirerekomenda": "recommend",
    "inirerekomenda mo": "recommend",
    "recommended": "recommend",
    "recommend mo": "recommend",
    "iminumungkahi": "suggest",
    "pinakabago": "latest",
    "pinakabagong": "latest",
    "ipakita": "show",
    "ipadala": "send",
    "sukat": "reading",
    "mga": "",
    "lahat": "all",
    "problema": "problem",
    "masama": "wrong",
}
_TAGALOG_MARKERS = {
    "buksan", "paandarin", "patayin", "sindihan", "i-on", "i-off", "pa-on", "pa-off",
    "paki-on", "paki-off", "isara", "ihinto", "bentilador", "ilaw", "bomba",
    "pandilig", "huwag", "wag",
    "batay", "ayon", "paano", "dapat", "diligan", "didiligan", "magdilig",
    "pagdidilig", "halaman", "lupa", "tubig", "tuyo", "basa", "liwanag",
    "halumigmig", "maalinsangan", "gaano", "rekomendasyon", "mairerekomenda",
    "inirerekomenda", "iminumungkahi", "pinakabago", "pinakabagong", "ipakita",
    "ipadala", "araw-araw", "lahat", "problema",
    "dapat", "oo", "opo", "tama", "hindi", "ano", "anong", "saan",
    "bakit", "kailan", "alin", "kailangan", "pinakamaganda", "pangangalaga",
    "patubig", "pwede", "maaari",
    "ko", "mo", "yung", "ba", "naman", "kasi", "muna", "lang",
    "sakin", "akin", "natin", "namin", "nyo", "po", "tayo", "rin",
    "din", "talaga", "kaya",
    "siya", "sila", "kami", "i-water", "mag-water",
    "i-grow", "mag-grow", "mag-fertilize", "mag-prune",
    "gusto", "sobrang", "sobra", "kapag", "kung", "para", "dahil",
    "habang", "pero", "sana", "paki", "pakisabi", "nakaka", "ilang",
    "bang", "ng", "nga", "nang", "kahit", "mga", "nito", "nitong",
    "niya", "nila", "niyo", "saan", "alin", "kanino", "dito", "doon",
}


class SteviaChatbot:
    """Answers Stevia questions with a local intent classifier and deterministic fallback."""

    MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "stevia-intent-bert"
    DEVICE_ALIASES = {
        "light": "light",
        "lights": "light",
        "lamp": "light",
        "lamps": "light",
        "ilaw": "light",
        "fan": "fan",
        "bentilador": "fan",
        "pump": "pump",
        "bomba": "pump",
        "sprinkler": "pump",
        "pandilig": "pump",
    }
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
            classifier_path = self.MODEL_PATH / "intent_classifier.joblib"
            if classifier_path.exists():
                import joblib

                self.model = joblib.load(classifier_path)
            else:
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
        if not self.model:
            return {}

        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba([question])[0]
            labels = self.model.classes_
            temperature = max(float(getattr(self.model, "confidence_temperature", 1.0)), 1e-6)
            scaled = [
                max(float(probability), 1e-12) ** (1 / temperature)
                for probability in probabilities
            ]
            total = sum(scaled)
            return {
                str(label): probability / total
                for label, probability in zip(labels, scaled)
            }

        if not self.tokenizer:
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
            model_config = getattr(self.model, "config", None)
            confidence_threshold = float(
                getattr(
                    self.model,
                    "intent_confidence_threshold",
                    getattr(model_config, "intent_confidence_threshold", 1.0),
                )
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
                return model_intent, "intent classifier"

        # A specific care word such as "water" should beat the generic
        # "plant" match in questions like "Should I water the plant?".
        specific_keyword_score = max(
            (score for intent, score in keyword_scores.items() if intent != "general"),
            default=0.0,
        )
        if specific_keyword_score > 0:
            keyword_scores["general"] = 0.0

        # Keyword evidence keeps answers stable while the classifier supplies context.
        combined = {
            intent: keyword_scores[intent] + model_scores.get(intent, 0.0) * 0.35
            for intent in KNOWLEDGE_BASE
        }
        intent = max(combined, key=combined.get)
        if combined[intent] == 0:
            return "general", "fallback"
        source = "intent classifier + knowledge-base" if model_scores else "knowledge-base fallback"
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

    @classmethod
    def _device_command(cls, question: str) -> dict[str, str] | None:
        question_lower = question.casefold()
        if re.search(r"\b(?:do not|don't|never|huwag|wag)\b", question_lower):
            return None
        device_match = re.search(
            r"\b(light|lights|lamp|lamps|ilaw|fan|bentilador|pump|bomba|sprinkler|pandilig)\b",
            question_lower,
        )
        if not device_match or re.search(
            r"\bshould\s+(?:i|we)\b|\b(?:dapat|kailangan)\s+(?:ko|natin)\b",
            question_lower,
        ):
            return None

        device = cls.DEVICE_ALIASES[device_match.group(1)]
        if re.search(
            r"\b(?:turn\s+(?:the\s+)?off|switch\s+(?:the\s+)?off|"
            r"close|stop|deactivate)\b|\b(?:fan|lights?|pump|sprinkler)\s+off\b",
            question_lower,
        ):
            action = "OFF"
        elif re.search(
            r"\b(?:open|turn\s+(?:the\s+)?on|switch\s+(?:the\s+)?on|"
            r"start|activate)\b|\b(?:fan|lights?|pump|sprinkler)\s+on\b",
            question_lower,
        ):
            action = "ON"
        else:
            return None

        return {"device": device, "action": action}

    @classmethod
    def _pending_device_confirmation(
        cls,
        history: list[dict[str, Any]] | None,
    ) -> dict[str, str] | None:
        if not isinstance(history, list):
            return None

        assistant_index = next(
            (
                index
                for index in range(len(history) - 1, -1, -1)
                if isinstance(history[index], dict)
                and history[index].get("role") == "assistant"
            ),
            None,
        )
        if assistant_index is None:
            return None

        previous_user = next(
            (
                str(entry.get("content", ""))
                for entry in reversed(history[:assistant_index])
                if isinstance(entry, dict) and entry.get("role") == "user"
            ),
            None,
        )
        if not previous_user:
            return None

        assistant_text = str(history[assistant_index].get("content", "")).strip()
        confirmation_patterns = (
            (
                r"Would you like me to turn (on|off) the "
                r"(lights?|fan|sprinkler|pump)\?",
                {"on": "ON", "off": "OFF"},
            ),
            (
                r"Gusto mo bang (buksan|patayin) ang "
                r"(ilaw|bentilador|sprinkler|bomba)\?",
                {"buksan": "ON", "patayin": "OFF"},
            ),
            (
                r"I-(on|off) ang (ilaw|bentilador|sprinkler|bomba)\?",
                {"on": "ON", "off": "OFF"},
            ),
        )
        prompt_command = None
        prompted_label = ""
        for pattern, action_map in confirmation_patterns:
            match = re.fullmatch(pattern, assistant_text, re.IGNORECASE)
            if match:
                prompted_action = action_map[match.group(1).casefold()]
                prompted_device = cls.DEVICE_ALIASES[match.group(2).casefold()]
                prompted_label = match.group(2).casefold()
                prompt_command = {
                    "device": prompted_device,
                    "action": prompted_action,
                }
                break

        if prompt_command is None:
            return None

        normalized_request = cls._normalize_tagalog_question(previous_user)
        if cls._device_command(normalized_request) != prompt_command:
            return None
        return {**prompt_command, "label": prompted_label}

    def answer(
        self,
        question: str,
        sensor_data: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
        pending_device_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        manual_sensor_data = sensor_data if isinstance(sensor_data, dict) else {}
        combined_sensor_data = {**LATEST_READING, **manual_sensor_data}
        history = self._history_before_current(question, history)
        tagalog_response = self._contains_tagalog(question)
        taglish_response = self._contains_taglish(question)
        pending_typo = self._pending_typo_confirmation(history)
        if pending_typo:
            original_question, corrected_question = pending_typo
            reply = question.strip().casefold()
            if reply in {"yes", "y", "correct", "that's right", "that is right", "oo", "opo", "tama"}:
                question = corrected_question
            elif reply in {"no", "n", "nope", "not that", "hindi", "hindi po"}:
                return self._response(
                    (
                        "Sige. Pakisabi muli ang tanong gamit ang mga salitang nais mong gamitin."
                        if tagalog_response
                        else "Okay. Please rephrase your request with the intended words."
                    ),
                    "Typo clarification",
                    "typo_clarification",
                    "typo confirmation",
                    combined_sensor_data,
                )
            else:
                question = self._correct_typographical_errors(question)
        tagalog_response = tagalog_response or self._contains_tagalog(question)
        taglish_response = taglish_response or self._contains_taglish(question)
        question = self._normalize_tagalog_question(question)

        def respond(
            answer: str,
            topic: str,
            intent: str,
            source: str,
            sensor_data: dict[str, Any],
            sensor_records: list[dict[str, Any]] | None = None,
        ) -> dict[str, Any]:
            if tagalog_response:
                answer = self._tagalog_answer(
                    answer,
                    intent,
                    sensor_data,
                    sensor_records,
                )
                if taglish_response:
                    answer = self._taglish_style(answer, intent)
            return self._response(
                answer,
                topic,
                intent,
                source,
                sensor_data,
                sensor_records=sensor_records,
            )

        pending_device_command = self._pending_device_confirmation(history)
        if pending_device_command is None and isinstance(pending_device_context, dict):
            context_device = pending_device_context.get("device")
            context_action = pending_device_context.get("action")
            if (
                isinstance(context_device, str)
                and context_device in {"light", "fan", "pump"}
                and isinstance(context_action, str)
                and context_action in {"ON", "OFF"}
            ):
                pending_device_command = {
                    "device": context_device,
                    "action": context_action,
                    "label": {
                        "light": "lights",
                        "fan": "fan",
                        "pump": "pump",
                    }[context_device],
                }
        confirmation_reply = question.strip().casefold()
        if pending_device_command and confirmation_reply in {
            "yes", "y", "yeah", "yep", "sure", "go ahead", "confirm",
            "oo", "opo", "sige", "tama",
        }:
            device_name = pending_device_command["label"]
            action_label = "on" if pending_device_command["action"] == "ON" else "off"
            result = respond(
                f"Confirmed. Sending the command to turn the {device_name} {action_label}.",
                "Device control",
                "device_control",
                "device confirmation",
                combined_sensor_data,
            )
            result["device_command"] = {
                "device": pending_device_command["device"],
                "action": pending_device_command["action"],
            }
            result["execute_device_command"] = True
            return result
        if pending_device_command and confirmation_reply in {
            "no", "n", "nope", "cancel", "cancel it", "hindi", "hindi po", "ayaw",
        }:
            return respond(
                "Sige, nakansela. Walang device command na ipinadala."
                if tagalog_response
                else "Cancelled. No device command was sent.",
                "Device control",
                "device_control",
                "device confirmation",
                combined_sensor_data,
            )

        device_command = self._device_command(question)
        if device_command:
            device_label = (
                "lights" if device_command["device"] == "light"
                else "sprinkler" if "sprinkler" in question.casefold()
                else device_command["device"]
            )
            action_label = "turn on" if device_command["action"] == "ON" else "turn off"
            if tagalog_response:
                device_name = {
                    "light": "ilaw",
                    "fan": "bentilador",
                    "pump": "sprinkler" if "sprinkler" in question.casefold() else "bomba",
                }[device_command["device"]]
                if taglish_response:
                    prompt = (
                        f"I-on ang {device_name}?"
                        if device_command["action"] == "ON"
                        else f"I-off ang {device_name}?"
                    )
                else:
                    prompt = (
                        f"Gusto mo bang buksan ang {device_name}?"
                        if device_command["action"] == "ON"
                        else f"Gusto mo bang patayin ang {device_name}?"
                    )
            else:
                prompt = f"Would you like me to {action_label} the {device_label}?"
            result = respond(
                prompt,
                "Device control",
                "device_control",
                "device confirmation",
                combined_sensor_data,
            )
            result["device_command"] = device_command
            result["confirmation"] = {
                "required": True,
                "options": [
                    {"label": "Yes", "value": "yes"},
                    {"label": "No", "value": "no"},
                ],
            }
            return result

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
            return respond(
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
            return respond(
                social_reply, "About Stevia", "general", "social", combined_sensor_data
            )

        if self._is_awaiting_sensor_records_selection(history):
            selection = self._sensor_records_selection(question)
            if selection is None and not self._is_new_topic_question(question):
                return respond(
                    self.SENSOR_RECORDS_CONFIRMATION,
                    "Historical sensor records",
                    "sensor_records",
                    "sensor-record confirmation",
                    combined_sensor_data,
                )

            if selection is not None:
                records = load_sensor_records()
                selected_records = records if selection == "all" else records[:selection]
                return respond(
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
            return respond(
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
            return respond(
                self._sensor_records_summary(selected_records, limit, offset),
                "Historical sensor records",
                "sensor_records",
                "sensor-data",
                combined_sensor_data,
                sensor_records=selected_records,
            )

        if not self._is_in_scope(question, history):
            return respond(
                self.OUT_OF_SCOPE_RESPONSE,
                "Outside scope",
                "out_of_scope",
                "scope validation",
                combined_sensor_data,
            )

        if self._asks_for_sensor_records(question):
            return respond(
                self.SENSOR_RECORDS_CONFIRMATION,
                "Historical sensor records",
                "sensor_records",
                "sensor-record confirmation",
                combined_sensor_data,
            )

        if self._asks_for_latest_reading_count(question):
            return respond(
                self.SENSOR_RECORDS_CONFIRMATION,
                "Historical sensor records",
                "sensor_records",
                "sensor-record confirmation",
                combined_sensor_data,
            )

        if self._asks_for_sensor_assessment(question):
            return respond(
                self._sensor_reading_assessment(combined_sensor_data),
                "Sensor-based plant check",
                "problems",
                "rule-based sensor assessment",
                combined_sensor_data,
            )

        if self._asks_for_current_readings(question):
            return respond(
                self._current_reading_summary(combined_sensor_data, history),
                "Current sensor readings",
                "sensor",
                "sensor-data",
                combined_sensor_data,
            )

        intent, source = self._select_intent(question, history=history)
        sensor_note = self._sensor_note(combined_sensor_data, intent)
        if sensor_note:
            return respond(
                sensor_note,
                KNOWLEDGE_BASE[intent]["title"],
                intent,
                "rule-based SPROUT sensor comparison",
                combined_sensor_data,
            )

        llm_response = (
            None
            if tagalog_response
            else self._llm_answer(question, history, combined_sensor_data)
        )
        if llm_response:
            return respond(
                llm_response,
                KNOWLEDGE_BASE[intent]["title"],
                intent,
                "llm (OpenAI)",
                combined_sensor_data,
            )

        response = self._choose_response(intent, question, history)

        return respond(
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
    def _contains_tagalog(question: str) -> bool:
        words = set(re.findall(r"[a-zA-Z-]+", question.casefold()))
        return bool(words.intersection(_TAGALOG_MARKERS))

    @staticmethod
    def _contains_taglish(question: str) -> bool:
        if not SteviaChatbot._contains_tagalog(question):
            return False
        words = set(re.findall(r"[a-zA-Z-]+", question.casefold()))
        english_care_terms = {
            "based", "sensor", "sensors", "reading", "readings", "soil", "plant",
            "water", "watering", "temperature", "temp", "light", "sunlight",
            "humidity", "humid", "fertilizer", "fertilize", "grow", "growing",
            "check", "recommend", "recommendation", "recommendations", "dry",
            "moist", "moisture", "problem", "problems", "latest", "show",
            "need", "best", "safe", "okay", "healthy", "pH", "ph",
            "sweetener", "sweet", "sugar", "herb", "vegetable", "perennial",
            "foliage", "indoors", "indoor", "pot", "container", "garden",
            "cuttings", "stem", "warm", "cold", "frost", "heat", "degree",
            "acidic", "neutral", "compost", "drainage", "sun", "shade",
            "bright", "window", "air", "mist", "feed", "nutrient", "yellow",
            "wilt", "fungus", "pest", "insect", "disease", "prune",
            "fan", "fans", "light", "lights", "pump", "sprinkler", "open",
            "close", "on", "off", "turn", "start", "stop", "activate",
            "history", "historical", "records", "record", "log", "measurements",
            "data", "old", "previous", "all", "table", "today", "yesterday",
            "leaves", "leaf", "roots", "root", "stem", "degrees", "hours",
            "soil", "waterlogged", "overwatering", "underwatering", "misting",
            "airflow", "ventilation", "soggy", "sunny", "shady", "frost",
            "greenhouse", "potting", "mix", "container", "garden", "watering",
            "season", "week", "daily", "morning", "evening", "temperature",
            "planting", "harvest", "history", "sensor", "measurement",
            "i-water", "mag-water", "i-grow", "mag-grow", "mag-fertilize",
            "mag-prune", "mag-check", "i-check", "i-grow",
        }
        return bool(words.intersection(english_care_terms))

    @staticmethod
    def _normalize_tagalog_question(question: str) -> str:
        if not SteviaChatbot._contains_tagalog(question):
            return question
        normalized = question.casefold()
        for phrase, replacement in sorted(
            _TAGALOG_PHRASES.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        ):
            normalized = re.sub(
                rf"(?<!\w){re.escape(phrase)}(?!\w)",
                replacement,
                normalized,
            )
        return re.sub(r"\s+", " ", normalized).strip()

    @staticmethod
    def _taglish_style(answer: str, intent: str) -> str:
        if intent == "device_control":
            return answer
        starters = {
            "sensor": "Quick sensor check: ",
            "sensor_records": "Quick records check: ",
            "watering": "Quick plant check: ",
            "temperature": "Quick temp check: ",
            "humidity": "Quick humidity check: ",
            "sunlight": "Quick light check: ",
            "soil": "Quick soil check: ",
            "problems": "Quick plant check: ",
        }
        styled_answer = starters.get(intent, "Quick plant-care check: ") + answer
        if intent == "watering":
            styled_answer += (
                " Low-key important: huwag mag-water based sa sensor percentage lang; "
                "soil check muna ang priority."
            )
        return styled_answer

    @staticmethod
    def _tagalog_answer(
        english_answer: str,
        intent: str,
        sensor_data: dict[str, Any],
        sensor_records: list[dict[str, Any]] | None = None,
    ) -> str:
        if intent == "device_control":
            prompt_match = re.fullmatch(
                r"Would you like me to turn (on|off) the "
                r"(lights?|fan|sprinkler|pump)\?",
                english_answer,
                re.IGNORECASE,
            )
            if prompt_match:
                action = "buksan" if prompt_match.group(1).casefold() == "on" else "patayin"
                device = prompt_match.group(2).casefold()
                if device in {"light", "lights"}:
                    device = "ilaw"
                return f"Gusto mo bang {action} ang {device}?"
            if english_answer.startswith("Cancelled."):
                return "Sige, nakansela. Walang device command na ipinadala."
            return english_answer
        if intent == "out_of_scope":
            return "Pasensya na, tungkol lamang sa pag-aalaga ng Stevia ang kaya kong sagutin."
        if intent == "typo_clarification":
            return "Paumanhin, hindi ko naunawaan ang isang salita. Pakisubukang sabihin muli sa ibang paraan."
        if intent == "sensor_records":
            if english_answer == SteviaChatbot.SENSOR_RECORDS_CONFIRMATION:
                return 'Ilang pinakabagong sensor reading ang gusto mong makita? Sumagot ng numero (halimbawa, 3) o "lahat".'
            if not sensor_records:
                return "Walang makukuhang lumang tala ng sensor sa ngayon."
            return f"Ipinapakita ang {len(sensor_records)} sensor reading, mula sa pinakabago."
        if intent == "sensor":
            if not sensor_data:
                return "Wala pang makukuhang kasalukuyang sensor reading."
            recorded_at = sensor_data.get("recorded_at", "hindi alam ang oras")
            temperature = sensor_data.get("temperature", "hindi alam")
            humidity = sensor_data.get("humidity", sensor_data.get("relative_humidity", "hindi alam"))
            moisture = sensor_data.get("water_level", sensor_data.get("soil_moisture", "hindi alam"))
            return (
                f"Pinakabagong sukat ({recorded_at}): temperatura {temperature} C, "
                f"halumigmig {humidity}%, at halumigmig ng lupa {moisture}%."
            )
        if intent == "watering":
            moisture = sensor_data.get("water_level", sensor_data.get("soil_moisture"))
            if moisture is None:
                return (
                    "Panatilihing bahagyang mamasa-masa ang lupa ngunit huwag itong hayaang lubog sa tubig. "
                    "Kapain ang unang 2-3 cm ng lupa araw-araw at diligan lamang kapag tuyo na ang bahaging iyon. "
                    "Hayaang umagos ang sobrang tubig."
                )
            unit_map = sensor_data.get("sensor_units", {})
            unit = unit_map.get("water_level", unit_map.get("soil_moisture", "%")) if isinstance(unit_map, dict) else "%"
            if str(unit).casefold() in {"percent", "percentage"}:
                unit = "%"
            elif unit and not str(unit).startswith("%"):
                unit = f" {unit}"
            trend = sensor_data.get("sensor_trends", {})
            direction = trend.get("water_level", trend.get("soil_moisture")) if isinstance(trend, dict) else None
            trend_text = {
                "rising": " Tumataas ang mga kamakailang sukat ng halumigmig ng lupa.",
                "falling": " Bumaba ang mga kamakailang sukat ng halumigmig ng lupa.",
                "stable": " Hindi nagbabago ang mga kamakailang sukat ng halumigmig ng lupa.",
            }.get(direction, "")
            answer = (
                f"Ayon sa sensor, {moisture}{unit} ang sukat ng halumigmig ng lupa.{trend_text} "
                "Walang nasuring tiyak na hangganan ang porsiyentong ito para malaman kung tuyo o basa "
                "ang lupa, kaya huwag itong gamiting nag-iisang batayan sa pagdidilig. Sa mga susunod na "
                "araw, kapain ang unang 2-3 cm ng lupa araw-araw. Diligan lamang kapag tuyo na ang ibabaw; "
                "kung mamasa-masa pa, maghintay. Kung lubog o puspos ng tubig ang lupa, itigil muna ang "
                "pagdidilig at ayusin ang daluyan ng tubig."
            )
            humidity = sensor_data.get("humidity", sensor_data.get("relative_humidity"))
            try:
                high_humidity = humidity is not None and float(humidity) > 80
            except (TypeError, ValueError):
                high_humidity = False
            if high_humidity:
                answer += (
                    f" Mataas ang halumigmig ng hangin ({humidity}%); panatilihing maayos ang bentilasyon "
                    "at iwasang wisikan o basain ang mga dahon. Hindi nito ipinapakita kung kailangan nang diligan ang lupa."
                )
            return answer

        if intent == "problems":
            concerns = []
            for key, label, low, high, unit in (
                ("temperature", "temperatura", 18, 28, "C"),
                ("humidity", "halumigmig ng hangin", 65, 80, "%"),
                ("soil_ph", "pH ng lupa", 6.5, 7.2, ""),
            ):
                value = sensor_data.get(key)
                if value is None and key == "humidity":
                    value = sensor_data.get("relative_humidity")
                try:
                    numeric_value = float(value) if value is not None else None
                except (TypeError, ValueError):
                    numeric_value = None
                if numeric_value is not None and (numeric_value < low or numeric_value > high):
                    position = "mas mababa" if numeric_value < low else "mas mataas"
                    concerns.append(
                        f"{label} na {numeric_value:g}{unit} ay {position} sa target na {low:g}-{high:g}{unit}"
                    )
            if concerns:
                return (
                    "May mga sukat na dapat bantayan, ngunit hindi ito tiyak na diagnosis: "
                    + "; ".join(concerns)
                    + ". Suriin din ang kondisyon ng lupa at mga nakikitang sintomas."
                )
            if sensor_data.get("water_level", sensor_data.get("soil_moisture")) is not None:
                return (
                    "Walang nakitang malinaw na problema sa mga sukat na may target range. "
                    "Walang kalibradong hangganan ang porsiyento ng soil moisture para matukoy kung tuyo o basa ang lupa; "
                    "kapain ang ibabaw at tingnan ang mga sintomas ng halaman."
                )
            return (
                "Hindi sapat ang iisang sensor reading para makapagbigay ng tiyak na diagnosis. "
                "Suriin ang lupa, drainage, at mga dahon; ibahagi ang nakikitang sintomas upang mas matukoy ang posibleng sanhi."
            )

        responses = {
            "general": "Ang Stevia ay halamang-gamot na itinatanim para sa matatamis nitong dahon. Kailangan nito ng init, maliwanag na sikat ng araw, at lupang mamasa-masa ngunit mahusay ang daluyan ng tubig.",
            "growing": "Para lumago nang maayos ang Stevia, bigyan ito ng init, sapat na liwanag, maayos na daloy ng hangin, at lupang mahusay ang drainage. Iwasang manatiling nakababad sa tubig ang mga ugat.",
            "temperature": "Pinakamainam sa Stevia ang temperaturang humigit-kumulang 18-28 C. Protektahan ito sa hamog na nagyelo at magbigay ng bahagyang lilim at bentilasyon kapag napakainit.",
            "soil": "Gumamit ng mataba at maluwag na lupa na mahusay ang drainage. Target ng SPROUT ang pH na 6.5-7.2; iwasang manatiling lubog sa tubig ang mga ugat.",
            "sunlight": "Bigyan ang Stevia ng 6-8 oras na direktang sikat ng araw bawat araw. Sa matinding init, maaaring magbigay ng bahagyang lilim sa hapon.",
            "humidity": "Target ng SPROUT ang 65-80% na halumigmig. Kapag mas mataas, pagandahin ang bentilasyon, bawasan ang pagwisik, at tingnan kung may batik-singaw ang mga dahon.",
            "fertilization": "Gumamit ng kaunting balanseng pataba ayon sa tagubilin sa label. Iwasan ang labis na pataba at panatilihing maayos ang drainage ng lupa.",
            "problems": "Hindi sapat ang iisang sensor reading para makapagbigay ng tiyak na diagnosis. Suriin ang lupa, drainage, at mga dahon; ibahagi ang nakikitang sintomas upang mas matukoy ang posibleng sanhi.",
        }
        return responses.get(
            intent,
            "Naiintindihan ko ang tanong mo. Maaari kitang tulungan sa pag-aalaga, pagdidilig, lupa, liwanag, temperatura, at halumigmig ng Stevia.",
        )

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
        if SteviaChatbot._asks_for_care_advice(words):
            return False

        reading_terms = {
            "reading", "readings", "measurement", "measurements", "snapshot", "value", "values",
        }
        if words.intersection(reading_terms):
            return True

        sensor_terms = {"sensor", "sensors", "data"}
        request_terms = {
            "show", "display", "send", "give", "provide", "fetch", "retrieve", "get",
            "latest", "newest", "recent", "current", "now",
        }
        return bool(words.intersection(sensor_terms) and words.intersection(request_terms))

    @staticmethod
    def _asks_for_care_advice(words: set[str]) -> bool:
        advice_terms = {
            "recommend", "recommendation", "recommendations", "advise", "advice",
            "suggest", "suggestion", "suggestions",
        }
        care_actions = {"water", "watering", "moisture", "irrigate", "irrigation"}
        return bool(
            words.intersection(advice_terms)
            or (
                words.intersection(care_actions)
                and words.intersection({"how", "should"})
            )
        )

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
        if cls._asks_for_care_advice(words):
            return None

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
            note = (
                f"SPROUT's Stevia moisture target is {target}. The current soil-moisture reading is "
                f"{reading}{unit_text}.{trend} SPROUT provides no numeric dry/moist thresholds or "
                "sensor calibration for this percentage, so it cannot reliably determine whether to irrigate. "
                "For the next few days, check the top inch once daily; do not water just because this "
                "percentage changes; water only when that layer is dry. As a rough warm-weather guide, "
                "this may mean about 2-3 waterings per week, but the soil check takes priority. Wait if it is still "
                "moist, and stop watering and improve drainage if the soil is saturated."
            )
            humidity = sensor_data.get("humidity", sensor_data.get("relative_humidity"))
            humidity_target = next(
                (item for item in SPROUT_KNOWLEDGE
                 if item.get("parameter", "").lower() == "relative_humidity"),
                None,
            )
            try:
                numeric_humidity = float(humidity) if humidity is not None else None
            except (TypeError, ValueError):
                numeric_humidity = None
            humidity_range = humidity_target.get("optimal_range") if humidity_target else None
            if (
                numeric_humidity is not None
                and humidity_range
                and numeric_humidity > humidity_range[1]
            ):
                note += (
                    f" Air humidity is {numeric_humidity:g}%, above the "
                    f"{humidity_range[0]:g}-{humidity_range[1]:g}% target; maintain airflow "
                    "and avoid misting or wetting the foliage. High air humidity does not show "
                    "whether the soil itself needs water."
                )
            return note

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
