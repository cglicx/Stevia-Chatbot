import re
import unicodedata


_PHRASE_TRANSLATIONS = (
    (re.compile(r"\bsikat\s+ng\s+araw\b"), "sunlight"),
    (re.compile(r"\b(?:paki-)?i-on\b|\bpaki-on\b"), "turn on"),
    (re.compile(r"\b(?:paki-)?i-off\b|\bpaki-off\b"), "turn off"),
)
_TOKEN_TRANSLATIONS = {
    "ano": "what",
    "anong": "what",
    "ang": "",
    "araw": "sun",
    "ba": "",
    "bakit": "why",
    "dahon": "leaf",
    "dapat": "should",
    "diligan": "water",
    "didiligan": "water",
    "diligin": "water",
    "dilaw": "yellow",
    "gaano": "how",
    "halumigmig": "humidity",
    "halaman": "plant",
    "init": "heat",
    "buksan": "open",
    "paandarin": "start",
    "patayin": "turn off",
    "sindihan": "turn on",
    "isara": "close",
    "ihinto": "stop",
    "bentilador": "fan",
    "ilaw": "lights",
    "bomba": "pump",
    "pandilig": "sprinkler",
    "itanim": "grow",
    "kailan": "when",
    "kailangan": "need",
    "ko": "my",
    "kumukulot": "curl",
    "lupa": "soil",
    "mainit": "hot",
    "manilaw": "yellow",
    "magtanim": "grow",
    "magdilig": "water",
    "mga": "",
    "naninilaw": "yellow",
    "nalalanta": "wilt",
    "nagtatanim": "grow",
    "ng": "",
    "pagtatanim": "grow",
    "pagdidilig": "watering",
    "paninilaw": "yellow",
    "paano": "how",
    "pataba": "fertilizer",
    "peste": "pest",
    "problema": "problem",
    "pwede": "can",
    "sakit": "disease",
    "sa": "in",
    "tanim": "plant",
    "tuyo": "dry",
    "yung": "the",
}


def normalize_intent_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    for pattern, replacement in _PHRASE_TRANSLATIONS:
        normalized = pattern.sub(replacement, normalized)

    tokens = re.findall(r"[\w']+", normalized)
    return " ".join(
        translated
        for token in tokens
        if (translated := _TOKEN_TRANSLATIONS.get(token, token))
    )
