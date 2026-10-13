import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
JSON_DATASET = PROJECT_ROOT / "sprout_knowledge_base.json"
CSV_DATASET = PROJECT_ROOT / "sprout_knowledge_base.csv"


def _parameter_key(value: str) -> str:
    aliases = {
        "soil ph": "soil_pH",
        "temperature": "temperature",
        "relative humidity": "relative_humidity",
        "light intensity (daily direct sun)": "light_intensity",
        "soil moisture": "soil_moisture",
    }
    return aliases.get(value.strip().lower(), value.strip().lower().replace(" ", "_"))


def _load_sprout_knowledge() -> list[dict]:
    """Load the uploaded SPROUT data, with CSV as a fallback format."""
    if JSON_DATASET.exists():
        try:
            payload = json.loads(JSON_DATASET.read_text(encoding="utf-8"))
            return payload.get("parameters", [])
        except (OSError, json.JSONDecodeError, AttributeError):
            pass

    if CSV_DATASET.exists():
        try:
            with CSV_DATASET.open(newline="", encoding="utf-8-sig") as file:
                rows = list(csv.DictReader(file))
            grouped = {}
            for row in rows:
                parameter = _parameter_key(row["parameter"])
                grouped.setdefault(parameter, {
                    "parameter": parameter,
                    "unit": row["unit"],
                    "optimal_range": None,
                    "classifications": [],
                    "sources": [],
                })
                minimum = row["condition_min"]
                maximum = row["condition_max"]
                condition_range = None if minimum == "NA" else [float(minimum), float(maximum)]
                grouped[parameter]["classifications"].append({
                    "label": row["classification"],
                    "range": condition_range,
                    "interpretation": row["interpretation"],
                    "recommendation": row["recommended_action"],
                })
                grouped[parameter]["sources"].append(row["source"])
            return list(grouped.values())
        except (OSError, KeyError, ValueError):
            pass

    return []


SPROUT_KNOWLEDGE = _load_sprout_knowledge()
KNOWLEDGE_SOURCE = JSON_DATASET.name if JSON_DATASET.exists() else CSV_DATASET.name if CSV_DATASET.exists() else "built-in fallback"


KNOWLEDGE_BASE = {
    "general": {
        "title": "About Stevia",
        "keywords": ["stevia", "plant", "sweet", "sweetener", "rebaudiana", "leaf"],
        "response": "Stevia (Stevia rebaudiana) is a warm-season perennial herb grown for its naturally sweet leaves. It is native to South America and is commonly grown in containers or garden beds with well-drained soil.",
        "responses": [
            "Stevia (Stevia rebaudiana) is a warm-season herb grown for its naturally sweet leaves. It prefers bright light, warm temperatures, and soil that drains freely.",
            "Quick overview: Stevia is a South American perennial commonly grown for its sweet foliage. In cold climates, treat it as an annual or move the container indoors before frost.",
            "Think of Stevia as a sun-loving, frost-sensitive herb: give it warmth, steady moisture without soggy roots, and regular harvesting to encourage leafy growth.",
        ],
    },
    "growing": {
        "title": "Growing conditions",
        "keywords": ["grow", "growing", "care", "container", "garden"],
        "response": "Give Stevia warm conditions, good airflow, bright light, and soil that drains freely. Space plants so leaves dry quickly after watering. In colder climates, grow it as an annual or move containers indoors before frost.",
        "responses": [
            "For reliable growth, provide warmth, bright light, good airflow, and loose soil. Leave space around the plant so foliage can dry after watering.",
            "A simple Stevia setup has three priorities: plenty of light, roots that can drain, and protection from frost. Containers are useful because they can be moved indoors when weather turns cold.",
            "Care checklist: choose a sunny position, use well-drained fertile soil, avoid crowded foliage, and harvest the tips regularly to keep the plant bushy.",
        ],
    },
    "temperature": {
        "title": "Temperature",
        "keywords": ["temperature", "warm", "cold", "frost", "heat", "degree"],
        "response": "Stevia prefers roughly 18-30 C (65-86 F). Protect it from frost and prolonged temperatures below about 10 C (50 F). During extreme heat, provide afternoon shade and check soil moisture more often.",
        "responses": [
            "Keep Stevia around 18-30 C (65-86 F). Shield it from frost and temperatures below about 10 C (50 F); during heat waves, add afternoon shade and inspect the soil more often.",
            "Temperature guide: 18-28 C is the uploaded SPROUT optimal range. Cool conditions slow growth, while heat above 28 C increases stress and water demand.",
            "Watch the extremes rather than chasing an exact number. Frost can kill Stevia, and strong heat can scorch leaves, so move containers or improve shade and airflow when needed.",
        ],
    },
    "watering": {
        "title": "Watering",
        "keywords": ["water", "watering", "moisture", "dry", "drought", "irrigate"],
        "response": "Keep the soil evenly and lightly moist, but never waterlogged. Water when the top 2-3 cm (about 1 inch) feels dry, then let excess drain away. Containers may need more frequent checks in hot weather.",
        "responses": [
            "For a small indoor Stevia plant, check the soil every 2-3 days. This often works out to about 2-3 waterings per week, but water only when the top 2-3 cm (about 1 inch) feels dry.",
            "There is no fixed weekly schedule: Stevia may need water around 2-3 times per week in warm weather and less often in cool conditions. Feel the top inch of soil before watering.",
            "Water slowly until a little drains from the pot, then empty the saucer. The next watering is due when the top inch of soil has dried, which is usually more reliable than counting days.",
        ],
    },
    "soil": {
        "title": "Soil",
        "keywords": ["soil", "potting", "drain", "compost", "pH", "ground"],
        "response": "Use fertile, loose, well-drained soil enriched with compost. A slightly acidic to neutral pH, approximately 6.0-7.5, works well. Improve heavy clay with organic matter and drainage material rather than letting roots sit wet.",
        "responses": [
            "Use fertile, loose soil with plenty of drainage and some compost. The uploaded SPROUT data gives a target pH of 6.5-7.2 for nutrient uptake and root health.",
            "Soil recipe: combine a quality potting mix with organic matter, make sure excess water can escape, and test pH occasionally. Avoid leaving Stevia roots in saturated soil.",
            "If leaves yellow while the soil stays wet, drainage is the first thing to inspect. Heavy clay can be improved gradually with compost and drainage amendments rather than frequent watering.",
        ],
    },
    "sunlight": {
        "title": "Sunlight",
        "keywords": ["sun", "sunlight", "light", "shade", "bright", "window"],
        "response": "Aim for 6-8 hours of bright light daily. Morning and midday sun are helpful; in very hot climates, light afternoon shade can prevent leaf stress. Indoors, place it close to a bright window.",
        "responses": [
            "Give Stevia 6-8 hours of bright light each day. Use gentle afternoon shade in very hot weather, or place an indoor plant close to a bright window.",
            "Light target: the uploaded SPROUT range is 6-8 hours of direct sun. Less than 6 hours may lead to spindly growth and fewer leaves.",
            "A useful test is the plant's shape: long, weak stems usually mean more light is needed. Increase exposure gradually so leaves can adjust without scorching.",
        ],
    },
    "humidity": {
        "title": "Humidity",
        "keywords": ["humidity", "humid", "air", "dry air", "mist"],
        "response": "Average household humidity is usually fine. Stevia appreciates moderate humidity with good airflow, but avoid constantly wet leaves or stagnant air because they can encourage fungal problems.",
        "responses": [
            "Stevia does well around moderate household humidity. Keep air moving and avoid wetting the leaves repeatedly, since stagnant dampness encourages fungal problems.",
            "Humidity target: SPROUT classifies 65-80% relative humidity as optimal. Below that, watch for curling; above 80%, prioritize ventilation and reduce misting.",
            "Do not solve dry air by keeping foliage constantly wet. A small humidifier, spacing between plants, and gentle airflow are safer than frequent leaf misting.",
        ],
    },
    "fertilization": {
        "title": "Fertilization",
        "keywords": ["fertilize", "fertilizer", "feed", "nutrient", "compost"],
        "response": "Feed lightly during active growth with compost or a balanced fertilizer diluted to about half strength every 4-6 weeks. Avoid heavy nitrogen feeding, which can produce lush growth with less concentrated flavor.",
        "responses": [
            "Feed during active growth with compost or a balanced fertilizer at about half strength every 4-6 weeks. Light feeding is safer than pushing rapid, soft growth.",
            "Fertilizer routine: water first if the soil is dry, apply a diluted balanced feed, and avoid heavy nitrogen. Excess nitrogen can mean lots of foliage with less concentrated flavor.",
            "When growth is slow, check light, temperature, and root drainage before adding more fertilizer. Healthy roots and adequate light matter more than frequent feeding.",
        ],
    },
    "problems": {
        "title": "Common problems",
        "keywords": ["problem", "pest", "insect", "yellow", "wilting", "fungus", "disease", "prune"],
        "response": "Yellow leaves often point to overwatering, poor drainage, or low light; wilting can mean either dry soil or damaged roots. Inspect for aphids and spider mites, remove affected leaves, improve airflow, and adjust watering. Pruning growing tips encourages a bushier plant.",
        "responses": [
            "For yellow leaves, first check wet soil, drainage, and light. For wilting, feel the soil before watering because both drought and damaged roots can cause it. Inspect undersides of leaves for pests.",
            "Troubleshooting order: examine the roots and soil, look for aphids or spider mites, remove badly affected leaves, then improve airflow and adjust the watering routine.",
            "Common warning signs include yellowing, curling, wilt, and leaf spots. Avoid treating every symptom with more water; crowded, wet foliage can increase disease risk. Pinch growing tips to encourage branching.",
        ],
    },
}

WELCOME_MESSAGE = "Ask me about Stevia's light, water, soil, temperature, humidity, feeding, or common care problems."

SUGGESTED_QUESTIONS = [
    "How much sunlight does Stevia need?",
    "How often should I water it?",
    "What temperature is best for Stevia?",
]


JSON_TOPIC_MAP = {
    "soil": "soil_pH",
    "temperature": "temperature",
    "humidity": "relative_humidity",
    "sunlight": "light_intensity",
    "watering": "soil_moisture",
}


def _json_topic_responses(topic: str) -> list[str]:
    """Turn the uploaded JSON parameter record into answer variants."""
    parameter = JSON_TOPIC_MAP.get(topic)
    item = next((entry for entry in SPROUT_KNOWLEDGE if entry.get("parameter") == parameter), None)
    if not item:
        return []

    unit = item.get("unit", "").replace("_", " ")
    optimal_range = item.get("optimal_range")
    if optimal_range:
        optimal_text = f"{optimal_range[0]}-{optimal_range[1]} {unit}"
    else:
        optimal_text = "consistently moist but well-drained soil"

    classifications = item.get("classifications", [])
    status_lines = "; ".join(
        f"{entry.get('label')}: {entry.get('interpretation')} Recommended action: {entry.get('recommendation')}"
        for entry in classifications
    )
    return [
        f"According to the SPROUT Stevia dataset, the target for {parameter.replace('_', ' ')} is {optimal_text}. {classifications[0].get('interpretation', '') if classifications else ''}",
        f"Dataset care guide for {parameter.replace('_', ' ')}: {status_lines}",
    ]


for _topic in KNOWLEDGE_BASE:
    _json_responses = _json_topic_responses(_topic)
    if _json_responses:
        KNOWLEDGE_BASE[_topic]["responses"] = _json_responses + KNOWLEDGE_BASE[_topic].get(
            "responses", [KNOWLEDGE_BASE[_topic]["response"]]
        )
