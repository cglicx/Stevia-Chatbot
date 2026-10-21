import argparse
import difflib
import json
import os
import random
import re
from pathlib import Path

<<<<<<< HEAD
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

from stevia_bot.intent_text import normalize_intent_text


MODEL_NAME = "multilingual-character-word-tfidf-sgd"
=======
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_NAME = "bert-base-uncased"
>>>>>>> 186cd39e424c87e5968dfb615c47a51e44355bd6
EXPECTED_COLUMNS = {"text", "intent"}


def _normalized_text(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower())


def split_indices(
    texts: list[str],
    labels: list[str],
    seed: int,
    similarity_threshold: float = 0.82,
) -> dict[str, list[int]]:
    if len(texts) != len(labels):
        raise ValueError("Texts and labels must have the same number of examples.")
    if not 0 < similarity_threshold <= 1:
        raise ValueError("Similarity threshold must be between 0 and 1.")

    rng = np.random.default_rng(seed)
    splits = {"train": [], "validation": [], "test": []}
    split_targets = {"train": 0.70, "validation": 0.15, "test": 0.15}

    for label in sorted(set(labels)):
        class_indices = np.flatnonzero(np.asarray(labels) == label).tolist()
        if len(class_indices) < 3:
            raise ValueError(f"Intent {label!r} needs at least 3 examples for TTV splitting.")

        parent = {index: index for index in class_indices}

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        normalized = {index: _normalized_text(texts[index]) for index in class_indices}
        for position, left in enumerate(class_indices):
            for right in class_indices[position + 1:]:
                similarity = difflib.SequenceMatcher(
                    None, normalized[left], normalized[right]
                ).ratio()
                if similarity >= similarity_threshold:
                    parent[find(right)] = find(left)

        groups: dict[int, list[int]] = {}
        for index in class_indices:
            groups.setdefault(find(index), []).append(index)
        grouped_indices = list(groups.values())
        if len(grouped_indices) < 3:
            raise ValueError(
                f"Intent {label!r} needs at least 3 distinct phrasing groups for TTV splitting."
            )

        order = rng.permutation(len(grouped_indices)).tolist()
        shuffled_groups = [grouped_indices[index] for index in order]
        shuffled_groups.sort(key=len, reverse=True)
        counts = {name: 0 for name in split_targets}

        for group_index, group in enumerate(shuffled_groups):
            if group_index < 3:
                split_name = ("train", "validation", "test")[group_index]
            else:
                split_name = min(
                    split_targets,
                    key=lambda name: (
                        (counts[name] + len(group) - split_targets[name] * len(class_indices))
                        / max(split_targets[name] * len(class_indices), 1)
                    ) ** 2
                    + sum(
                        (
                            (counts[other] - split_targets[other] * len(class_indices))
                            / max(split_targets[other] * len(class_indices), 1)
                        ) ** 2
                        for other in split_targets
                        if other != name
                    ),
                )
            splits[split_name].extend(group)
            counts[split_name] += len(group)

    for indices in splits.values():
        indices.sort()
    return splits


<<<<<<< HEAD
=======
def make_dataset(texts: list[str], labels: list[int], tokenizer) -> TensorDataset:
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )
    return TensorDataset(
        encoded["input_ids"],
        encoded["attention_mask"],
        torch.tensor(labels, dtype=torch.long),
    )


def collect_logits(model, loader: DataLoader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    logits = []
    labels = []
    with torch.no_grad():
        for input_ids, attention_mask, batch_labels in loader:
            outputs = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
            )
            logits.append(outputs.logits.cpu())
            labels.append(batch_labels)
    return torch.cat(logits), torch.cat(labels)


>>>>>>> 186cd39e424c87e5968dfb615c47a51e44355bd6
def classification_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_names: list[str],
    temperature: float = 1.0,
) -> dict:
    calibrated_logits = logits / temperature
    predictions = calibrated_logits.argmax(dim=1)
    loss = torch.nn.functional.cross_entropy(calibrated_logits, labels).item()
    confusion = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    for actual, predicted in zip(labels.tolist(), predictions.tolist()):
        confusion[actual, predicted] += 1

    true_positives = np.diag(confusion).astype(float)
<<<<<<< HEAD
    precision = np.divide(
        true_positives,
        confusion.sum(axis=0),
        out=np.zeros_like(true_positives),
        where=confusion.sum(axis=0) != 0,
    )
    recall = np.divide(
        true_positives,
        confusion.sum(axis=1),
        out=np.zeros_like(true_positives),
        where=confusion.sum(axis=1) != 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )
=======
    precision = np.divide(true_positives, confusion.sum(axis=0), out=np.zeros_like(true_positives), where=confusion.sum(axis=0) != 0)
    recall = np.divide(true_positives, confusion.sum(axis=1), out=np.zeros_like(true_positives), where=confusion.sum(axis=1) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) != 0)
>>>>>>> 186cd39e424c87e5968dfb615c47a51e44355bd6

    return {
        "loss": loss,
        "accuracy": float(true_positives.sum() / max(len(labels), 1)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "per_intent": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(confusion[index].sum()),
            }
            for index, name in enumerate(class_names)
        },
        "confusion_matrix": confusion.tolist(),
        "confusion_matrix_labels": class_names,
    }


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    temperatures = torch.logspace(
        np.log10(0.25), np.log10(5.0), steps=200, dtype=logits.dtype
    )
    losses = [
        torch.nn.functional.cross_entropy(logits / temperature, labels).item()
        for temperature in temperatures
    ]
    return float(temperatures[int(np.argmin(losses))])


def _routing_predictions(
    frame: pd.DataFrame,
    indices: list[int],
<<<<<<< HEAD
    model: Pipeline,
=======
    model,
    tokenizer,
>>>>>>> 186cd39e424c87e5968dfb615c47a51e44355bd6
    chatbot,
    confidence_threshold: float,
) -> tuple[list[str], list[str]]:
    from stevia_bot.knowledge_base import KNOWLEDGE_BASE

    questions = frame.iloc[indices]["text"].tolist()
<<<<<<< HEAD
    probabilities = model.predict_proba(questions)
    temperature = max(float(getattr(model, "confidence_temperature", 1.0)), 1e-6)
    log_probabilities = np.log(np.clip(probabilities, 1e-12, 1.0)) / temperature
    log_probabilities -= log_probabilities.max(axis=1, keepdims=True)
    probabilities = np.exp(log_probabilities)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    class_names = model.classes_
=======
    encoded = tokenizer(
        questions,
        truncation=True,
        max_length=64,
        padding=True,
        return_tensors="pt",
    )
    model.eval()
    device = next(model.parameters()).device
    encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
    with torch.no_grad():
        logits = model(**encoded).logits
        temperature = max(float(getattr(model.config, "confidence_temperature", 1.0)), 1e-6)
        probabilities = torch.softmax(logits / temperature, dim=-1)
>>>>>>> 186cd39e424c87e5968dfb615c47a51e44355bd6

    predictions = []
    truths = frame.iloc[indices]["intent"].tolist()
    for question, probabilities_row in zip(questions, probabilities):
        if chatbot._asks_for_sensor_records(question):
            predictions.append("sensor_records")
            continue

        scores = {
<<<<<<< HEAD
            str(class_names[index]): float(probability)
=======
            model.config.id2label[index]: float(probability)
>>>>>>> 186cd39e424c87e5968dfb615c47a51e44355bd6
            for index, probability in enumerate(probabilities_row)
        }
        model_intent = max(scores, key=scores.get)
        intent_keywords = {
            word
            for keyword in KNOWLEDGE_BASE.get(model_intent, {}).get("keywords", [])
            for word in keyword.lower().split()
        }
        negated_intent = bool(intent_keywords.intersection(chatbot._negated_words(question)))
        if (
            model_intent in chatbot._keyword_scores(question)
            and not negated_intent
            and scores[model_intent] >= confidence_threshold
        ):
            predictions.append(model_intent)
            continue

        keyword_scores = chatbot._keyword_scores(question)
        if any(key != "general" and value > 0 for key, value in keyword_scores.items()):
            keyword_scores["general"] = 0.0
        predictions.append(max(
            keyword_scores,
            key=lambda intent: keyword_scores[intent] + scores.get(intent, 0.0) * 0.35,
        ))
    return truths, predictions


def fit_intent_confidence_threshold(
    frame: pd.DataFrame,
    validation_indices: list[int],
<<<<<<< HEAD
    model: Pipeline,
=======
    model,
    tokenizer,
>>>>>>> 186cd39e424c87e5968dfb615c47a51e44355bd6
) -> float:
    from stevia_bot.chatbot import SteviaChatbot

    chatbot = SteviaChatbot()
    best_threshold = 1.0
    best_accuracy = -1.0
    for threshold in np.linspace(0.0, 1.0, 101):
        truths, predictions = _routing_predictions(
<<<<<<< HEAD
            frame, validation_indices, model, chatbot, float(threshold)
        )
        accuracy = sum(
            actual == predicted for actual, predicted in zip(truths, predictions)
        ) / len(truths)
=======
            frame, validation_indices, model, tokenizer, chatbot, float(threshold)
        )
        accuracy = sum(actual == predicted for actual, predicted in zip(truths, predictions)) / len(truths)
>>>>>>> 186cd39e424c87e5968dfb615c47a51e44355bd6
        if accuracy > best_accuracy or (
            accuracy == best_accuracy and threshold > best_threshold
        ):
            best_accuracy = accuracy
            best_threshold = float(threshold)
    return best_threshold


def evaluate_chatbot_routing(
    frame: pd.DataFrame,
    test_indices: list[int],
<<<<<<< HEAD
    model: Pipeline,
=======
    model,
    tokenizer,
>>>>>>> 186cd39e424c87e5968dfb615c47a51e44355bd6
    class_names: list[str],
) -> dict:
    from stevia_bot.chatbot import SteviaChatbot

    chatbot = SteviaChatbot()
    chatbot.model = model
<<<<<<< HEAD
=======
    chatbot.tokenizer = tokenizer
>>>>>>> 186cd39e424c87e5968dfb615c47a51e44355bd6
    chatbot._model_loaded = True
    label_to_id = {label: index for index, label in enumerate(class_names)}
    truths, routed_intents = _routing_predictions(
        frame,
        test_indices,
        model,
<<<<<<< HEAD
        chatbot,
        float(getattr(model, "intent_confidence_threshold", 1.0)),
=======
        tokenizer,
        chatbot,
        float(getattr(model.config, "intent_confidence_threshold", 1.0)),
>>>>>>> 186cd39e424c87e5968dfb615c47a51e44355bd6
    )
    labels = [label_to_id[label] for label in truths]
    predictions = [
        label_to_id.get(intent, label_to_id.get("general", 0))
        for intent in routed_intents
    ]
<<<<<<< HEAD
=======

>>>>>>> 186cd39e424c87e5968dfb615c47a51e44355bd6
    one_hot_logits = torch.full((len(labels), len(class_names)), -1.0)
    for index, prediction in enumerate(predictions):
        one_hot_logits[index, prediction] = 1.0
    return classification_metrics(one_hot_logits, torch.tensor(labels), class_names)


<<<<<<< HEAD
def evaluate_multilingual_smoke_tests(model: Pipeline) -> dict:
    from stevia_bot.chatbot import SteviaChatbot

    examples = [
        ("English", "What is Stevia?", "general"),
        ("English", "How can I grow Stevia in a pot?", "growing"),
        ("English", "How often should I water my Stevia plant?", "watering"),
        ("English", "What soil drains well for Stevia?", "soil"),
        ("English", "How much sunlight does Stevia need?", "sunlight"),
        ("English", "Is this temperature okay for Stevia?", "temperature"),
        ("English", "How humid should the air be for Stevia?", "humidity"),
        ("English", "When should I fertilize Stevia?", "fertilization"),
        ("English", "Why are my Stevia leaves turning yellow?", "problems"),
        ("English", "Show me the latest sensor records.", "sensor_records"),
        ("Tagalog", "Ano ang stevia?", "general"),
        ("Tagalog", "Paano magtanim ng stevia sa paso?", "growing"),
        ("Tagalog", "Gaano kadalas ko dapat diligan ang stevia?", "watering"),
        ("Tagalog", "Anong lupa ang bagay sa stevia?", "soil"),
        ("Tagalog", "Gaano karaming sikat ng araw ang kailangan ng stevia?", "sunlight"),
        ("Tagalog", "Anong temperatura ang okay para sa stevia?", "temperature"),
        ("Tagalog", "Ano ang tamang halumigmig para sa stevia?", "humidity"),
        ("Tagalog", "Kailan dapat lagyan ng pataba ang stevia?", "fertilization"),
        ("Tagalog", "Bakit naninilaw ang dahon ng stevia?", "problems"),
        ("Tagalog", "Ipakita ang pinakabagong sensor records.", "sensor_records"),
        ("Taglish", "Ano ba yung stevia, plant ba siya?", "general"),
        ("Taglish", "Paano ko i-grow yung stevia sa pot?", "growing"),
        ("Taglish", "Gaano kadalas mag-water ng stevia, bes?", "watering"),
        ("Taglish", "Anong soil mix ang swak sa stevia?", "soil"),
        ("Taglish", "Need ba ng full sunlight yung stevia?", "sunlight"),
        ("Taglish", "Mainit masyado yung temp, okay pa ba sa stevia?", "temperature"),
        ("Taglish", "Ano ideal humidity para hindi sad yung leaves?", "humidity"),
        ("Taglish", "Kailan best mag-fertilize, bes?", "fertilization"),
        ("Taglish", "Bakit yellow na yung leaves ng stevia ko?", "problems"),
        ("Taglish", "Pakita latest sensor records, pls.", "sensor_records"),
    ]
    texts = [text for _, text, _ in examples]
    expected = [intent for _, _, intent in examples]
    probabilities = model.predict_proba(texts)
    predictions = [str(model.classes_[index]) for index in probabilities.argmax(axis=1)]
    chatbot = SteviaChatbot()
    chatbot.model = model
    chatbot._model_loaded = True
    frame = pd.DataFrame({"text": texts, "intent": expected})
    _, routed_predictions = _routing_predictions(
        frame,
        list(range(len(examples))),
        model,
        chatbot,
        float(getattr(model, "intent_confidence_threshold", 1.0)),
    )

    per_language = {}
    for language in ("English", "Tagalog", "Taglish"):
        language_indices = [
            index for index, (sample_language, _, _) in enumerate(examples)
            if sample_language == language
        ]
        per_language[language] = {
            "support": len(language_indices),
            "accuracy": float(np.mean([
                predictions[index] == expected[index] for index in language_indices
            ])),
            "routed_accuracy": float(np.mean([
                routed_predictions[index] == expected[index] for index in language_indices
            ])),
        }

    return {
        "per_language": per_language,
        "classifier_accuracy": float(np.mean([
            predicted == actual for predicted, actual in zip(predictions, expected)
        ])),
        "routed_accuracy": float(np.mean([
            predicted == actual for predicted, actual in zip(routed_predictions, expected)
        ])),
        "cases": [
            {
                "language": language,
                "text": text,
                "expected": intent,
                "predicted": prediction,
                "routed_intent": routed,
            }
            for (language, text, intent), prediction, routed in zip(
                examples, predictions, routed_predictions
            )
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the Stevia multilingual intent classifier."
    )
    parser.add_argument("--data", type=Path, default=Path("intent_training_data.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/stevia-intent-bert"))
    parser.add_argument(
        "--baseline-report",
        type=Path,
        default=Path("models/stevia-intent-bert/previous_evaluation.json"),
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=5)
=======
def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune and evaluate the Stevia intent classifier.")
    parser.add_argument("--data", type=Path, default=Path("intent_training_data.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/stevia-intent-bert"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--patience", type=int, default=2)
>>>>>>> 186cd39e424c87e5968dfb615c47a51e44355bd6
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("Batch size must be a positive integer.")
    if args.epochs < 1:
        raise ValueError("Epoch count must be a positive integer.")
    if args.learning_rate <= 0:
        raise ValueError("Learning rate must be positive.")
<<<<<<< HEAD
    if args.alpha <= 0:
        raise ValueError("Regularization alpha must be positive.")
=======
>>>>>>> 186cd39e424c87e5968dfb615c47a51e44355bd6
    if args.patience < 1:
        raise ValueError("Early-stopping patience must be positive.")

    random.seed(args.seed)
    np.random.seed(args.seed)
<<<<<<< HEAD
    baseline_report_path = args.baseline_report
    if (
        not baseline_report_path.is_file()
        and baseline_report_path.name == "previous_evaluation.json"
    ):
        baseline_report_path = baseline_report_path.with_name("evaluation.json")
    baseline = None
    if baseline_report_path.is_file():
        baseline = json.loads(baseline_report_path.read_text(encoding="utf-8"))
=======
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
>>>>>>> 186cd39e424c87e5968dfb615c47a51e44355bd6

    frame = pd.read_csv(args.data)
    if not EXPECTED_COLUMNS.issubset(frame.columns):
        raise ValueError("CSV must contain 'text' and 'intent' columns.")
    frame = frame[["text", "intent"]].dropna()
    frame["text"] = frame["text"].astype(str).str.strip()
    frame["intent"] = frame["intent"].astype(str).str.strip()
    frame = frame[(frame["text"] != "") & (frame["intent"] != "")].reset_index(drop=True)
    if frame.empty:
        raise ValueError("No usable labeled rows were found in the CSV.")

    class_names = sorted(frame["intent"].unique().tolist())
    label_to_id = {name: index for index, name in enumerate(class_names)}
<<<<<<< HEAD
    splits = split_indices(frame["text"].tolist(), frame["intent"].tolist(), args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
=======
    labels = [label_to_id[name] for name in frame["intent"].tolist()]
    splits = split_indices(
        frame["text"].tolist(),
        frame["intent"].tolist(),
        args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

>>>>>>> 186cd39e424c87e5968dfb615c47a51e44355bd6
    split_manifest = {
        split_name: [int(index) for index in indices]
        for split_name, indices in splits.items()
    }
    (args.output_dir / "split_indices.json").write_text(
        json.dumps(split_manifest, indent=2), encoding="utf-8"
    )

<<<<<<< HEAD
    vectorizer = FeatureUnion(
        [
            ("char", TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(2, 5),
                max_features=100_000,
                min_df=1,
                preprocessor=normalize_intent_text,
                sublinear_tf=True,
                dtype=np.float32,
            )),
            ("word", TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                token_pattern=r"(?u)\b\w+\b",
                min_df=1,
                preprocessor=normalize_intent_text,
                sublinear_tf=True,
                dtype=np.float32,
            )),
        ],
        transformer_weights={"char": 1.0, "word": 0.8},
    )
    train_texts = frame.iloc[splits["train"]]["text"].tolist()
    train_features = vectorizer.fit_transform(train_texts)
    features = {
        split_name: vectorizer.transform(frame.iloc[indices]["text"].tolist())
        for split_name, indices in splits.items()
    }
    training_labels = np.asarray(frame.iloc[splits["train"]]["intent"])
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.asarray(class_names),
        y=training_labels,
    )
    classifier = SGDClassifier(
        loss="log_loss",
        alpha=args.alpha,
        learning_rate="constant",
        eta0=args.learning_rate,
        class_weight=dict(zip(class_names, class_weights)),
        average=True,
        random_state=args.seed,
    )
    model = Pipeline([("vectorizer", vectorizer), ("classifier", classifier)])
    validation_labels = torch.tensor(
        [label_to_id[label] for label in frame.iloc[splits["validation"]]["intent"]],
        dtype=torch.long,
    )
    rng = np.random.default_rng(args.seed)
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Device: CPU; model: {MODEL_NAME}; batch size: {args.batch_size}; "
        f"maximum epochs: {args.epochs}; learning rate: {args.learning_rate}; "
        f"early-stopping patience: {args.patience}",
        flush=True,
    )
    vocabulary_sizes = {
        name: len(transformer.vocabulary_)
        for name, transformer in vectorizer.transformer_list
    }
    vocabulary_size = sum(vocabulary_sizes.values())
    print(
        f"CPU threads: {min(4, os.cpu_count() or 1)}; split sizes: "
        f"{len(splits['train'])}/{len(splits['validation'])}/{len(splits['test'])}; "
        f"vocabulary size: {vocabulary_size}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(training_labels))
        for start in range(0, len(order), args.batch_size):
            batch_indices = order[start:start + args.batch_size]
            classifier.partial_fit(
                train_features[batch_indices],
                training_labels[batch_indices],
                classes=np.asarray(class_names),
            )

        validation_probabilities = classifier.predict_proba(features["validation"])
        validation_logits = torch.from_numpy(
            np.log(np.clip(validation_probabilities, 1e-12, 1.0)).astype(np.float32)
        )
        validation = classification_metrics(
            validation_logits,
            validation_labels,
            class_names,
        )
        print(
            f"Epoch {epoch:02d}/{args.epochs} "
=======
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(class_names),
        id2label={index: name for name, index in label_to_id.items()},
        label2id=label_to_id,
    )

    datasets = {
        split_name: make_dataset(
            frame.iloc[indices]["text"].tolist(),
            [labels[index] for index in indices],
            tokenizer,
        )
        for split_name, indices in splits.items()
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_loader = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(datasets["validation"], batch_size=args.batch_size)
    test_loader = DataLoader(datasets["test"], batch_size=args.batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    print(
        f"Device: {device}; batch size: {args.batch_size}; maximum epochs: {args.epochs}; "
        f"learning rate: {args.learning_rate}; early-stopping patience: {args.patience}",
        flush=True,
    )
    print(f"TTV sizes: {len(datasets['train'])}/{len(datasets['validation'])}/{len(datasets['test'])}", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0
        for input_ids, attention_mask, batch_labels in train_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=batch_labels,
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_size = batch_labels.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size

        validation_logits, validation_labels = collect_logits(model, validation_loader, device)
        validation = classification_metrics(validation_logits, validation_labels, class_names)
        train_loss = total_loss / max(total_examples, 1)
        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"train_loss={train_loss:.4f} "
>>>>>>> 186cd39e424c87e5968dfb615c47a51e44355bd6
            f"validation_loss={validation['loss']:.4f} "
            f"validation_macro_f1={validation['macro_f1']:.4f}",
            flush=True,
        )
        if validation["loss"] < best_validation_loss:
            best_validation_loss = validation["loss"]
            best_epoch = epoch
            epochs_without_improvement = 0
<<<<<<< HEAD
            joblib.dump(model, args.output_dir / "intent_classifier.joblib", compress=3)
=======
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
>>>>>>> 186cd39e424c87e5968dfb615c47a51e44355bd6
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping after epoch {epoch}.", flush=True)
                break

<<<<<<< HEAD
    best_model = joblib.load(args.output_dir / "intent_classifier.joblib")
    validation_probabilities = best_model.named_steps["classifier"].predict_proba(
        features["validation"]
    )
    validation_logits = torch.from_numpy(
        np.log(np.clip(validation_probabilities, 1e-12, 1.0)).astype(np.float32)
    )
    confidence_temperature = fit_temperature(validation_logits, validation_labels)
    best_model.confidence_temperature = confidence_temperature
=======
    best_model = AutoModelForSequenceClassification.from_pretrained(args.output_dir).to(device)
    validation_logits, validation_labels = collect_logits(best_model, validation_loader, device)
    confidence_temperature = fit_temperature(validation_logits, validation_labels)
    best_model.config.confidence_temperature = confidence_temperature
>>>>>>> 186cd39e424c87e5968dfb615c47a51e44355bd6
    intent_confidence_threshold = fit_intent_confidence_threshold(
        frame,
        splits["validation"],
        best_model,
<<<<<<< HEAD
    )
    best_model.intent_confidence_threshold = intent_confidence_threshold
    joblib.dump(best_model, args.output_dir / "intent_classifier.joblib", compress=3)

    test_labels = torch.tensor(
        [label_to_id[label] for label in frame.iloc[splits["test"]]["intent"]],
        dtype=torch.long,
    )
    test_probabilities = best_model.named_steps["classifier"].predict_proba(features["test"])
    test_logits = torch.from_numpy(
        np.log(np.clip(test_probabilities, 1e-12, 1.0)).astype(np.float32)
    )
=======
        tokenizer,
    )
    best_model.config.intent_confidence_threshold = intent_confidence_threshold
    best_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    test_logits, test_labels = collect_logits(best_model, test_loader, device)
>>>>>>> 186cd39e424c87e5968dfb615c47a51e44355bd6
    test_metrics = classification_metrics(
        test_logits,
        test_labels,
        class_names,
        temperature=confidence_temperature,
    )
    test_metrics["uncalibrated_accuracy"] = classification_metrics(
        test_logits,
        test_labels,
        class_names,
    )["accuracy"]
<<<<<<< HEAD
    test_metrics["classification_report"] = classification_report(
        test_labels.numpy(),
        (test_logits / confidence_temperature).argmax(dim=1).numpy(),
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
=======
>>>>>>> 186cd39e424c87e5968dfb615c47a51e44355bd6
    chatbot_routing_metrics = evaluate_chatbot_routing(
        frame,
        splits["test"],
        best_model,
<<<<<<< HEAD
        class_names,
    )
    multilingual_smoke_tests = evaluate_multilingual_smoke_tests(best_model)

    baseline_comparison = None
    if baseline:
        previous_metrics = baseline.get("test_metrics", {})
        baseline_comparison = {
            "source": str(baseline_report_path),
            "previous_model_name": baseline.get("model_name"),
            "previous_test_accuracy": previous_metrics.get("accuracy"),
            "previous_test_macro_f1": previous_metrics.get("macro_f1"),
            "accuracy_delta": (
                test_metrics["accuracy"] - previous_metrics["accuracy"]
                if previous_metrics.get("accuracy") is not None else None
            ),
            "macro_f1_delta": (
                test_metrics["macro_f1"] - previous_metrics["macro_f1"]
                if previous_metrics.get("macro_f1") is not None else None
            ),
            "comparable_test_sets": False,
            "note": (
                "The previous report used a different English-only dataset and 58 test samples. "
                "Metric deltas are descriptive, not a controlled comparison."
            ),
        }

    model_config = {
        "model_name": MODEL_NAME,
        "model_type": (
            "TF-IDF character and word n-gram features with Tagalog/Taglish normalization "
            "and incremental SGD logistic regression"
        ),
        "labels": class_names,
        "vectorizer": {
            "type": "character and word n-gram TF-IDF feature union",
            "normalization": "common Tagalog and Taglish care terms normalized before vectorization",
            "features": {
                "character": {
                    "analyzer": "char_wb",
                    "ngram_range": [2, 5],
                    "vocabulary_size": vocabulary_sizes["char"],
                },
                "word": {
                    "analyzer": "word",
                    "ngram_range": [1, 2],
                    "vocabulary_size": vocabulary_sizes["word"],
                },
            },
            "feature_weights": {"character": 1.0, "word": 0.8},
            "total_vocabulary_size": vocabulary_size,
        },
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "alpha": args.alpha,
        "epochs_completed": epoch,
        "best_epoch": best_epoch,
        "early_stopping_patience": args.patience,
        "confidence_temperature": confidence_temperature,
        "intent_confidence_threshold": intent_confidence_threshold,
        "training_device": "cpu",
    }
    (args.output_dir / "model_config.json").write_text(
        json.dumps(model_config, indent=2), encoding="utf-8"
    )
=======
        tokenizer,
        class_names,
    )
>>>>>>> 186cd39e424c87e5968dfb615c47a51e44355bd6
    report = {
        "model_name": MODEL_NAME,
        "epochs_completed": epoch,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
<<<<<<< HEAD
        "alpha": args.alpha,
        "early_stopping_patience": args.patience,
        "training_method": (
            "Multilingual character and word n-gram TF-IDF with Tagalog/Taglish feature "
            "normalization and minibatch SGD logistic regression; best validation-loss "
            "checkpoint and early stopping"
        ),
        "training_device": {
            "device": "CPU",
            "torch_cuda_available": torch.cuda.is_available(),
            "cpu_threads": min(4, os.cpu_count() or 1),
        },
=======
        "early_stopping_patience": args.patience,
        "training_method": "full BERT fine-tuning with validation-loss early stopping",
>>>>>>> 186cd39e424c87e5968dfb615c47a51e44355bd6
        "seed": args.seed,
        "best_epoch_by_validation_loss": best_epoch,
        "best_validation_loss": best_validation_loss,
        "confidence_temperature": confidence_temperature,
        "intent_confidence_threshold": intent_confidence_threshold,
<<<<<<< HEAD
        "split_method": (
            "stratified by intent and grouped by same-intent text similarity "
            "(threshold 0.82)"
        ),
        "split_sizes": {name: len(indices) for name, indices in splits.items()},
        "class_counts": frame["intent"].value_counts().sort_index().to_dict(),
        "vocabulary_size": vocabulary_size,
        "test_metrics": test_metrics,
        "chatbot_routing_test_metrics": chatbot_routing_metrics,
        "multilingual_smoke_tests": multilingual_smoke_tests,
        "baseline_comparison": baseline_comparison,
        "note": (
            "Training phrases are synthetic. Held-out test scores are not estimates of real-world "
            "performance; validate against independently collected, consented user conversations "
            "before deployment. The multilingual smoke suite is a small manually written check, "
            "not a statistically representative benchmark."
        ),
    }
    (args.output_dir / "evaluation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
=======
        "split_method": "stratified by intent and grouped by same-intent text similarity (threshold 0.82)",
        "split_sizes": {name: len(indices) for name, indices in splits.items()},
        "class_counts": frame["intent"].value_counts().sort_index().to_dict(),
        "test_metrics": test_metrics,
        "chatbot_routing_test_metrics": chatbot_routing_metrics,
        "note": (
            "Training phrases are synthetic. Held-out test scores are not estimates of real-world "
            "performance; validate against independently collected user conversations before deployment."
        ),
    }
    (args.output_dir / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
>>>>>>> 186cd39e424c87e5968dfb615c47a51e44355bd6
    print("Held-out test metrics:", json.dumps(test_metrics, indent=2), flush=True)
    print(
        "Chatbot routing test metrics:",
        json.dumps(chatbot_routing_metrics, indent=2),
        flush=True,
    )
<<<<<<< HEAD
    print("Multilingual smoke tests:", json.dumps(multilingual_smoke_tests, indent=2), flush=True)
    print(
        f"Best validation epoch: {best_epoch}; confidence temperature: "
        f"{confidence_temperature:.3f}; saved model and report to {args.output_dir}",
=======
    print(
        f"Best validation epoch: {best_epoch}; confidence temperature: {confidence_temperature:.3f}; "
        f"saved model and report to {args.output_dir}",
>>>>>>> 186cd39e424c87e5968dfb615c47a51e44355bd6
        flush=True,
    )


if __name__ == "__main__":
<<<<<<< HEAD
    main()
=======
    main()
>>>>>>> 186cd39e424c87e5968dfb615c47a51e44355bd6
