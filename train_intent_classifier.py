import argparse
import difflib
import json
import os
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_NAME = "bert-base-uncased"
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
    precision = np.divide(true_positives, confusion.sum(axis=0), out=np.zeros_like(true_positives), where=confusion.sum(axis=0) != 0)
    recall = np.divide(true_positives, confusion.sum(axis=1), out=np.zeros_like(true_positives), where=confusion.sum(axis=1) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) != 0)

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
    model,
    tokenizer,
    chatbot,
    confidence_threshold: float,
) -> tuple[list[str], list[str]]:
    from stevia_bot.knowledge_base import KNOWLEDGE_BASE

    questions = frame.iloc[indices]["text"].tolist()
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

    predictions = []
    truths = frame.iloc[indices]["intent"].tolist()
    for question, probabilities_row in zip(questions, probabilities):
        if chatbot._asks_for_sensor_records(question):
            predictions.append("sensor_records")
            continue

        scores = {
            model.config.id2label[index]: float(probability)
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
    model,
    tokenizer,
) -> float:
    from stevia_bot.chatbot import SteviaChatbot

    chatbot = SteviaChatbot()
    best_threshold = 1.0
    best_accuracy = -1.0
    for threshold in np.linspace(0.0, 1.0, 101):
        truths, predictions = _routing_predictions(
            frame, validation_indices, model, tokenizer, chatbot, float(threshold)
        )
        accuracy = sum(actual == predicted for actual, predicted in zip(truths, predictions)) / len(truths)
        if accuracy > best_accuracy or (
            accuracy == best_accuracy and threshold > best_threshold
        ):
            best_accuracy = accuracy
            best_threshold = float(threshold)
    return best_threshold


def evaluate_chatbot_routing(
    frame: pd.DataFrame,
    test_indices: list[int],
    model,
    tokenizer,
    class_names: list[str],
) -> dict:
    from stevia_bot.chatbot import SteviaChatbot

    chatbot = SteviaChatbot()
    chatbot.model = model
    chatbot.tokenizer = tokenizer
    chatbot._model_loaded = True
    label_to_id = {label: index for index, label in enumerate(class_names)}
    truths, routed_intents = _routing_predictions(
        frame,
        test_indices,
        model,
        tokenizer,
        chatbot,
        float(getattr(model.config, "intent_confidence_threshold", 1.0)),
    )
    labels = [label_to_id[label] for label in truths]
    predictions = [
        label_to_id.get(intent, label_to_id.get("general", 0))
        for intent in routed_intents
    ]

    one_hot_logits = torch.full((len(labels), len(class_names)), -1.0)
    for index, prediction in enumerate(predictions):
        one_hot_logits[index, prediction] = 1.0
    return classification_metrics(one_hot_logits, torch.tensor(labels), class_names)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune and evaluate the Stevia intent classifier.")
    parser.add_argument("--data", type=Path, default=Path("intent_training_data.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/stevia-intent-bert"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--patience", type=int, default=2)
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("Batch size must be a positive integer.")
    if args.epochs < 1:
        raise ValueError("Epoch count must be a positive integer.")
    if args.learning_rate <= 0:
        raise ValueError("Learning rate must be positive.")
    if args.patience < 1:
        raise ValueError("Early-stopping patience must be positive.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, os.cpu_count() or 1))

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
    labels = [label_to_id[name] for name in frame["intent"].tolist()]
    splits = split_indices(
        frame["text"].tolist(),
        frame["intent"].tolist(),
        args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    split_manifest = {
        split_name: [int(index) for index in indices]
        for split_name, indices in splits.items()
    }
    (args.output_dir / "split_indices.json").write_text(
        json.dumps(split_manifest, indent=2), encoding="utf-8"
    )

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
            f"validation_loss={validation['loss']:.4f} "
            f"validation_macro_f1={validation['macro_f1']:.4f}",
            flush=True,
        )
        if validation["loss"] < best_validation_loss:
            best_validation_loss = validation["loss"]
            best_epoch = epoch
            epochs_without_improvement = 0
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping after epoch {epoch}.", flush=True)
                break

    best_model = AutoModelForSequenceClassification.from_pretrained(args.output_dir).to(device)
    validation_logits, validation_labels = collect_logits(best_model, validation_loader, device)
    confidence_temperature = fit_temperature(validation_logits, validation_labels)
    best_model.config.confidence_temperature = confidence_temperature
    intent_confidence_threshold = fit_intent_confidence_threshold(
        frame,
        splits["validation"],
        best_model,
        tokenizer,
    )
    best_model.config.intent_confidence_threshold = intent_confidence_threshold
    best_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    test_logits, test_labels = collect_logits(best_model, test_loader, device)
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
    chatbot_routing_metrics = evaluate_chatbot_routing(
        frame,
        splits["test"],
        best_model,
        tokenizer,
        class_names,
    )
    report = {
        "model_name": MODEL_NAME,
        "epochs_completed": epoch,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "early_stopping_patience": args.patience,
        "training_method": "full BERT fine-tuning with validation-loss early stopping",
        "seed": args.seed,
        "best_epoch_by_validation_loss": best_epoch,
        "best_validation_loss": best_validation_loss,
        "confidence_temperature": confidence_temperature,
        "intent_confidence_threshold": intent_confidence_threshold,
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
    print("Held-out test metrics:", json.dumps(test_metrics, indent=2), flush=True)
    print(
        "Chatbot routing test metrics:",
        json.dumps(chatbot_routing_metrics, indent=2),
        flush=True,
    )
    print(
        f"Best validation epoch: {best_epoch}; confidence temperature: {confidence_temperature:.3f}; "
        f"saved model and report to {args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()