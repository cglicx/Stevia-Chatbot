import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_NAME = "bert-base-uncased"
EXPECTED_COLUMNS = {"text", "intent"}


def split_indices(labels: list[str], seed: int) -> dict[str, list[int]]:
    rng = np.random.default_rng(seed)
    splits = {"train": [], "validation": [], "test": []}

    for label in sorted(set(labels)):
        class_indices = np.flatnonzero(np.asarray(labels) == label)
        shuffled = rng.permutation(class_indices).tolist()
        if len(shuffled) < 3:
            raise ValueError(f"Intent {label!r} needs at least 3 examples for TTV splitting.")

        train_end = max(1, int(len(shuffled) * 0.70))
        validation_count = max(1, int(len(shuffled) * 0.15))
        if train_end + validation_count >= len(shuffled):
            train_end = len(shuffled) - 2
            validation_count = 1

        splits["train"].extend(shuffled[:train_end])
        splits["validation"].extend(shuffled[train_end:train_end + validation_count])
        splits["test"].extend(shuffled[train_end + validation_count:])

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


def encode_dataset(model, dataset: TensorDataset, batch_size: int, device: torch.device) -> TensorDataset:
    model.eval()
    features = []
    labels = []
    loader = DataLoader(dataset, batch_size=batch_size)
    with torch.no_grad():
        for input_ids, attention_mask, batch_labels in loader:
            outputs = model.bert(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                return_dict=True,
            )
            features.append(outputs.pooler_output.cpu())
            labels.append(batch_labels)
    return TensorDataset(torch.cat(features), torch.cat(labels))


def evaluate(model, loader: DataLoader, device: torch.device, class_names: list[str]) -> dict:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    confusion = np.zeros((len(class_names), len(class_names)), dtype=np.int64)

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            logits = model.classifier(model.dropout(features))
            loss = torch.nn.functional.cross_entropy(logits, labels)
            predictions = logits.argmax(dim=1)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size
            for actual, predicted in zip(labels.cpu().tolist(), predictions.cpu().tolist()):
                confusion[actual, predicted] += 1

    true_positives = np.diag(confusion).astype(float)
    precision = np.divide(true_positives, confusion.sum(axis=0), out=np.zeros_like(true_positives), where=confusion.sum(axis=0) != 0)
    recall = np.divide(true_positives, confusion.sum(axis=1), out=np.zeros_like(true_positives), where=confusion.sum(axis=1) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) != 0)

    return {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": float(true_positives.sum() / max(total_examples, 1)),
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune BERT to classify Stevia-care questions.")
    parser.add_argument("--data", type=Path, default=Path("intent_training_data.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/stevia-intent-bert"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("Batch size must be a positive integer.")
    if args.epochs < 1:
        raise ValueError("Epoch count must be a positive integer.")

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
    splits = split_indices(frame["intent"].tolist(), args.seed)
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
    print("Encoding texts with frozen BERT encoder...", flush=True)
    feature_datasets = {
        name: encode_dataset(model, dataset, args.batch_size, device)
        for name, dataset in datasets.items()
    }
    train_loader = DataLoader(feature_datasets["train"], batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(feature_datasets["validation"], batch_size=args.batch_size)
    test_loader = DataLoader(feature_datasets["test"], batch_size=args.batch_size)

    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3)
    best_validation_loss = float("inf")
    best_epoch = 0

    print(f"Device: {device}; batch size: {args.batch_size}; epochs: {args.epochs}", flush=True)
    print(f"TTV sizes: {len(datasets['train'])}/{len(datasets['validation'])}/{len(datasets['test'])}", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0
        for features, batch_labels in train_loader:
            features = features.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model.classifier(model.dropout(features))
            loss = torch.nn.functional.cross_entropy(logits, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_size = batch_labels.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size

        validation = evaluate(model, validation_loader, device, class_names)
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
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)

    best_model = AutoModelForSequenceClassification.from_pretrained(args.output_dir).to(device)
    test_metrics = evaluate(best_model, test_loader, device, class_names)
    report = {
        "model_name": MODEL_NAME,
        "epochs_completed": args.epochs,
        "batch_size": args.batch_size,
        "training_method": "frozen BERT encoder with trained classification head",
        "seed": args.seed,
        "best_epoch_by_validation_loss": best_epoch,
        "best_validation_loss": best_validation_loss,
        "split_sizes": {name: len(indices) for name, indices in splits.items()},
        "class_counts": frame["intent"].value_counts().sort_index().to_dict(),
        "test_metrics": test_metrics,
        "note": "Synthetic starter data; test metrics are not estimates of real-world performance.",
    }
    (args.output_dir / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Held-out test metrics:", json.dumps(test_metrics, indent=2), flush=True)
    print(f"Best validation epoch: {best_epoch}; saved model and report to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()