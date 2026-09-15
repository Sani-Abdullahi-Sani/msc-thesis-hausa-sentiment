
import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}
MODEL_NAME = "castorini/afriberta_base"
NATIVE_TRAIN = "data/matched/hausa/train.tsv"
NATIVE_TEST = "data/matched/hausa/test.tsv"
TRANSLATED_TRAIN = "data/matched/english_translated_hausa/train.tsv"


def load_tsv(path):
    df = pd.read_csv(path, sep="\t")
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["label"].isin(LABEL_MAP.keys())].reset_index(drop=True)
    df["label_id"] = df["label"].map(LABEL_MAP)
    return df


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            list(texts), truncation=True, padding="max_length",
            max_length=max_length, return_tensors="pt",
        )
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=["native", "translated"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--output_base", type=str, default="results_afriberta")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    set_seed(args.seed)

    print(f"Condition={args.condition}  Seed={args.seed}  Epochs={args.epochs}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")

    train_path = NATIVE_TRAIN if args.condition == "native" else TRANSLATED_TRAIN
    train_df = load_tsv(train_path)
    test_df = load_tsv(NATIVE_TEST)  # always evaluate on native Hausa test set
    print(f"Train examples: {len(train_df)}  (from {train_path})")
    print(f"Test examples:  {len(test_df)}  (from {NATIVE_TEST})")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

    train_dataset = SentimentDataset(train_df["tweet"], train_df["label_id"], tokenizer)
    test_dataset = SentimentDataset(test_df["tweet"], test_df["label_id"], tokenizer)

    out_dir = Path(args.output_base) / f"{args.condition}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        logging_steps=50,
        save_strategy="no",
        report_to=[],
        seed=args.seed,
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)
    trainer.train()

    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=1)
    labels = predictions.label_ids

    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")

    results = {
        "condition": args.condition,
        "seed": args.seed,
        "epochs": args.epochs,
        "test_accuracy": float(acc),
        "test_macro_f1": float(macro_f1),
        "n_train": len(train_df),
        "n_test": len(test_df),
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n===== RESULT =====")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
