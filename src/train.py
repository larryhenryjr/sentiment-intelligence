"""Train and evaluate a BiLSTM sentiment classifier."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from data import load_dataset

SEED = 42
def set_seed(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_model(vocabulary_size: int, sequence_length: int) -> Sequential:
    model = Sequential(
        [
            Embedding(vocabulary_size, 96, input_length=sequence_length),
            Bidirectional(LSTM(64)),
            Dropout(0.35),
            Dense(32, activation="relu"),
            Dropout(0.25),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def train(data_dir: Path, output_dir: Path, epochs: int) -> dict[str, float]:
    set_seed()
    dataset = load_dataset(data_dir)
    train_text, test_text, y_train, y_test = train_test_split(
        dataset["text"],
        dataset["label"],
        test_size=0.2,
        random_state=SEED,
        stratify=dataset["label"],
    )

    tokenizer = Tokenizer(num_words=12_000, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_text)
    max_length = 80
    x_train = pad_sequences(tokenizer.texts_to_sequences(train_text), maxlen=max_length)
    x_test = pad_sequences(tokenizer.texts_to_sequences(test_text), maxlen=max_length)

    model = build_model(min(12_000, len(tokenizer.word_index) + 1), max_length)
    history = model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=32,
        callbacks=[EarlyStopping(monitor="val_auc", mode="max", patience=3, restore_best_weights=True)],
        verbose=2,
    )

    loss, accuracy, auc = model.evaluate(x_test, y_test, verbose=0)
    probabilities = model.predict(x_test, verbose=0).reshape(-1)
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "test_loss": float(loss),
        "test_accuracy": float(accuracy),
        "test_auc": float(roc_auc_score(y_test, probabilities)),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir / "sentiment_bilstm.keras")
    (output_dir / "tokenizer.json").write_text(tokenizer.to_json(), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "history.json").write_text(
        json.dumps({key: [float(value) for value in values] for key, values in history.history.items()}, indent=2),
        encoding="utf-8",
    )
    (output_dir / "classification_report.txt").write_text(
        classification_report(y_test, predictions),
        encoding="utf-8",
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--epochs", type=int, default=15)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = train(args.data_dir, args.output_dir, args.epochs)
    print(json.dumps(results, indent=2))
