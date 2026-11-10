"""Data loading and validation utilities for sentiment modeling."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SOURCE_FILES = {
    "amazon": "amazon_cells_labelled.txt",
    "imdb": "imdb_labelled.txt",
    "yelp": "yelp_labelled.txt",
}


def load_dataset(data_dir: Path) -> pd.DataFrame:
    """Load, validate, and deduplicate the three labelled-sentence files."""
    frames: list[pd.DataFrame] = []
    for source, filename in SOURCE_FILES.items():
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. See data/README.md for setup.")

        frame = pd.read_csv(path, sep="\t", names=["text", "label"], quoting=3)
        frame["source"] = source
        frames.append(frame)

    dataset = pd.concat(frames, ignore_index=True).dropna()
    dataset["text"] = dataset["text"].astype(str).str.strip()
    dataset["label"] = dataset["label"].astype(int)

    invalid_labels = sorted(set(dataset["label"]) - {0, 1})
    if invalid_labels:
        raise ValueError(f"Labels must be binary; found {invalid_labels}.")

    return (
        dataset[dataset["text"].ne("")]
        .drop_duplicates(subset=["text"])
        .reset_index(drop=True)
    )
