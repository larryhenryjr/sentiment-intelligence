"""Tests for sentiment dataset preparation."""

from pathlib import Path

import pandas as pd
import pytest

from src.data import SOURCE_FILES, load_dataset


def write_sources(directory: Path, label: int = 1) -> None:
    """Write one representative row for every expected source."""
    for index, filename in enumerate(SOURCE_FILES.values()):
        content = f"Review {index}\t{label}\n"
        (directory / filename).write_text(content, encoding="utf-8")


def test_load_dataset_adds_source_and_binary_label(tmp_path: Path) -> None:
    write_sources(tmp_path)

    result = load_dataset(tmp_path)

    assert list(result.columns) == ["text", "label", "source"]
    assert len(result) == 3
    assert set(result["source"]) == {"amazon", "imdb", "yelp"}
    assert pd.api.types.is_integer_dtype(result["label"])


def test_load_dataset_rejects_nonbinary_labels(tmp_path: Path) -> None:
    write_sources(tmp_path, label=2)

    with pytest.raises(ValueError, match="Labels must be binary"):
        load_dataset(tmp_path)


def test_load_dataset_reports_missing_source(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="See data/README.md"):
        load_dataset(tmp_path)
