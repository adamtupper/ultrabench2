"""Shared helpers for locating, partitioning and caching datasets."""

import csv
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TypeVar

MANIFEST_DIR = Path(__file__).resolve().parent / "manifests"
"""The directory holding the tracked split manifests."""

T = TypeVar("T")


def load_split_manifest(name: str, split: str) -> set[str]:
    """Read the names of the examples belonging to one split.

    The raw and local datasets are preprocessed on the user's machine,
    so their splits are read from a tracked manifest rather than
    re-derived from a seeded splitter, which would make the result
    depend on the scikit-learn version and on the raw data being
    byte-identical.

    Args:
        name: The registry name of the dataset.
        split: The split to read, either "train" or "test".

    Returns:
        The names of the examples in the split.

    Raises:
        FileNotFoundError: If the dataset has no manifest for the split.
    """
    path = MANIFEST_DIR / name / f"{split}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"No {split} manifest for {name}: {path}")

    with path.open(newline="") as f:
        return {row["filename"] for row in csv.DictReader(f)}


def partition_by_manifest(
    items: Sequence[T], name: str, key: Callable[[T], str]
) -> tuple[list[T], list[T]]:
    """Split a sequence into its train and test parts by manifest.

    An item in neither manifest is dropped, which is how subsampling and
    de-duplication are expressed. A manifest entry with no matching item
    is an error: it means the raw data no longer holds the examples the
    published splits were built from.

    Args:
        items: The items to partition.
        name: The registry name of the dataset.
        key: Returns the manifest name identifying an item. Datasets
            listed at video level match on the video stem rather than
            the record's filename.

    Returns:
        The train items and the test items, each in the order given.

    Raises:
        ValueError: If a manifest entry has no matching item.
    """
    train_names = load_split_manifest(name, "train")
    test_names = load_split_manifest(name, "test")

    train = [item for item in items if key(item) in train_names]
    test = [item for item in items if key(item) in test_names]

    found = {key(item) for item in items}
    missing = sorted((train_names | test_names) - found)
    if missing:
        raise ValueError(
            f"The raw data for {name} is missing {len(missing)} example(s) "
            f"listed in its split manifests: {', '.join(missing[:10])}"
            f"{', ...' if len(missing) > 10 else ''}"
        )

    return train, test
