"""Deterministic source-grouped dataset splitting."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, TypeVar

T = TypeVar("T")
SPLIT_NAMES = ("train", "validation", "test")


def _group_id(record: Any, group_key: str) -> str:
    if isinstance(record, Mapping):
        try:
            value = record[group_key]
        except KeyError as error:
            raise ValueError(f"record is missing required group key {group_key!r}") from error
    else:
        try:
            value = getattr(record, group_key)
        except AttributeError as error:
            raise ValueError(f"record is missing required group key {group_key!r}") from error
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{group_key} must be a non-empty string; received {value!r}")
    return value


def _normalize_fractions(
    fractions: Mapping[str, float] | Sequence[float] | None,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
) -> tuple[float, float, float]:
    if fractions is None:
        values: tuple[Any, ...] = (train_fraction, validation_fraction, test_fraction)
    elif isinstance(fractions, Mapping):
        unknown = set(fractions) - set(SPLIT_NAMES)
        missing = set(SPLIT_NAMES) - set(fractions)
        if unknown or missing:
            details = []
            if missing:
                details.append(f"missing {sorted(missing)!r}")
            if unknown:
                details.append(f"unknown {sorted(unknown)!r}")
            raise ValueError(
                "fractions must define train/validation/test (" + ", ".join(details) + ")"
            )
        values = tuple(fractions[name] for name in SPLIT_NAMES)
    else:
        values = tuple(fractions)
        if len(values) != len(SPLIT_NAMES):
            raise ValueError("fractions must contain train, validation, and test values")

    normalized: list[float] = []
    for name, value in zip(SPLIT_NAMES, values, strict=True):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} fraction must be numeric; received {value!r}")
        fraction = float(value)
        if not math.isfinite(fraction) or fraction < 0:
            raise ValueError(f"{name} fraction must be finite and non-negative")
        normalized.append(fraction)
    if not math.isclose(sum(normalized), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"split fractions must sum to 1.0; received {sum(normalized):.12g}")
    if not any(normalized):
        raise ValueError("at least one split fraction must be positive")
    return tuple(normalized)  # type: ignore[return-value]


def _allocation(group_count: int, fractions: tuple[float, float, float]) -> list[int]:
    exact = [group_count * fraction for fraction in fractions]
    counts = [math.floor(value) for value in exact]
    remaining = group_count - sum(counts)
    order = sorted(range(len(counts)), key=lambda index: (-(exact[index] - counts[index]), index))
    for index in order[:remaining]:
        counts[index] += 1

    # When possible, make every requested split usable.  This matters for the
    # common small-dataset case (for example three groups at 80/10/10).
    positive = [index for index, fraction in enumerate(fractions) if fraction > 0]
    if group_count >= len(positive):
        for empty in (index for index in positive if counts[index] == 0):
            donors = [index for index in positive if counts[index] > 1]
            donor = max(
                donors, key=lambda index: (counts[index] - exact[index], counts[index], -index)
            )
            counts[donor] -= 1
            counts[empty] += 1
    return counts


def _stable_group_order(groups: Iterable[str], seed: int | str) -> list[str]:
    seed_bytes = str(seed).encode("utf-8")

    def key(group: str) -> tuple[bytes, str]:
        digest = hashlib.sha256(seed_bytes + b"\0" + group.encode("utf-8")).digest()
        return digest, group

    return sorted(set(groups), key=key)


def group_assignments(
    records: Iterable[Any],
    *,
    fractions: Mapping[str, float] | Sequence[float] | None = None,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int | str = 0,
    group_key: str = "group_id",
) -> dict[str, str]:
    """Return a deterministic ``group_id -> split name`` assignment.

    Assignment is based on group count rather than row count.  No record-level
    random operation is performed, so records sharing a group can never leak
    across splits.
    """

    materialized = list(records)
    ratios = _normalize_fractions(
        fractions,
        train_fraction,
        validation_fraction,
        test_fraction,
    )
    groups = _stable_group_order((_group_id(record, group_key) for record in materialized), seed)
    counts = _allocation(len(groups), ratios)
    assignments: dict[str, str] = {}
    offset = 0
    for name, count in zip(SPLIT_NAMES, counts, strict=True):
        for group in groups[offset : offset + count]:
            assignments[group] = name
        offset += count
    return assignments


def grouped_split(
    records: Iterable[T],
    *,
    fractions: Mapping[str, float] | Sequence[float] | None = None,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int | str = 0,
    group_key: str = "group_id",
) -> dict[str, list[T]]:
    """Split records deterministically while keeping each source group intact.

    Input order is preserved *within* each output split but does not affect
    which split receives a group.
    """

    materialized = list(records)
    assignments = group_assignments(
        materialized,
        fractions=fractions,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        seed=seed,
        group_key=group_key,
    )
    result: dict[str, list[T]] = {name: [] for name in SPLIT_NAMES}
    for record in materialized:
        result[assignments[_group_id(record, group_key)]].append(record)
    return result


def grouped_train_val_test_split(
    records: Iterable[T],
    *,
    train_size: float = 0.8,
    validation_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int | str = 0,
    group_key: str = "group_id",
) -> tuple[list[T], list[T], list[T]]:
    """Tuple-returning convenience wrapper around :func:`grouped_split`."""

    result = grouped_split(
        records,
        train_fraction=train_size,
        validation_fraction=validation_size,
        test_fraction=test_size,
        seed=random_state,
        group_key=group_key,
    )
    return result["train"], result["validation"], result["test"]


__all__ = [
    "SPLIT_NAMES",
    "group_assignments",
    "grouped_split",
    "grouped_train_val_test_split",
]
