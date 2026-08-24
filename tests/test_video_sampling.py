from __future__ import annotations

import pytest

from egosieve.video.sampling import (
    gather_window_values,
    plan_frame_samples,
    sample_once,
    sample_timestamps,
    sliding_windows,
)


def test_sliding_windows_adds_one_end_anchored_tail() -> None:
    assert sliding_windows(10, 4, 3) == ((0.0, 4.0), (3.0, 7.0), (6.0, 10.0))
    assert sliding_windows(11, 4, 3) == (
        (0.0, 4.0),
        (3.0, 7.0),
        (6.0, 10.0),
        (7.0, 11.0),
    )
    assert sliding_windows(2, 4, 3) == ((0.0, 2.0),)


def test_center_sampling_never_requests_interval_end() -> None:
    assert sample_timestamps(0, 4, 4) == (0.5, 1.5, 2.5, 3.5)
    assert sample_timestamps(10, 14, 1) == (12.0,)


def test_plan_has_fixed_window_counts_and_deduplicates_overlaps() -> None:
    plan = plan_frame_samples(
        8,
        window_duration_s=4,
        stride_s=1,
        frames_per_window=4,
        start_time_s=2.5,
    )

    assert len(plan.windows) == 5
    assert all(len(window.sample_indices) == 4 for window in plan.windows)
    assert sum(len(window.sample_indices) for window in plan.windows) == 20
    assert len(plan.samples) == 8
    assert plan.timestamps_s == (0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5)
    assert plan.source_timestamps_s == tuple(value + 2.5 for value in plan.timestamps_s)
    assert plan.windows[0].sample_indices == (0, 1, 2, 3)
    assert plan.windows[1].sample_indices == (1, 2, 3, 4)


def test_sample_once_calls_decoder_once_per_unique_timestamp() -> None:
    calls: list[float] = []

    def decode(timestamp: float) -> str:
        calls.append(timestamp)
        return f"frame@{timestamp}"

    values = sample_once([0.5, 1.5, 0.5, 1.5, 2.5], decode)

    assert calls == [0.5, 1.5, 2.5]
    assert values[0] == values[2]
    assert values[1] == values[3]


def test_gather_window_values_rejects_incomplete_unique_decode() -> None:
    plan = plan_frame_samples(4, window_duration_s=2, stride_s=1, frames_per_window=2)
    values = tuple(f"frame-{sample.index}" for sample in plan.samples)
    gathered = gather_window_values(plan, values)
    assert all(len(window) == 2 for window in gathered)
    with pytest.raises(ValueError, match="one item"):
        gather_window_values(plan, values[:-1])
