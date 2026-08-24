from __future__ import annotations

import pytest

from egosieve.compiler.segments import (
    DISCARD,
    KEEP,
    REVIEW,
    Segment,
    SegmentCompilerConfig,
    WindowScore,
    compile_segments,
    hysteresis_mask,
    merge_segments,
)


def test_hysteresis_holds_until_exit_threshold() -> None:
    assert hysteresis_mask(
        [0.1, 0.71, 0.6, 0.5, 0.49, 0.69, 0.7],
        enter_threshold=0.7,
        exit_threshold=0.5,
    ) == (False, True, True, True, False, False, True)


def test_compile_routes_score_band_and_explicit_uncertainty_to_review() -> None:
    windows = [
        WindowScore(0, 0, 2, 0.1),
        WindowScore(1, 2, 4, 0.8),
        WindowScore(2, 4, 6, 0.6),
        WindowScore(3, 6, 8, 0.55),
        WindowScore(4, 8, 10, 0.4),
        WindowScore(5, 10, 12, 0.9, uncertainty=0.8),
        WindowScore(6, 12, 14, 0.9),
    ]
    segments = compile_segments(
        windows,
        config=SegmentCompilerConfig(merge_gap_s=0, min_duration_s=0),
    )

    assert [(segment.route, segment.start_s, segment.end_s) for segment in segments] == [
        (KEEP, 2, 4),
        (REVIEW, 4, 8),
        (REVIEW, 10, 12),
        (KEEP, 12, 14),
    ]
    assert segments[1].window_indices == (2, 3)


def test_merge_gap_reconnects_stable_segments() -> None:
    segments = compile_segments(
        [(0, 1, 0.9), (1, 2, 0.1), (2.5, 3.5, 0.9)],
        config=SegmentCompilerConfig(
            merge_gap_s=1.5,
            min_duration_s=0,
            uncertainty_threshold=None,
        ),
    )

    assert len(segments) == 1
    assert segments[0].route == KEEP
    assert (segments[0].start_s, segments[0].end_s) == (0, 3.5)
    assert segments[0].window_indices == (0, 2)


def test_overlapping_routes_form_exclusive_partition_with_review_precedence() -> None:
    segments = compile_segments(
        [
            WindowScore(0, 0, 6, 0.9),
            WindowScore(1, 2, 8, 0.9, uncertainty=0.9),
            WindowScore(2, 4, 10, 0.9),
        ],
        config=SegmentCompilerConfig(merge_gap_s=0, min_duration_s=0),
    )

    assert [(segment.route, segment.start_s, segment.end_s) for segment in segments] == [
        (KEEP, 0, 2),
        (REVIEW, 2, 8),
        (KEEP, 8, 10),
    ]
    assert [segment.window_indices for segment in segments] == [(0,), (1,), (2,)]
    assert all(
        left.end_s <= right.start_s for left, right in zip(segments, segments[1:], strict=False)
    )


def test_short_keep_segments_follow_min_duration_route() -> None:
    windows = [(0, 0.5, 0.9)]
    assert (
        compile_segments(
            windows,
            config=SegmentCompilerConfig(min_duration_s=1, short_segment_route=DISCARD),
        )
        == ()
    )
    review = compile_segments(
        windows,
        config=SegmentCompilerConfig(min_duration_s=1, short_segment_route=REVIEW),
    )
    assert len(review) == 1
    assert review[0].route == REVIEW
    assert review[0].reason == "short"


def test_merge_segments_uses_window_weighted_score_mean() -> None:
    left = Segment(0, 2, KEEP, 0.8, 0.7, 0.9, (0, 1))
    right = Segment(2.25, 3, KEEP, 0.5, 0.5, 0.5, (2,))

    merged = merge_segments([left, right], max_gap_s=0.25)

    assert len(merged) == 1
    assert merged[0].mean_score == pytest.approx(0.7)
    assert merged[0].window_indices == (0, 1, 2)


def test_invalid_threshold_order_is_rejected() -> None:
    with pytest.raises(ValueError, match="exit <= enter"):
        SegmentCompilerConfig(enter_threshold=0.4, exit_threshold=0.6)
