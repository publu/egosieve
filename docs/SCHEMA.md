# Manifest schema

EgoSieve writes newline-delimited JSON with schema
`egosieve.video-compilation`, version `1`. The first row is a manifest header;
following rows describe sampled windows and compiled segments. Times are
seconds on the decoded presentation timeline, not frame numbers.

## Manifest row

```json
{
  "record_type": "manifest",
  "schema": "egosieve.video-compilation",
  "schema_version": 1,
  "generator": "egosieve 0.1.0",
  "source": {
    "source_path": "video.mp4",
    "source_sha256": "…",
    "duration_s": 91.42,
    "width": 1920,
    "height": 1080,
    "display_width": 1920,
    "display_height": 1080,
    "rotation_degrees": 0,
    "fps": 29.97
  },
  "model": {
    "id": "org/EgoSieve-S",
    "revision": "resolved-hub-commit",
    "requested_revision": "release-tag-or-commit",
    "processor_revision": "resolved-hub-commit"
  },
  "counts": {
    "windows": 44,
    "segments": 9,
    "window_decisions": {"KEEP": 28, "REVIEW": 10, "REJECT": 6},
    "segment_decisions": {"KEEP": 5, "REVIEW": 4, "REJECT": 0}
  },
  "sampling": {
    "window_duration_s": 6.0,
    "stride_s": 2.0,
    "frames_per_window": 12,
    "window_count": 44,
    "unique_sample_count": 276
  },
  "policy": {
    "enter_threshold": 0.7,
    "exit_threshold": 0.5,
    "uncertainty_threshold": 0.5
  }
}
```

## Window row

```json
{
  "record_type": "window",
  "schema_version": 1,
  "window_index": 7,
  "start_s": 12.0,
  "end_s": 18.0,
  "source_start_s": 12.0,
  "source_end_s": 18.0,
  "timestamps_s": [12.25, 12.75, 13.25],
  "decision": "KEEP",
  "readiness": {"KEEP": 0.91, "REVIEW": 0.07, "REJECT": 0.02},
  "score": 0.91,
  "uncertainty": 0.24,
  "routing_uncertainty": 0.24,
  "issues": {"acting_hand_not_visible": 0.31, "blur": 0.04},
  "reported_issues": [],
  "boundary": [
    {"timestamp_s": 12.25, "start": 0.83, "end": 0.04}
  ]
}
```

`uncertainty` is normalized readiness entropy. `routing_uncertainty` also
accounts for a confident `REVIEW` class and is the value used by the compiler.
The full issue dictionary is retained for recalibration; `reported_issues`
applies the threshold recorded in the header. Embeddings are optional because
JSON is a poor bulk vector store.

`counts.window_decisions` counts the model argmax for every sampled window.
`counts.segment_decisions` counts emitted compiler segments separately. The
latter can contain zero `REJECT` entries when discard segments are omitted;
the former still reports rejected windows.

Boundary probabilities are diagnostic in schema version 1. They do not refine
segment endpoints. This keeps endpoint behavior reproducible until a boundary
calibration and conflict-resolution policy is evaluated and versioned.

## Segment row

```json
{
  "record_type": "segment",
  "schema_version": 1,
  "segment_index": 2,
  "segment_id": "sha256:…",
  "start_s": 12.0,
  "end_s": 20.0,
  "route": "keep",
  "decision": "KEEP",
  "reason": "stable",
  "window_indices": [6, 7, 8, 9],
  "mean_score": 0.89,
  "readiness": {"KEEP": 0.89, "REVIEW": 0.08, "REJECT": 0.03},
  "issues": {"acting_hand_not_visible": 0.31},
  "reported_issues": []
}
```

Compiler routes are lowercase `keep`, `review`, and `discard`. The uppercase
`decision` maps those routes to the model-card vocabulary (`DISCARD` maps to
`REJECT`). Before optional discard filtering, segments form an exclusive
partition of the routed time union. Overlap precedence is
`review > keep > discard`; after discard filtering, gaps may remain but emitted
segments never overlap.

## Stability and provenance rules

- The source hash identifies source bytes. A changed file is a new source.
- Segment identity includes source hash, model id and resolved revision,
  timestamps, and route.
- Hub model and processor assets are loaded from the same resolved revision.
- Sampling and routing thresholds are recorded in the header.
- A decoder failure is an exception; it never becomes a `REJECT` prediction.
- Windows with insufficient evidence become `REVIEW`, not silent drops.
- Paths may be made relative or redacted; hashes and timestamps remain.
