# Architecture

```text
video ──▶ metadata + timestamp sampler ──▶ frame tensor [T,C,H,W]
                                                  │
                                        shared vision encoder
                                                  │
                                          temporal encoder
                                    ┌───────┬──────┼───────┐
                                    ▼       ▼      ▼       ▼
                                readiness issues boundary embedding
                                    │       │      │       │
                                    └───────┴──┬───┴───────┘
                                               ▼
                                  calibrated window predictions
                                               │
                              hysteresis + gap merge + uncertainty gate
                                               │
                              JSONL / review artifacts / optional clips
```

## Components

### Timestamp sampler

The sampler probes the presentation timeline and requests frames by time. It
does not assume constant frame rate or derive timestamps by dividing a frame
index by nominal FPS. Rotation and display dimensions are preserved in source
metadata. External processes are invoked with argument arrays and bounded
timeouts. Unique timestamp requests are extracted in bounded batches, so a
long clip does not launch one ffmpeg process per sampled frame.

### Vision and time

The small release uses an Apache-2.0 DINOv2 vision encoder. Per-frame pooled
features receive learned position encodings and pass through compact, gated
temporal mixers with dilated depthwise convolutions and channel mixing.
Padding frames are masked before and after temporal operations and in pooling.
The temporal module, rather than the image encoder alone, owns decisions that
require motion or repetition evidence.

### Heads

- Readiness: mutually exclusive `KEEP`, `REVIEW`, `REJECT` logits.
- Issues: independent logits; several issues may coexist.
- Boundary: start/end logits for every valid sample time.
- Embedding: L2-normalized vector for similarity operations.

Losses are computed only where their target mask is valid. Future-feature
prediction is a research direction, not part of the version 0.1 architecture
or inference contract.

### Compiler

The compiler is deterministic. It applies two-threshold hysteresis, merges
same-route windows across the declared short-gap tolerance, and resolves every
overlap into an exclusive timeline partition. Conflict precedence is
`review > keep > discard`: ambiguous evidence remains reviewable, while an
established keep run can bridge a contradictory discard window within the gap
tolerance. Minimum-duration routing applies to the resulting exclusive
fragments. Discard intervals may be omitted from output, leaving gaps but never
overlaps. The compiler never changes source media unless clip export is
explicitly requested.

Boundary-head probabilities are diagnostic evidence in version 0.1. They are
written at sampled timestamps for review and evaluation, but do not move
segment endpoints. Endpoint refinement needs a separately calibrated policy;
until then, segment bounds come only from routed window intervals.

## Extension points

Geometry, hand-pose, depth, IMU, and language systems are optional providers.
They may enrich a segment row without changing its identity or the core model
inputs. Provider outputs must include their own model revision, coordinate
frame, units, confidence, and validity mask.
