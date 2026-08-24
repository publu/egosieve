# EgoSieve product brief

## One-line promise

EgoSieve finds the short spans of first-person video that are worth labeling,
reconstructing, or training on, and explains what is wrong with the rest.

## The problem

Raw egocentric collections are long, redundant, and uneven. A typical data
pipeline pays its largest costs—video-language inference, depth, pose, or 4D
reconstruction—before it knows whether a window contains a visible, stable,
useful interaction. Dataset teams therefore duplicate private threshold code,
lose provenance between stages, and cannot compare curation policies.

EgoSieve is a cheap first pass. It turns a video into scored windows and a
versioned manifest. It does not claim to recover robot actions from monocular
RGB and it does not replace a pose tracker, labeler, or policy.

## Users

- Robot-learning teams choosing which human demonstrations to reconstruct.
- Dataset maintainers auditing first-person video before publication.
- Researchers comparing data-selection strategies for VLA pretraining.
- Annotation teams routing uncertain windows to people instead of silently
  dropping them.

## Release contract

Given an MP4, MOV, or MKV, a checkpoint produces:

1. `KEEP`, `REVIEW`, or `REJECT` probabilities for each time window.
2. Independent probabilities for eight observable issue families:
   `no_hands`, `low_hand_activity`, `hand_occlusion`, `camera_instability`,
   `blur`, `exposure`, `scene_cut`, and `duplicate_frames`.
3. Start/end boundary probabilities for interaction-aware segmentation.
4. A normalized clip embedding for deduplication, clustering, and retrieval.
5. An uncertainty score and the evidence needed to reproduce the decision.

The compiler converts overlapping windows into stable segments and writes
JSONL. Optional exporters may add Parquet, a Hugging Face Dataset, thumbnails,
or source-preserving clips; JSONL remains the compatibility contract.

## Non-goals for v0.1

- Metric 3D hand pose from a single RGB view.
- Camera calibration recovery.
- Natural-language action labeling.
- Robot control or automatic hardware execution.
- A universal definition of video quality.

These are downstream or provider-specific jobs. Keeping them out of the core
makes the first pass fast, commercially usable, and easy to validate.

## What makes the model useful

The model is deliberately multi-task. A single visual pass feeds a temporal
encoder and four heads: readiness, observable issues, boundaries, and a
retrieval embedding. Training can use strong labels, weak programmatic labels,
or a mixture. Every target has its own validity mask, so missing or uncertain
annotations are not treated as negatives.

At serving time, the compiler uses hysteresis rather than a single threshold:
high-confidence useful windows open a segment, a lower threshold keeps it
open, short gaps merge, and ambiguous spans route to `REVIEW`. This avoids
chattering boundaries and makes the policy inspectable.

## Acceptance bar for a public checkpoint

A checkpoint is release-ready only when its card includes:

- Grouped train/validation/test splits with no source-video leakage.
- Per-class precision, recall, F1, and confusion matrix.
- Macro AUROC and average precision for issue heads.
- Boundary F1 at declared temporal tolerances.
- Expected calibration error and selective-risk curves.
- Throughput on CPU and one common GPU.
- Results broken out by capture source or embodiment.
- Human-grounded readiness and boundary rows with an explicit annotation
  guide, plus separately identified controlled-corruption issue rows.
- Known failure cases, especially unusual viewpoints and assistive devices.

No metric may be filled with a result from the training labels themselves.
The repository rubric is [ANNOTATION.md](ANNOTATION.md); a release may extend
it, but must version and publish every change used for evaluation.
`egosieve validate-release` enforces this bundle structurally, loads the local
custom AutoClasses, inspects the safetensors file, and runs a finite forward
pass. Because that step executes custom model code, it is intended only for a
trusted release candidate produced by the project build.

The validator also requires a hashed training report, leakage-audited split
membership, raw test predictions with task-level validity and label provenance,
and a complete artifact hash manifest. It recomputes the published readiness,
issue, boundary, calibration, and readiness-by-source results from those
predictions. See
[RELEASE.md](RELEASE.md) for the versioned evidence schemas and linkage rules.

## HF release bundle

- `EgoSieve-S`: Apache-2.0 Transformers checkpoint in safetensors.
- `EgoSieve-Eval`: redistributable annotation index and mixed-evidence split.
- `EgoSieve`: Gradio Space for upload, timeline review, and JSONL download.
- This repository: training, evaluation, inference, and export code.
