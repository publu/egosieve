# EgoSieve

**Find the useful seconds before you spend GPU-hours reconstructing them.**

EgoSieve is a compact model and dataset compiler for first-person video. It
scans long recordings, ranks short manipulation windows, explains observable
quality failures, routes uncertain spans to review, and writes a reproducible
manifest for downstream labeling, pose reconstruction, or robot-learning
pipelines.

```text
raw MP4 ──▶ timestamp sampler ──▶ EgoSieve-S ──▶ stable segments
                                                       │
                                   JSONL / review timeline / optional clips
```

EgoSieve is intentionally not a robot policy or a monocular 3D-action claim.
It solves the expensive step immediately before those systems: deciding which
video windows are actually usable.

> **v0.1.0 is published.** The checkpoint, metadata-only evaluation set, raw
> held-out predictions, grouped splits, and machine-checked release evidence
> are public and versioned.

[Model](https://huggingface.co/itspublu/EgoSieve-S) ·
[Evaluation set](https://huggingface.co/datasets/itspublu/EgoSieve-Eval) ·
[Manifest inspector](https://huggingface.co/spaces/itspublu/EgoSieve)

## What one pass returns

For each window:

- `KEEP`, `REVIEW`, and `REJECT` probabilities;
- seven independent signals: acting-hand visibility, low hand activity,
  camera instability, blur, exposure, scene cuts, and duplicate frames;
- interaction start/end likelihoods;
- a normalized embedding for clustering and deduplication;
- timestamps, source hash, sampling configuration, and uncertainty.

Overlapping windows become stable, non-chattering segments through a
deterministic compiler. Low-confidence input is reviewable rather than
silently discarded.

## Quick start

Requirements: Python 3.10+ and `ffmpeg`/`ffprobe` on `PATH`.

```bash
python -m venv .venv
.venv/bin/pip install -e .

egosieve inspect recording.mp4
egosieve plan recording.mp4 --window 6 --stride 2 --frames 12

egosieve scan recording.mp4 \
  --model itspublu/EgoSieve-S \
  --revision 40f8549166c9c64f205e63c0836cd88f4208d112 \
  --include-embeddings \
  --output recording.egosieve.jsonl
```

Python:

```python
from egosieve import EgoSieveConfig, EgoSieveModel

model = EgoSieveModel.from_pretrained(
    "itspublu/EgoSieve-S",
    revision="40f8549166c9c64f205e63c0836cd88f4208d112",
    trust_remote_code=True,
)
```

Checkpoint loading uses `safetensors`. The model repository carries its
processor configuration and custom AutoClass mapping. For a Hub
model, pass an immutable commit with `--revision`; the scanner loads the
processor from the model's resolved commit and records that identity.

To initialize a training checkpoint from a pinned DINOv2 backbone:

```python
from transformers import Dinov2Model

from egosieve.initialization import initialize_from_backbone, save_training_seed

backbone = Dinov2Model.from_pretrained(
    "facebook/dinov2-small",
    revision="BACKBONE_COMMIT_SHA",
)
model = initialize_from_backbone(backbone, num_frames=12)
save_training_seed(model, "artifacts/egosieve-seed")
```

The resulting directory contains `UNTRAINED_HEADS`; the release validator
refuses it until the heads are trained and the mixed-evidence metrics bundle is
complete.

## Manifest

```json
{"record_type":"manifest","schema":"egosieve.video-compilation","schema_version":1,"source":{"source_sha256":"…","duration_s":91.42},"model":{"id":"itspublu/EgoSieve-S","revision":"…"}}
{"record_type":"segment","schema_version":1,"start_s":12.4,"end_s":19.1,"route":"keep","decision":"KEEP","readiness":{"KEEP":0.91,"REVIEW":0.07,"REJECT":0.02},"issues":{"acting_hand_not_visible":0.31}}
```

Times live on the decoded presentation timeline. Variable-frame-rate sources
are sampled by timestamp, not by pretending nominal FPS is exact. See
[the schema](docs/SCHEMA.md) for stability and provenance rules.

## Design principles

1. **Cheap before expensive.** Reject or review weak windows before VLM,
   depth, pose, or 4D reconstruction.
2. **Uncertainty is output.** Missing evidence and ambiguous predictions are
   explicit; they are not coerced into a clean label.
3. **One visual pass, several jobs.** Readiness, issue detection, boundaries,
   and retrieval share features.
4. **Source truth survives.** Hashes, timestamps, model revision, and compiler
   parameters travel with every decision.
5. **Geometry is a provider.** Optional pose/depth/IMU integrations must name
   coordinate frames, units, scale source, validity, and confidence.

## Model shape

`EgoSieve-S` uses an Apache-2.0 DINOv2-small image encoder and a lightweight,
mask-aware stack of gated temporal mixers (about 23.9M parameters with the
default configuration). It accepts
`[batch, time, channels, height, width]` plus a frame-validity mask. The four
output heads are readiness, multi-label issues, per-frame boundaries, and an
L2-normalized clip embedding. Each training target has a separate validity
mask.

The model is a native `PreTrainedModel`/`PretrainedConfig` implementation so
checkpoints can use `save_pretrained`, `from_pretrained`, AutoClass metadata,
and Hugging Face Hub versioning without a bespoke loader.

## Training and evaluation

Training data follows a source-grouped, mask-aware JSONL contract described in
[docs/TRAINING_DATA.md](docs/TRAINING_DATA.md). A checkpoint is not considered
release-ready without grouped splits, human-grounded readiness and boundary
evaluation, controlled-corruption issue evaluation, calibration, readiness
breakdowns by source, temporal-boundary metrics, and throughput numbers. The
full acceptance bar is in [docs/PRODUCT.md](docs/PRODUCT.md).

On the 196-window held-out split, v0.1.0 reaches 0.674 readiness macro-F1,
0.741 issue macro-AUROC, 0.722 issue macro average precision, and 0.103
boundary micro-F1 at 0.3 seconds. Readiness and boundaries are HoloAssist
fine-action-derived proxies, five issue tasks use controlled corruptions, and
the acting-hand signal follows the source acting-hand modifier. These are
mixed-evidence results, not an independently annotated in-the-wild benchmark;
the [model card](https://huggingface.co/itspublu/EgoSieve-S) gives per-task
support and limitations.

Licensed public media is never fetched implicitly. The opt-in
[public-corpus workflow](docs/PUBLIC_CORPUS.md) requires exact files, an
immutable source revision, and explicit license acknowledgement, then records
file hashes and attribution. Its Ego-Tactile adapter marks contact-derived
boundaries as proxies and does not manufacture human readiness or quality
labels. Its HoloAssist profile keeps the official annotation release separate
from a commit-pinned media mirror; the adapter emits disclosed fixed-window
occupancy proxies while preserving audited action intervals and leaving every
technical issue unknown.
The [observable-issue augmentation builder](docs/AUGMENTATION.md) emits
deterministic MP4 variants and positive-only targets with exact tool, command,
hash, license, split, and label-source provenance.
The machine-checked artifact and raw-prediction contract is documented in
[docs/RELEASE.md](docs/RELEASE.md).

## Repository map

```text
src/egosieve/modeling/   Transformers model and configuration
src/egosieve/video/      metadata probing and timestamp frame sampling
src/egosieve/compiler/   windows → stable segments → manifests
src/egosieve/corpus/     opt-in acquisition, integrity, and source adapters
src/egosieve/training/   masked multi-task data and training utilities
hub/model/               model-card and export assets
space/                   local-only Hugging Face manifest inspector
tests/                   unit and end-to-end contract tests
```

## Scope and safety

EgoSieve scores visual suitability for dataset work. It does not establish
that an action is physically safe, successful, legal, or appropriate for a
robot to execute. Predictions should not directly control hardware. Performance
will vary with camera placement, protective equipment, skin tone, lighting,
assistive devices, and domains absent from evaluation.

## License

Apache 2.0. Model and dataset releases document their own training-data and
artifact terms in their cards.
