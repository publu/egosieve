---
license: apache-2.0
library_name: transformers
pipeline_tag: video-classification
base_model: facebook/dinov2-small
tags:
  - robotics
  - egocentric-video
  - video-quality
  - dataset-curation
  - physical-ai
---

# EgoSieve-S

EgoSieve-S finds manipulation-ready spans in first-person video before a data
pipeline spends compute on language annotation, pose, depth, or 4D
reconstruction. It returns readiness probabilities, observable failure
signals, interaction boundary scores, and a retrieval embedding.

This model is for dataset curation. It is not a robot policy and must not be
used to control hardware.

## Model details

- **Architecture:** DINOv2-small frame encoder plus compact gated temporal
  mixers and independent readiness, issue, boundary, and embedding heads
  (about 23.5M parameters in this configuration).
- **Input:** 12 RGB frames, center-cropped to 224 × 224, plus a frame-validity
  mask. The bundled processor handles normalization; timestamp-based video
  sampling is provided by the EgoSieve package.
- **Readiness classes:** `KEEP`, `REVIEW`, `REJECT`.
- **Issue labels:** `no_hands`, `low_hand_activity`, `hand_occlusion`,
  `camera_instability`, `blur`, `exposure`, `scene_cut`, `duplicate_frames`.
- **License:** Apache 2.0. The base encoder is also Apache 2.0.

## Usage

```bash
pip install egosieve
egosieve scan input.mp4 --model {{MODEL_ID}} --revision {{MODEL_COMMIT}} \
  --output input.egosieve.jsonl
```

```python
from transformers import AutoModelForVideoClassification, AutoProcessor

processor = AutoProcessor.from_pretrained(
    "{{MODEL_ID}}", revision="{{MODEL_COMMIT}}", trust_remote_code=True
)
model = AutoModelForVideoClassification.from_pretrained(
    "{{MODEL_ID}}", revision="{{MODEL_COMMIT}}", trust_remote_code=True
).eval()

# `frames` is a timestamp-selected list of PIL images or HWC arrays.
inputs = processor(frames, return_tensors="pt")
outputs = model(**inputs)
print(outputs.readiness_logits.softmax(-1))
print(outputs.issue_logits.sigmoid())
```

## Training data

{{TRAINING_DATA}}

Splits are grouped by original capture unit so near-identical windows from one
source cannot cross train, validation, and test boundaries. Missing weak-label
targets use explicit masks and are not interpreted as negatives.

## Evaluation

{{EVALUATION}}

Metrics use a source-grouped, mixed-evidence test split. Human-grounded rows
measure readiness, calibration, and boundaries; issue labels may combine
audited human annotations with programmatic controlled corruptions, with counts
reported separately by provenance. Boundary F1 uses the temporal tolerance
stated in the evaluation artifact. Calibration is reported because `REVIEW`
routing depends on confidence, not just top-1 accuracy.

## Intended use

- Rank raw egocentric windows before expensive processing.
- Propose stable interaction segments for human review.
- Detect common visual failure modes.
- Create embeddings for deduplication, retrieval, and diversity sampling.

## Limitations

- Readiness is downstream-task dependent; thresholds should be calibrated on
  your capture setup.
- Performance may vary with camera placement, unusual viewpoints, gloves,
  assistive devices, skin tone, lighting, and domains absent from evaluation.
- The model observes RGB only. It cannot measure force, slip, metric depth,
  success, intent, or physical safety.
- Boundary scores are proposals, not ground truth action annotations.
- High confidence does not make a clip legally or ethically publishable.

## Privacy and safety

First-person recordings may contain faces, screens, documents, homes, and
bystanders. EgoSieve does not perform consent checks, redaction, or identity
removal. Run a separate privacy review before sharing media or derived data.
Predictions must not directly trigger robot actions.

## Reproducibility

The repository includes the training-data schema, grouped split logic,
evaluation code, checkpoint export validation, and a Space that exposes the
same manifest contract as the CLI. This model repository also includes hashed
training, split, raw-prediction, and release-evidence artifacts. Published
metrics are recomputed from linked predictions carrying task-level label
validity and provenance by `egosieve validate-release`.
