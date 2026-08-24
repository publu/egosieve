---
license: {{DATASET_LICENSE}}
task_categories:
  - video-classification
tags:
  - egocentric-video
  - robotics
  - video-quality
  - temporal-segmentation
pretty_name: EgoSieve-Eval
---

# EgoSieve-Eval

EgoSieve-Eval is the human-reviewed evaluation set for manipulation readiness
and observable quality failures in first-person video. It exists to evaluate
data selection and calibration, not action recognition or robot control.

## Fields

| Field | Type | Meaning |
|---|---|---|
| `source_id` | string | Stable, non-secret source identifier |
| `video` | Video | Short licensed clip or gated source reference |
| `start_s`, `end_s` | float | Original presentation timestamps |
| `readiness` | class | `KEEP`, `REVIEW`, or `REJECT` |
| `issues` | sequence[class] | Observable issue labels |
| `interaction_start_s`, `interaction_end_s` | float, nullable | Reviewed boundaries |
| `annotator_count` | int | Number of independent reviews |
| `agreement` | float | Declared agreement statistic |
| `capture_group` | string | Leakage-safe split group |

## Annotation process

{{ANNOTATION_PROCESS}}

Labels follow the versioned EgoSieve annotation rubric shipped with the source
repository. Any deviations or task-specific readiness criteria are listed
above rather than silently folded into the class names.

## Splits and leakage controls

{{SPLITS}}

## Licensing, consent, and privacy

{{DATA_GOVERNANCE}}

## Known limitations

The evaluation set measures visual usefulness under its declared curation
rubric. It does not establish whether an action succeeded, was safe, or can be
transferred to a particular robot embodiment.
