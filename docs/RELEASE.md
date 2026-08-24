# Release evidence contract

`egosieve validate-release` treats a model directory as a linked evidence
bundle, not as a collection of claims. A release candidate must contain the
standard Transformers files, custom model code, and the four evidence files
below. Every JSON document uses the exact `v1` schema identifier shown here.

## Evidence files

### `training_report.json`

Schema: `egosieve.training-report/v1`.

Required fields are a non-empty run ID and source commit, `completed: true`, a
positive optimizer-step count, exact train/validation/test example counts, and
the SHA-256 digests of `model.safetensors` and `splits.json`. `backbone` names
both the upstream model ID and immutable revision.

Because the public model exposes retrieval embeddings, the report must also
record a nonzero retrieval objective:

```json
{
  "retrieval_objective": {
    "name": "supervised-contrastive",
    "weight": 0.1,
    "positive_pairs": 4812
  }
}
```

Removing the training marker without this evidence does not make a checkpoint
releasable.

### `splits.json`

Schema: `egosieve.splits/v1`.

`examples` contains one row per evaluated training example with non-empty
`id`, `group_id`, `source`, and a `split` of `train`, `validation`, or `test`.
IDs are unique, every split is populated, and a group may occur in only one
split. The reported split counts must match this file exactly.

### `test_predictions.json`

Schema: `egosieve.test-predictions/v1`.

The file links the checkpoint and split hashes, declares the fixed label order,
and contains one prediction row for every test-split ID—no subsampling. Each
row carries:

- matching ID, group, and source provenance;
- a `label_provenance` object with separate `readiness`, `issues`, and
  `boundaries` kinds;
- `readiness_valid`, a nullable readiness target, and three probabilities
  summing to one;
- eight issue targets, validity flags, and probabilities;
- start/end reference and predicted timestamps with independent validity
  flags.

The allowed provenance kinds are `human`, `human-derived`,
`programmatic-controlled-corruption`, and `unlabeled`. A valid readiness label
must be human or human-derived. Those rows require `review_count >= 2` and a
non-empty `rubric_version`; issue-only controlled-corruption rows do not.
Programmatic issue rows use `readiness_valid: false`, a null readiness target,
and unlabeled boundary provenance.

```json
{
  "label_provenance": {
    "readiness": {"kind": "unlabeled"},
    "issues": {"kind": "programmatic-controlled-corruption"},
    "boundaries": {"kind": "unlabeled"}
  },
  "readiness_valid": false,
  "readiness_target": null
}
```

Every released readiness class must have positive support among valid
readiness rows. Each issue requires at least one valid positive and negative;
those labels may be human, human-derived, or controlled-corruption evidence,
and at least one issue-valid row must be a controlled corruption. Both boundary
types require valid human-grounded test references. Unknown labels remain
masked rather than becoming negatives.

### `evidence.json`

Schema: `egosieve.release-evidence/v1`.

`artifacts` maps every required release file other than `evidence.json` itself
to its lowercase SHA-256 digest. This includes the card, configuration,
processor, checkpoint, metrics, training report, splits, raw predictions, and
custom Python files. The training report, predictions, and metrics also contain
cross-links to the artifacts they describe.

## Metric recomputation

`metrics.json` uses schema `egosieve.release-metrics/v1`. It is validated
structurally and then checked against the raw test predictions. The validator
recomputes:

- readiness confusion matrix, per-class precision/recall/F1/support, and macro
  F1;
- per-issue and macro AUROC/average precision plus positive/negative support;
- start/end boundary matching and micro F1 at the declared tolerance;
- top-label ECE using the declared bin count and the complete selective-risk
  curve, using only rows where `readiness_valid` is true;
- total, readiness, issue, and boundary evidence-row counts;
- readiness macro F1 by source, using only valid readiness rows and omitting
  issue-only sources.

`metrics.evaluation` declares `readiness_human_grounded: true` and
`issues_controlled_corruptions: true`, and its
`issue_examples_by_provenance` object reports row counts for `human`,
`human-derived`, and `programmatic-controlled-corruption` issue evidence. The
former blanket `human_reviewed` field is rejected because it mischaracterizes
a mixed-evidence test set.

Numeric values must agree within floating-point tolerance. `boundaries.f1` is
the recomputed micro F1. Throughput remains a separately measured artifact;
the model card must state its hardware and measurement boundary.

## Validation order

Run the validator only on a trusted build output because Transformers custom
code is loaded for the final forward-pass check:

```bash
egosieve validate-release artifacts/EgoSieve-S
```

A successful result proves that the bundle is complete, internally linked,
group-disjoint, numerically self-consistent, and runnable. It does not prove
that annotations were collected ethically or that the selected data license
permits publication; those remain human review responsibilities documented in
the model and dataset cards.
