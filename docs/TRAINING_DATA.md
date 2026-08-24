# Training data contract

The training-data utilities consume one JSON object per source video. Media
stays in its original location; a trainer can combine this contract with the
provided target builder and collator.

```json
{
  "schema": "egosieve.training/v1",
  "id": "episode-001",
  "group_id": "capture-session-019",
  "video": "videos/episode-001.mp4",
  "license": "declared-by-dataset-owner",
  "windows": [
    {
      "start_s": 8.0,
      "end_s": 14.0,
      "readiness": "KEEP",
      "readiness_valid": true,
      "issues": {"blur": false, "acting_hand_not_visible": true},
      "issue_valid": {"blur": true, "acting_hand_not_visible": true},
      "boundaries_s": {"start": 8.4, "end": 13.7},
      "boundary_valid": true,
      "annotator": "human"
    }
  ]
}
```

`group_id` is mandatory for split generation. It should identify the largest
unit that could leak nearly identical content—for example a continuous
recording session, operator/task repetition block, or original source video.

## Label sources

Human labels, programmatic rules, and teacher models may coexist, but their
source must be recorded. Weak targets should receive lower sample weights or
be used for pretraining followed by calibration on human labels. Missing issue
keys are unknown, not false.

Rows intended for mixed-evidence release evaluation use task-level provenance:
`label_provenance.readiness`, `.issues`, and `.boundaries` each contain their
own `kind`. The release vocabulary is `human`, `human-derived`,
`programmatic-controlled-corruption`, and `unlabeled`; provenance for one task
does not establish any other task.

The [observable-issue augmentation builder](AUGMENTATION.md) can create
deterministic positive proxies for six visual failure modes from explicitly
allowed source licenses. The transform named `hand_occlusion` draws a synthetic
opaque overlay and maps it to `acting_hand_not_visible` only as
programmatic-controlled-corruption evidence. It is not a naturally observed or
human-inherited visibility label. In contrast, the transform named
`acting_hand_not_visible` draws no occluder: it only makes a faithful temporal
and codec re-encode of a source window that already has a valid,
human-grounded positive for `acting_hand_not_visible`. The analogous
`low_hand_activity` transform also preserves an existing valid positive.
Although the synthetic overlay and faithful re-encode share a downstream issue
name, their provenance remains distinct. Outputs preserve source group/split
identity and mask every target that the transformation or inherited annotation
does not establish.

The optional [public-corpus adapter](PUBLIC_CORPUS.md) follows this rule
literally. Ego-Tactile contact/grip-force action spans are emitted with
`label_source.kind: "proxy"` and `human_reviewed: false`. They provide proxy
boundary targets only: readiness remains invalid and all issue targets remain
unknown. In particular, a source value such as `annotation.hand: "none"` is
not silently converted into an EgoSieve `acting_hand_not_visible` label.

The HoloAssist adapter emits fixed-window readiness proxies from the temporal
union of independently audited publisher fine-action intervals. It records
`programmatic_readiness_proxy`, `human_reviewed: false`, the versioned rubric,
and `review_count_scope: source_action_intervals`. Action correctness is
retained as source metadata but is not a readiness target. Windows with zero
fine-action occupancy are derived REJECT proxies, not human background labels.
No HoloAssist window receives a technical-issue target. Records use
`source: "HoloAssist"` and group by original video.

The v1 issue vocabulary is fixed to `acting_hand_not_visible`,
`low_hand_activity`, `camera_instability`, `blur`, `exposure`, `scene_cut`,
and `duplicate_frames`. The parser rejects any other issue or validity-mask key so
a spelling mistake cannot be silently dropped during vectorization. A dataset
with a deliberately different vocabulary must pass that vocabulary explicitly
as `issue_names` to both parsing and target building.

### Visibility-label migration

Development annotations using separate `no_hands` and `hand_occlusion` fields
must be migrated before parsing. A valid positive from either old field may
support a positive `acting_hand_not_visible` label only when it refers to the
task-relevant acting hand. A negative is established only when both old fields
were valid negatives; otherwise the new target remains unknown. Conflicts and
ambiguous references require review.

The classifier changed from eight issue logits to seven and the public order
changed. Earlier seed checkpoints and trained checkpoints are shape- and
semantics-incompatible: regenerate the training seed and retrain rather than
renaming configuration metadata or reusing classifier rows.

## Sampled-frame boundary targets

`boundaries_s` values are absolute timestamps in the same video coordinate
system as `start_s`, `end_s`, and the exact sampled frame timestamps supplied
to the target builder. They are converted to model targets with this declared
policy:

1. For each valid `start` or `end` annotation, select the sampled frame with
   the smallest absolute timestamp error. An exact tie selects the earlier
   frame (the lower sample index).
2. Assign a positive only if that error is less than or equal to the configured
   `boundary_tolerance_s`.
3. When assigned, the selected frame is `1`, every other real sampled frame in
   that boundary channel is `0`, and the channel mask is true at all real
   frames. Padding is always masked.
4. If the annotation is missing, marked invalid, or has no sampled frame within
   tolerance, the whole channel mask is false. It is never converted to an
   all-negative example.

The lower-level `encode_windows` API exposes absolute annotations as
`boundary_times_s` and `boundary_time_mask`; those `[windows, 2]` arrays are
intermediate metadata and must not be passed to the model as boundary labels.

`build_sampled_targets` returns `[windows, frames, 2]` `boundary_labels` and
`boundary_label_mask` arrays plus the exact padded timestamp matrix and frame
mask. `TrainingCollator` applies the same conversion while padding
`frame_embeddings` or `pixel_values`:

```python
from egosieve.training import TrainingCollator

collate = TrainingCollator(boundary_tolerance_s=0.25)
batch = collate(
    [
        {
            "window": record.windows[0],
            "sampled_timestamps_s": [8.25, 8.75, 9.25, 9.75],
            "frame_embeddings": embeddings,
        }
    ]
)
```

The collator is NumPy-only. Convert its arrays to tensors in the training
framework after collation.

## Privacy and licensing

The public model card must name every training dataset, its license, and the
filtering applied. Do not publish example frames, annotations, or derived
checkpoints until the data owner has confirmed that the applicable terms
permit those uses. Faces, screens, documents, and bystanders require an
explicit review policy.
