---
license: {{DATASET_LICENSE}}
task_categories:
  - video-classification
tags:
  - egocentric-video
  - robotics
  - video-quality
  - temporal-segmentation
  - weak-supervision
  - metadata-only
pretty_name: EgoSieve-Eval
---

# EgoSieve-Eval

EgoSieve-Eval is a metadata-only annotation index and group-disjoint,
mixed-evidence split for manipulation readiness, temporal boundaries, and
observable video-quality signals in first-person manipulation video. It exists
to evaluate data selection and calibration, not action recognition or robot
control.

This is not a wholly human-reviewed benchmark. The repository contains no
video, audio, or extracted frame bytes. A `video` value is a reference to
separately acquired source or generated media and may not resolve outside the
release build environment. Access to this index does not grant access to the
referenced media.

## Fields

| Field | Type | Meaning |
|---|---|---|
| `id` | string | Stable record identifier |
| `group_id` | string | Leakage-safe original-video group |
| `source` | string | Source or derived-evidence category |
| `video` | string | Reference to separately managed media; no media payload is included |
| `license` | string | Declared identifier for the referenced source media |
| `windows` | sequence[object] | Timestamped examples and task-specific targets |
| `windows.start_s`, `windows.end_s` | float | Presentation timestamps in the referenced video |
| `windows.readiness`, `windows.readiness_valid` | class, bool | `KEEP`, `REVIEW`, or `REJECT`, plus an explicit validity mask |
| `windows.issues`, `windows.issue_valid` | mapping | Per-issue targets and per-issue validity masks; missing is unknown, not negative |
| `windows.boundaries_s`, `windows.boundary_valid` | mapping | Start/end targets and independent validity masks |
| `windows.label_provenance` | mapping | Separate readiness, issue, and boundary evidence kinds |
| `windows.review_count`, `windows.review_count_scope` | int, string | Review count and the evidence to which it applies |
| `windows.rubric_version` | string | Version of the rule used to derive the target |

## Evidence and annotation process

The readiness rows are deterministic scanner-window proxies derived from the
temporal union of HoloAssist publisher fine-action intervals. With the released
rubric, a six-second window is `KEEP` at or above 0.5 fine-action occupancy,
`REVIEW` when occupancy is positive but below 0.5, and `REJECT` at zero
occupancy. Windows use a three-second stride and the scanner's deterministic
end-anchored tail policy. These classes measure interval occupancy; they do not
say that an action was correct, successful, safe, or useful for a particular
robot. HoloAssist action-correctness values are retained only as source
metadata and do not determine readiness. Readiness rows identify
`egosieve.holoassist-readiness-proxy/v1` as their rubric and retain
`HoloAssist fine-action annotations v1_1` as the source rubric; task-level
`label_provenance` distinguishes human-derived, programmatic, and unlabeled
evidence.

HoloAssist's published process describes an original professional-annotator
self-review followed by an independent reviewer audit. Accordingly,
`review_count: 2` is scoped as `source_action_intervals`. It does not mean that
two people reviewed an EgoSieve scanner window or its derived readiness class;
the derived readiness label is explicitly marked `human_reviewed: false`.
Boundary evidence is human-derived from the publisher interval timestamps.
All overlapping source annotations and boundary candidates are retained, and
a boundary pair that cannot be represented as one ordered pair is masked.

The HoloAssist adapter itself emits no technical-issue targets. Issue evidence
in the mixed corpus comes from separate, explicitly identified routes:

- narrow human-derived proxies from the publisher's acting-hand visibility
  modifier and from fine-action occupancy, each with its scope recorded; and
- deterministic controlled corruptions and their matched unmodified
  references for `blur`, `exposure`, `camera_instability`, `duplicate_frames`,
  and `scene_cut`.

Those issue rows are not HoloAssist technical-issue annotations. Controlled
corruptions are programmatic evidence rather than natural failures or human
reviews, and their readiness and boundary targets are masked. The acting-hand
modifier supervises only `acting_hand_not_visible`: the exact value `hand not
visible` is a positive proxy and explicit `left hand` or `right hand` values
are negative proxies. It never claims that every hand is absent from the
frame. Task-level validity and provenance fields must be consulted
independently; evidence for one task does not validate another.

The unmodified control rows are matched references, not human audits that the
source video naturally lacks blur, exposure problems, camera instability,
duplicate frames, or a scene cut. Metrics using those rows measure
injected-corruption-versus-unmodified discrimination and must not be reported
as performance on audited natural issue absence.

Release-specific construction and review details follow.

{{ANNOTATION_PROCESS}}

Labels follow the versioned EgoSieve annotation rubric shipped with the source
repository. Any deviations, thresholds, proxy scopes, or task-specific
criteria are listed above rather than silently folded into the class names.

## Splits and leakage controls

Splits are assigned by original HoloAssist video. Every derived row and
controlled variant retains its source video's `group_id`, so related windows
and transformations cannot cross split boundaries. The recommended seed is
the earliest seed with at least three examples of every readiness class, both
polarities of every issue, and both boundary types in each of train,
validation, and test.

{{SPLITS}}

## Licensing, consent, and privacy

This metadata-only release does not redistribute HoloAssist recordings or
controlled-corruption video bytes. Users must acquire any referenced source
media separately, comply with its applicable terms, and conduct their own
consent and privacy review. Source references can point to first-person video
containing faces, screens, documents, voices, or bystanders.

{{DATA_GOVERNANCE}}

## Known limitations

Readiness is an occupancy proxy, not a direct human judgment of visual
usefulness. In particular, `REJECT` means zero publisher fine-action occupancy
within the scanner window; it does not establish that the video is unusable.
Visibility and low-activity targets are narrowly scoped proxies, while
controlled corruptions cover only their declared transformations and need not
match naturally occurring failures. `acting_hand_not_visible` concerns only
the acting hand identified by the publisher modifier; it does not claim that
every hand is absent. Unmodified references are weak controls rather than
reviewed natural negatives. The set does not establish whether an action
succeeded, was correct or safe, or can be transferred to a particular robot
embodiment.
