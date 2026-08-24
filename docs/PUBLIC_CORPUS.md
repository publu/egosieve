# Public corpus acquisition

EgoSieve does not download training media during installation, import,
testing, or training. Public data acquisition is a separate opt-in command
with four required choices:

1. a reviewed publisher source;
2. a full immutable repository commit;
3. every exact repository file path;
4. an exact license acknowledgement.

Wildcards, directory selections, branch names, tags, duplicate paths, and
overwriting an existing output directory are rejected. A successful download
writes `corpus-manifest.json` with the publisher URL, repository id and
revision where applicable, license id and URL, attribution, exact selected
paths and download URLs, byte sizes, and locally computed SHA-256 digests.

This mechanism records reproducibility and the publisher's declared terms. It
does not replace a legal, privacy, consent, or publication review.

## Ego-Tactile Manipulation

The first reviewed source profile is the publisher's
[Ego-Tactile Manipulation repository](https://huggingface.co/datasets/OpenGraphLabs-Research/ego-tactile-manipulation).
The publisher declares the data `CC-BY-4.0`; the profile records the
[CC BY 4.0 legal code](https://creativecommons.org/licenses/by/4.0/legalcode)
and the publisher attribution. CC BY 4.0 requires attribution and an
indication of modifications when licensed material is shared. It does not by
itself grant privacy, publicity, trademark, or patent rights.

The adapter was checked against the official LeRobot v3.0 files at commit
`90c4e304b8e3a9578d5bb938992206358db1a660`:

- `meta/info.json` declares 30 fps video under
  `observation.images.ego`, frame/episode/task indexes, and six string
  annotation columns;
- `meta/episodes/chunk-000/file-000.parquet` maps an episode to its shared
  data Parquet and packed MP4, including the episode's timestamp range inside
  that MP4;
- `data/chunk-000/file-000.parquet` contains `timestamp`, `frame_index`,
  `episode_index`, `task_index`, `annotation.l0_action`,
  `annotation.l1_subaction`, `annotation.anchor`,
  `annotation.grasp_type`, `annotation.hand`, and
  `annotation.target_object`;
- videos follow
  `videos/observation.images.ego/chunk-{chunk}/file-{file}.mp4`.

The official episode table shows that episodes may share one packed video.
The generated `group_id` therefore identifies the packed MP4, not just the
episode, so adjacent material cannot leak across generated data splits.

### Acquire an explicit episode dependency set

For episode 0 at the inspected revision, select these four exact files. The
metadata and data Parquet files are shared; the selected MP4 also contains
adjacent packed material. The current selection is roughly 194 MB, rather than
the full 6.28 GB dataset.

```bash
egosieve corpus fetch \
  --source ego-tactile \
  --revision 90c4e304b8e3a9578d5bb938992206358db1a660 \
  --file meta/info.json \
  --file meta/episodes/chunk-000/file-000.parquet \
  --file data/chunk-000/file-000.parquet \
  --file videos/observation.images.ego/chunk-000/file-000.mp4 \
  --output corpora/ego-tactile-episode-0 \
  --accept-license CC-BY-4.0
```

The output directory must not already exist. The command does not expand the
selection based on metadata, patterns, or a dataset loader.

Verify every recorded size and digest later with:

```bash
egosieve corpus verify corpora/ego-tactile-episode-0
```

### Adapt publisher annotations

Install the optional Parquet dependency and explicitly choose each episode to
adapt:

```bash
python -m pip install -e '.[hub]'

egosieve corpus adapt-ego-tactile \
  corpora/ego-tactile-episode-0 \
  --episode 0 \
  --output corpora/ego-tactile-episode-0.proxy-boundaries.jsonl
```

The adapter first re-hashes every acquired file. It then joins the inspected
episode and frame schemas, converts episode-relative timestamps into packed
MP4 timestamps, groups contiguous source annotations, and emits byte-stable
`egosieve.training/v1` JSONL.

The publisher states that action boundaries come from physical contact and
grip force and are not human ground truth. The adapter preserves that
distinction:

- `label_source.kind` is `proxy` and `human_reviewed` is `false` on every
  generated window;
- source annotation text, anchor, hand, object, task index, and frame range are
  retained as metadata;
- proxy start/end boundaries are valid only as proxy boundary targets;
- `readiness_valid` is always `false`;
- no readiness class or issue label is inferred, including from
  `annotation.hand == "none"`.

Use these records for boundary pretraining or representation learning, with a
declared weak-label policy. Release evaluation and KEEP/REVIEW/REJECT
calibration still require separately collected human review.

## HoloAssist

The HoloAssist profile deliberately records two origins rather than treating a
mirror as the publisher:

- annotations come from the publisher's versioned
  [`data-annotation-trainval-v1_1.json`](https://hl2data.z5.web.core.windows.net/holoassist-data-release/data-annotation-trainval-v1_1.json),
  whose schema is documented by the
  [HoloAssist project](https://holoassist.github.io/data_links/README.html);
- selected RGB videos are transported from
  [`lmms-lab/EgoIT-99K`](https://huggingface.co/datasets/lmms-lab/EgoIT-99K)
  at exact commit `a57f1f2078a7b01ea87014050fdb3afe169e54f1`.

The publisher releases HoloAssist under
[`CDLA-Permissive-2.0`](https://cdla.dev/permissive-2-0/). The pinned mirror's
dataset card does not declare a license. The manifest therefore calls the
mirror `media-transport-only`, leaves its license declaration URL null, and
sets `publisher_byte_equivalence_verified` to false. It does not use the
mirror as evidence for data rights or annotations.

### Acquire exact annotation and video files

The annotation filename and every video must be selected explicitly. For one
recording:

```bash
egosieve corpus fetch \
  --source holoassist \
  --revision a57f1f2078a7b01ea87014050fdb3afe169e54f1 \
  --file data-annotation-trainval-v1_1.json \
  --file HoloAssist/video/R015-7July-Switch.mp4 \
  --output corpora/holoassist-r015-switch \
  --accept-license CDLA-Permissive-2.0

egosieve corpus verify corpora/holoassist-r015-switch
```

The official annotation URL is not commit-addressed. Its release filename and
the locally computed SHA-256 are both recorded. Video URLs contain the exact
reviewed mirror commit; other commits are rejected until the profile is
re-reviewed. A HoloAssist acquisition must include the annotation JSON and
at least one exact `HoloAssist/video/<recording>.mp4` path; dataset directories,
globs, and the mirror's derived Parquet files are rejected.

### Adapt audited action intervals into readiness proxies

```bash
egosieve corpus adapt-holoassist \
  corpora/holoassist-r015-switch \
  --video-id R015-7July-Switch \
  --window 6 \
  --stride 3 \
  --keep-occupancy-threshold 0.5 \
  --max-keep-per-video 64 \
  --max-review-per-video 64 \
  --max-reject-per-video 64 \
  --output corpora/holoassist-r015-switch.readiness-proxies.jsonl
```

The adapter uses the scanner's deterministic sliding-window planner, including
one end-anchored tail window when necessary. For each window it measures the
temporal union of all overlapping publisher fine-action intervals:

- `KEEP`: occupancy is at least the configured threshold;
- `REVIEW`: occupancy is positive but below the threshold;
- `REJECT`: occupancy is zero.

The three per-video caps are applied independently and select candidates evenly
across time. Positive caps preserve every available class; increase them to
retain all candidates. With the 43 locally selected sessions, the uncapped
6-second/3-second/0.5 rubric yields 3,100 KEEP, 625 REVIEW, and 291 REJECT
candidates, with all three classes present in every session.

These are deterministic readiness proxies, not fresh EgoSieve judgments.
HoloAssist action correctness—including wrong-action categories—is preserved
verbatim in `source_annotations` and never determines readiness. A wrong but
clearly visible manipulation can therefore be KEEP. Zero occupancy means only
"no publisher fine-action annotation in this window"; it is not a human
background judgment.

The [published audit process](https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_HoloAssist_an_Egocentric_Human_Interaction_Dataset_for_Interactive_AI_Assistants_ICCV_2023_paper.pdf)
describes an original professional annotation pass with self-review followed
by an independent reviewer audit. Generated windows carry `review_count: 2`
with `review_count_scope: source_action_intervals`; they also state
`human_reviewed: false` for the derived readiness label. The later targeted
text-field review is recorded in provenance but is not counted as another
reviewer because the paper does not state its reviewer cardinality.

Official action timestamps and attributes are retained. Boundary targets use
the first official action start and last official action end occurring inside
the window, each with its own validity mask. If those candidates belong to
different actions and form an inverted pair, the pair is masked for the v1
training contract while the exact candidates remain in source metadata.
Technical issues are never inferred: `issues` and `issue_valid` are absent.
Every record groups by its original HoloAssist video.

## Adding another source

A source should not be added merely because a mirror carries a permissive
metadata tag. A new profile needs an official publisher URL, a data-specific
license declaration and license text, stable direct file URLs, attribution,
and a schema inspection. Source-specific proxy adapters must state what was
measured, what was inferred, and which EgoSieve targets remain unknown.

For a composite source such as HoloAssist, keep publisher annotations, mirror
transport, license declarations, and byte-integrity claims separate in both
the profile and every derived record.
