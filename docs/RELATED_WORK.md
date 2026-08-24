# Related work and product boundary

EgoSieve is a focused curation layer, not an alternative robot policy or hand
reconstruction system.

## Why this layer exists

[LeRobotDataset v3](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)
provides a strong Hub-native storage contract for synchronized video, Parquet
signals, and episode metadata. It begins after useful episodes and actions
already exist. EgoSieve operates one step earlier: it finds useful spans in
ordinary raw video and records why a span was kept, reviewed, or rejected.

[EgoSmith](https://github.com/egosteer/egosmith) combines video filtering,
hand and camera reconstruction, metric depth, language annotation, and
dataset export. That is a powerful full pipeline, but its default geometry
path includes separately licensed research assets and significant GPU work.
EgoSieve turns the cheap readiness decision into a reusable, benchmarkable
model with no mandatory geometry stack.

The [EgoSteer paper](https://arxiv.org/html/2607.09701) reports that its
unfiltered-data ablation materially underperformed the curated-data setting.
The result is specific to that experiment, but supports a broader and
testable premise: selection quality deserves its own model, metrics, and
release artifact.

Other pipelines demonstrate demand for accessible conversion:

- [MINT](https://github.com/wuji-technology/wuji-ego-mint) covers camera and
  hand inference, benchmarking, and LeRobot export.
- [Ego2Robot](https://github.com/msunbot/ego2robot) demonstrates filtering,
  clustering, labeling, and LeRobot conversion on a compact example.
- [pi0-ego-pipeline](https://huggingface.co/ShubhamRasal/pi0-ego-pipeline)
  shows a lightweight MediaPipe-to-LeRobot workflow.
- [EgoLoc](https://github.com/IRMVLab/EgoLoc) targets contact and separation
  localization with a richer multi-model stack.

EgoSieve overlaps only at the front door. Its stable public contract is a
calibrated time-window manifest. Pose, depth, language, contact, and robot
retargeting remain optional downstream providers.

## Evaluation sources

Datasets such as [HOT3D](https://github.com/facebookresearch/hot3d),
[EgoDex](https://github.com/apple/ml-egodex), and
[STERA-10M](https://huggingface.co/datasets/fpvlabs/stera-10m) may provide
useful supervision or evaluation slices. Each has its own access, consent,
and licensing terms. The presence of a public loader does not by itself grant
permission to redistribute clips, annotations, or trained weights.

## Deliberate boundary

EgoSieve does not turn uncertain monocular estimates into purported robot
actions. Any future geometry provider must disclose coordinate frame, units,
metric-scale source, model revision, per-channel confidence, and validity
masks. Exporting a LeRobot-shaped file is not evidence that its action values
are physically meaningful.
