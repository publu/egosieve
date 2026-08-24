# Third-party components

EgoSieve's source code is Apache 2.0. A complete distribution also depends on
the terms of its selected model and system dependencies.

| Component | Purpose | Upstream terms |
|---|---|---|
| [DINOv2-small](https://huggingface.co/facebook/dinov2-small) | Default frame encoder | Apache 2.0 |
| [Transformers](https://github.com/huggingface/transformers) | Model/config/processor interfaces | Apache 2.0 |
| [PyTorch](https://github.com/pytorch/pytorch) | Tensor runtime | BSD-style |
| [FFmpeg](https://ffmpeg.org/legal.html) | Optional video probe/decode/clip export | Build-dependent LGPL/GPL terms |

Public checkpoints and Spaces should pin exact revisions and preserve the
corresponding notices. Optional providers are not covered by this table and
must document their own weights, assets, and data licenses.

## Optional public data

No third-party training media or annotation is bundled with EgoSieve. The
opt-in corpus CLI currently contains a provenance profile for
[Ego-Tactile Manipulation](https://huggingface.co/datasets/OpenGraphLabs-Research/ego-tactile-manipulation),
which its publisher declares under CC BY 4.0. Each local acquisition records
the exact revision, selected files, SHA-256 digests, license URL, and required
attribution.

The CLI also profiles the publisher's
[HoloAssist](https://holoassist.github.io/) annotation release under
CDLA-Permissive-2.0. Selected videos are transported from
[`lmms-lab/EgoIT-99K`](https://huggingface.co/datasets/lmms-lab/EgoIT-99K)
at an exact commit. That mirror's pinned dataset card does not declare a
license, so manifests identify it as transport only, do not claim publisher
byte equivalence, and keep the original publisher's terms separate. See the
[public-corpus guide](docs/PUBLIC_CORPUS.md) before use or redistribution.
