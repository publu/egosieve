---
title: EgoSieve
emoji: 🎞️
colorFrom: gray
colorTo: orange
sdk: gradio
sdk_version: 5.47.2
python_version: 3.12
app_file: app.py
suggested_hardware: cpu-basic
models:
  - itspublu/EgoSieve-S
---

# EgoSieve

Upload a first-person video, inspect its manipulation-readiness timeline, and
download the versioned evidence manifest. The Space does not retain uploads
beyond the runtime's bounded temporary workspace. Set `EGOSIEVE_MODEL_ID` and
`EGOSIEVE_MODEL_REVISION` to the released model repository and immutable commit
before deployment; the app intentionally refuses an unresolved revision.
