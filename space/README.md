---
title: EgoSieve
emoji: 🎞️
colorFrom: gray
colorTo: orange
sdk: static
app_file: index.html
models:
  - itspublu/EgoSieve-S
---

# EgoSieve manifest inspector

Inspect a versioned EgoSieve JSONL manifest alongside an optional local video.
The browser renders readiness curves, compiled segments, issue evidence, and
recorded run metadata. Segment selections seek the local video, and the loaded
manifest or a compact segment export can be downloaded for reuse.

This is a dependency-free Static Space. Files are processed only in browser
memory and are never uploaded. Model inference runs through the EgoSieve CLI;
the Space is deliberately an inspector, not an inference service.
