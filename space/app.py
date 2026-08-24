from __future__ import annotations

import os
import shutil
import tempfile
from collections import deque
from functools import lru_cache
from pathlib import Path

import gradio as gr
import plotly.graph_objects as go
from transformers import AutoModelForVideoClassification, AutoProcessor

from egosieve.compiler import read_manifest
from egosieve.inference import ScanConfig, scan_video
from egosieve.video import probe_video

MODEL_ID = os.environ.get("EGOSIEVE_MODEL_ID", "itspublu/EgoSieve-S")
MODEL_REVISION = os.environ.get("EGOSIEVE_MODEL_REVISION", "set-EGOSIEVE_MODEL_REVISION")
MAX_BYTES = 150 * 1024 * 1024
MAX_DURATION_S = 60
MAX_DISPLAY_PIXELS = 1920 * 1080
MAX_RETAINED_RUNS = 3

INK = "#171714"
PAPER = "#ebe8df"
ORANGE = "#ff5c1a"
ACID = "#c7f000"
MUTED = "#8f9088"
DEFAULT_PROGRESS = gr.Progress()
RUN_DIRECTORIES: deque[Path] = deque()


def _new_run_directory() -> Path:
    path = Path(tempfile.mkdtemp(prefix="egosieve-space-"))
    RUN_DIRECTORIES.append(path)
    while len(RUN_DIRECTORIES) > MAX_RETAINED_RUNS:
        stale = RUN_DIRECTORIES.popleft()
        shutil.rmtree(stale, ignore_errors=True)
    return path


@lru_cache(maxsize=1)
def load_model():
    if MODEL_REVISION == "set-EGOSIEVE_MODEL_REVISION":
        raise RuntimeError("Set EGOSIEVE_MODEL_REVISION to the released model commit.")
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=True,
    )
    model = AutoModelForVideoClassification.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=True,
    ).eval()
    return model, processor


def _timeline(records: tuple[dict, ...]) -> go.Figure:
    windows = [record for record in records if record.get("record_type") == "window"]
    segments = [record for record in records if record.get("record_type") == "segment"]
    midpoint = [(row["start_s"] + row["end_s"]) / 2 for row in windows]

    figure = go.Figure()
    for label, color in (("KEEP", ACID), ("REVIEW", ORANGE), ("REJECT", MUTED)):
        figure.add_trace(
            go.Scatter(
                x=midpoint,
                y=[row["readiness"][label] for row in windows],
                mode="lines+markers" if label == "KEEP" else "lines",
                name=label,
                line={"color": color, "width": 3 if label == "KEEP" else 1.5},
                marker={"size": 5},
                hovertemplate=f"{label} %{{y:.1%}}<br>%{{x:.2f}} s<extra></extra>",
            )
        )
    route_color = {"keep": "rgba(199,240,0,.12)", "review": "rgba(255,92,26,.14)"}
    for segment in segments:
        if segment["route"] in route_color:
            figure.add_vrect(
                x0=segment["start_s"],
                x1=segment["end_s"],
                fillcolor=route_color[segment["route"]],
                line_width=0,
                layer="below",
            )
    figure.update_layout(
        paper_bgcolor=INK,
        plot_bgcolor=INK,
        font={"family": "DM Mono, monospace", "color": PAPER, "size": 12},
        height=390,
        margin={"l": 52, "r": 20, "t": 35, "b": 50},
        legend={"orientation": "h", "x": 0, "y": 1.12},
        hovermode="x unified",
        xaxis={
            "title": "PRESENTATION TIME / SECONDS",
            "gridcolor": "rgba(235,232,223,.09)",
            "zeroline": False,
        },
        yaxis={
            "title": "READINESS",
            "range": [0, 1],
            "tickformat": ".0%",
            "gridcolor": "rgba(235,232,223,.09)",
            "zeroline": False,
        },
    )
    return figure


def _segment_rows(records: tuple[dict, ...]) -> list[list]:
    rows = []
    for record in records:
        if record.get("record_type") != "segment":
            continue
        issues = sorted(record.get("issues", {}).items(), key=lambda item: item[1], reverse=True)[
            :3
        ]
        rows.append(
            [
                record["segment_index"],
                record["decision"],
                round(record["start_s"], 2),
                round(record["end_s"], 2),
                round(record["end_s"] - record["start_s"], 2),
                ", ".join(f"{name} {score:.0%}" for name, score in issues),
            ]
        )
    return rows


def analyze(video_path: str | None, progress=DEFAULT_PROGRESS):
    if not video_path:
        raise gr.Error("Choose a video first.")
    source = Path(video_path)
    if source.stat().st_size > MAX_BYTES:
        raise gr.Error("This demo accepts videos up to 150 MB.")
    metadata = probe_video(source, calculate_hash=False)
    if metadata.duration_s > MAX_DURATION_S:
        raise gr.Error("This CPU demo accepts videos up to 60 seconds.")
    if metadata.display_width * metadata.display_height > MAX_DISPLAY_PIXELS:
        raise gr.Error("This demo accepts display resolutions up to 1920 × 1080.")

    progress(0.05, desc="Loading checkpoint")
    model, processor = load_model()
    output_dir = _new_run_directory()
    manifest = output_dir / f"{source.stem}.egosieve.jsonl"
    cache = output_dir / "frames"

    progress(0.12, desc="Sampling presentation timeline")
    result = scan_video(
        source,
        model_id=MODEL_ID,
        output_path=manifest,
        revision=MODEL_REVISION,
        cache_dir=cache,
        model=model,
        processor=processor,
        config=ScanConfig(device="auto"),
    )
    progress(0.92, desc="Compiling stable segments")
    records = read_manifest(manifest)
    counts = result.decision_counts
    kept_seconds = sum(
        segment.end_s - segment.start_s for segment in result.segments if segment.route == "keep"
    )
    review_seconds = sum(
        segment.end_s - segment.start_s for segment in result.segments if segment.route == "review"
    )
    summary = (
        f"### Scan complete\n\n"
        f"`{result.metadata.duration_s:.1f}s` source · `{len(result.windows)}` windows · "
        f"`{kept_seconds:.1f}s` kept · `{review_seconds:.1f}s` review\n\n"
        f"**{counts['KEEP']} keep** / **{counts['REVIEW']} review** / "
        f"**{counts['REJECT']} reject** windows"
    )
    progress(1.0, desc="Ready")
    return summary, _timeline(records), _segment_rows(records), str(manifest)


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Archivo+Black&family=DM+Mono:wght@300;400;500&display=swap');
:root { --ink:#171714; --paper:#ebe8df; --orange:#ff5c1a; --acid:#c7f000; }
.gradio-container {
  background:
    linear-gradient(rgba(235,232,223,.035) 1px, transparent 1px),
    linear-gradient(90deg, rgba(235,232,223,.035) 1px, transparent 1px),
    var(--ink) !important;
  background-size: 28px 28px !important;
  color: var(--paper) !important;
  font-family: 'DM Mono', monospace !important;
}
.eg-shell { max-width: 1240px; margin: 0 auto; }
.eg-kicker { color:var(--orange); letter-spacing:.22em; font-size:12px; font-weight:500; }
.eg-title {
  font-family:'Archivo Black', sans-serif; font-size:clamp(64px,11vw,148px);
  line-height:.78; letter-spacing:-.07em; color:var(--paper); margin:22px 0 30px;
}
.eg-title span { color:var(--acid); -webkit-text-stroke:1px var(--acid); }
.eg-deck { max-width:720px; font-size:16px; line-height:1.65; color:#b9b8b0; }
.eg-rule { height:1px; background:linear-gradient(90deg,var(--orange),transparent); margin:34px 0; }
.eg-index { color:var(--acid); font-size:11px; letter-spacing:.18em; }
.gr-button-primary {
  background:var(--orange) !important; border:0 !important; color:var(--ink) !important;
  border-radius:0 !important; font-family:'Archivo Black' !important; text-transform:uppercase;
  box-shadow:6px 6px 0 var(--acid) !important; transition:.16s transform,.16s box-shadow !important;
}
.gr-button-primary:hover { transform:translate(3px,3px); box-shadow:3px 3px 0 var(--acid) !important; }
.block, .panel { border-radius:0 !important; }
footer { display:none !important; }
"""


with gr.Blocks(css=CSS, theme=gr.themes.Base()) as demo, gr.Column(elem_classes="eg-shell"):
    gr.HTML(
        "<div class='eg-kicker'>VIDEO READINESS / INSPECTION CONSOLE 01</div>"
        "<div class='eg-title'>EGO<span>/</span>SIEVE</div>"
        "<div class='eg-deck'>Find the seconds worth reconstructing. "
        "Every decision ships with timestamps, uncertainty, and observable failure signals.</div>"
        "<div class='eg-rule'></div>"
    )
    with gr.Row(equal_height=True):
        with gr.Column(scale=5):
            gr.HTML("<div class='eg-index'>01 — SOURCE</div>")
            video = gr.Video(label="FIRST-PERSON VIDEO", sources=["upload"], format="mp4")
        with gr.Column(scale=2):
            gr.HTML("<div class='eg-index'>02 — RUN</div>")
            gr.Markdown(
                f"Checkpoint\n\n`{MODEL_ID}@{MODEL_REVISION}`\n\n"
                "RGB only · timestamp sampling · local evidence manifest"
            )
            run = gr.Button("SIEVE VIDEO", variant="primary", size="lg")
    gr.HTML("<div class='eg-rule'></div><div class='eg-index'>03 — EVIDENCE</div>")
    summary = gr.Markdown("Upload a source to begin.")
    timeline = gr.Plot(label="READINESS TIMELINE")
    table = gr.Dataframe(
        headers=["#", "decision", "start s", "end s", "duration s", "strongest signals"],
        datatype=["number", "str", "number", "number", "number", "str"],
        interactive=False,
        label="COMPILED SEGMENTS",
    )
    manifest_file = gr.File(label="DOWNLOAD EVIDENCE MANIFEST")
    run.click(analyze, inputs=video, outputs=[summary, timeline, table, manifest_file])

demo.queue(max_size=2, default_concurrency_limit=1)


if __name__ == "__main__":
    demo.launch(max_file_size="150mb", max_threads=2)
