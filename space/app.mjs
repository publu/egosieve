import {
  READINESS_LABELS,
  aggregateIssues,
  buildSegmentExport,
  decisionFor,
  formatDuration,
  parseManifest,
  summarizeManifest,
} from "./manifest.mjs";

const SVG_NS = "http://www.w3.org/2000/svg";
const MAX_MANIFEST_BYTES = 32 * 1024 * 1024;
const CHART = Object.freeze({ width: 1000, height: 360, left: 58, right: 20, top: 28, bottom: 45 });
const COLORS = Object.freeze({ KEEP: "#d7ff35", REVIEW: "#ff5b26", REJECT: "#898b83" });

const state = {
  parsed: null,
  manifestText: "",
  manifestFileName: "",
  manifestUrl: null,
  videoUrl: null,
  videoFileName: "",
  videoDurationWarning: null,
  selectedSegmentIndex: null,
  routeFilter: "all",
  toastTimer: null,
};

function byId(id) {
  const element = document.getElementById(id);
  if (!element) throw new Error(`Missing required element #${id}`);
  return element;
}

function clear(element) {
  element.replaceChildren();
}

function node(tagName, className = "", text = null) {
  const element = document.createElement(tagName);
  if (className) element.className = className;
  if (text !== null) element.textContent = text;
  return element;
}

function svgNode(tagName, attributes = {}) {
  const element = document.createElementNS(SVG_NS, tagName);
  for (const [name, value] of Object.entries(attributes)) {
    element.setAttribute(name, String(value));
  }
  return element;
}

function safeFileName(value, fallback) {
  const leaf = String(value || fallback).split(/[\\/]/u).pop() || fallback;
  const safe = leaf.replace(/[^a-zA-Z0-9._-]+/gu, "-").replace(/^-+|-+$/gu, "");
  return safe || fallback;
}

function displayPathLeaf(value) {
  return String(value || "unknown source").split(/[\\/]/u).pop() || "unknown source";
}

function formatPercent(value, digits = 0) {
  return typeof value === "number" && Number.isFinite(value)
    ? `${(value * 100).toFixed(digits)}%`
    : "—";
}

function formatTimecode(seconds) {
  if (typeof seconds !== "number" || !Number.isFinite(seconds) || seconds < 0) return "00:00.0";
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const remainder = seconds % 60;
  const base = `${String(minutes).padStart(2, "0")}:${remainder.toFixed(1).padStart(4, "0")}`;
  return hours > 0 ? `${hours}:${base}` : base;
}

function prettyIssue(name) {
  return String(name).replaceAll("_", " ");
}

function routeClass(route) {
  return `route-${route === "discard" ? "discard" : route}`;
}

function decisionRoute(segment) {
  if (["keep", "review", "discard"].includes(segment.route)) return segment.route;
  return { KEEP: "keep", REVIEW: "review", REJECT: "discard" }[decisionFor(segment)] ?? "review";
}

function showNotice(message, isError = false) {
  const notice = byId("notice");
  notice.textContent = message;
  notice.classList.toggle("is-error", isError);
  notice.hidden = false;
}

function hideNotice() {
  byId("notice").hidden = true;
}

function showToast(message) {
  const toast = byId("toast");
  toast.textContent = message;
  toast.hidden = false;
  if (state.toastTimer !== null) window.clearTimeout(state.toastTimer);
  state.toastTimer = window.setTimeout(() => {
    toast.hidden = true;
    state.toastTimer = null;
  }, 2600);
}

async function copyText(text, successMessage) {
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
    } else {
      const helper = document.createElement("textarea");
      helper.value = text;
      helper.setAttribute("readonly", "");
      helper.style.position = "fixed";
      helper.style.opacity = "0";
      document.body.append(helper);
      helper.select();
      const copied = document.execCommand("copy");
      helper.remove();
      if (!copied) throw new Error("copy command was refused");
    }
    showToast(successMessage);
  } catch {
    showToast("Copy was blocked. Select the text and copy it manually.");
  }
}

function revokeManifestUrl() {
  if (state.manifestUrl) URL.revokeObjectURL(state.manifestUrl);
  state.manifestUrl = null;
}

function updateManifestDownload() {
  revokeManifestUrl();
  const link = byId("download-manifest");
  if (!state.manifestText) {
    link.removeAttribute("href");
    return;
  }
  state.manifestUrl = URL.createObjectURL(
    new Blob([state.manifestText], { type: "application/x-ndjson;charset=utf-8" }),
  );
  link.href = state.manifestUrl;
  link.download = safeFileName(state.manifestFileName, "egosieve-manifest.jsonl");
}

function visibleSegments() {
  if (!state.parsed) return [];
  if (state.routeFilter === "all") return [...state.parsed.segments];
  return state.parsed.segments.filter((segment) => decisionRoute(segment) === state.routeFilter);
}

function renderWarnings() {
  const stack = byId("warning-stack");
  clear(stack);
  const warnings = [...(state.parsed?.warnings ?? [])];
  if (state.videoDurationWarning) warnings.push(state.videoDurationWarning);
  if (warnings.length === 0) {
    stack.hidden = true;
    return;
  }
  for (const warning of warnings) stack.append(node("p", "", `⚠ ${warning}`));
  stack.hidden = false;
}

function checkVideoDuration() {
  const video = byId("video");
  const manifestDuration = state.parsed?.header.source.duration_s;
  if (!state.videoUrl || !manifestDuration || !Number.isFinite(video.duration)) {
    state.videoDurationWarning = null;
    return;
  }
  const difference = Math.abs(video.duration - manifestDuration);
  const tolerance = Math.max(0.5, manifestDuration * 0.01);
  state.videoDurationWarning = difference > tolerance
    ? `Local video duration (${formatDuration(video.duration)}) differs from the manifest (${formatDuration(manifestDuration)}). Verify that the files match.`
    : null;
}

function readinessPoint(windowRecord, label) {
  const value = windowRecord.readiness?.[label];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function renderTimeline(parsed) {
  const svg = byId("timeline");
  clear(svg);
  const title = svgNode("title", { id: "timeline-title" });
  title.textContent = "Readiness probabilities over time";
  const description = svgNode("desc", { id: "timeline-description" });
  const evidenceWindows = parsed.windows.filter((windowRecord) =>
    READINESS_LABELS.every((label) => readinessPoint(windowRecord, label) !== null),
  );
  description.textContent = evidenceWindows.length > 0
    ? `${evidenceWindows.length} scored windows across ${formatDuration(parsed.header.source.duration_s)}.`
    : "This manifest contains no window-level readiness probability maps.";
  svg.append(title, description);

  const chartWidth = CHART.width - CHART.left - CHART.right;
  const chartHeight = CHART.height - CHART.top - CHART.bottom;
  const duration = Math.max(parsed.header.source.duration_s, 0.001);
  const x = (seconds) => CHART.left + (Math.max(0, Math.min(duration, seconds)) / duration) * chartWidth;
  const y = (probability) => CHART.top + (1 - probability) * chartHeight;

  for (const probability of [0, 0.25, 0.5, 0.75, 1]) {
    svg.append(
      svgNode("line", {
        x1: CHART.left,
        y1: y(probability),
        x2: CHART.left + chartWidth,
        y2: y(probability),
        stroke: "rgba(239,238,230,.13)",
        "stroke-width": 1,
        "vector-effect": "non-scaling-stroke",
      }),
    );
    const label = svgNode("text", { x: CHART.left - 12, y: y(probability) + 4, "text-anchor": "end" });
    label.textContent = `${Math.round(probability * 100)}`;
    svg.append(label);
  }

  const tickCount = duration > 900 ? 4 : 6;
  for (let index = 0; index <= tickCount; index += 1) {
    const seconds = (duration * index) / tickCount;
    svg.append(
      svgNode("line", {
        x1: x(seconds),
        y1: CHART.top,
        x2: x(seconds),
        y2: CHART.top + chartHeight,
        stroke: "rgba(239,238,230,.075)",
        "stroke-width": 1,
        "vector-effect": "non-scaling-stroke",
      }),
    );
    const label = svgNode("text", {
      x: x(seconds),
      y: CHART.top + chartHeight + 25,
      "text-anchor": index === 0 ? "start" : index === tickCount ? "end" : "middle",
    });
    label.textContent = formatTimecode(seconds);
    svg.append(label);
  }

  if (evidenceWindows.length === 0) {
    const empty = svgNode("text", {
      x: CHART.left + chartWidth / 2,
      y: CHART.top + chartHeight / 2,
      "text-anchor": "middle",
    });
    empty.textContent = "NO WINDOW PROBABILITY CURVES IN THIS MANIFEST";
    svg.append(empty);
  } else {
    for (const label of [...READINESS_LABELS].reverse()) {
      const coordinates = evidenceWindows.map((windowRecord) => ({
        x: x((windowRecord.start_s + windowRecord.end_s) / 2),
        y: y(readinessPoint(windowRecord, label)),
      }));
      const path = svgNode("path", {
        d: coordinates.map((point, index) => `${index === 0 ? "M" : "L"}${point.x.toFixed(2)},${point.y.toFixed(2)}`).join(" "),
        fill: "none",
        stroke: COLORS[label],
        "stroke-width": label === "KEEP" ? 3.5 : 2,
        "stroke-linejoin": "round",
        "stroke-linecap": "round",
        opacity: label === "KEEP" ? 1 : 0.82,
        "vector-effect": "non-scaling-stroke",
      });
      svg.append(path);
      if (label === "KEEP") {
        for (const point of coordinates) {
          svg.append(
            svgNode("circle", {
              cx: point.x,
              cy: point.y,
              r: 3.5,
              fill: COLORS.KEEP,
              stroke: "#10110f",
              "stroke-width": 1.5,
              "vector-effect": "non-scaling-stroke",
            }),
          );
        }
      }
    }
  }

  svg.append(
    svgNode("line", {
      id: "playhead-line",
      x1: CHART.left,
      y1: CHART.top,
      x2: CHART.left,
      y2: CHART.top + chartHeight,
      stroke: "#efeee6",
      "stroke-width": 1,
      "stroke-dasharray": "4 4",
      opacity: 0,
      "vector-effect": "non-scaling-stroke",
      "pointer-events": "none",
    }),
  );
  byId("timeline-end").textContent = formatTimecode(duration);
}

function renderSegmentTrack(parsed) {
  const track = byId("segment-track");
  clear(track);
  const duration = Math.max(parsed.header.source.duration_s, 0.001);
  if (parsed.segments.length === 0) {
    track.setAttribute("aria-label", "No compiled segments in this manifest");
    return;
  }
  for (const segment of parsed.segments) {
    const route = decisionRoute(segment);
    const button = node("button", `track-segment ${routeClass(route)}`);
    button.type = "button";
    button.dataset.segmentIndex = String(segment.segment_index);
    button.style.left = `${Math.max(0, Math.min(100, (segment.start_s / duration) * 100))}%`;
    button.style.width = `${Math.max(0.2, Math.min(100, ((segment.end_s - segment.start_s) / duration) * 100))}%`;
    button.title = `${decisionFor(segment)} · ${formatTimecode(segment.start_s)}–${formatTimecode(segment.end_s)}`;
    button.setAttribute("aria-label", `Select ${decisionFor(segment)} segment ${segment.segment_index}, ${formatTimecode(segment.start_s)} to ${formatTimecode(segment.end_s)}`);
    button.addEventListener("click", () => selectSegment(segment, true));
    track.append(button);
  }
}

function renderSegments() {
  const list = byId("segment-list");
  clear(list);
  const segments = visibleSegments();
  byId("download-segments").textContent = state.routeFilter === "all"
    ? "Export all segments"
    : `Export ${state.routeFilter === "discard" ? "reject" : state.routeFilter} segments`;
  if (segments.length === 0) {
    list.append(node("p", "empty-list", "No compiled segments match this route."));
    return;
  }
  for (const segment of segments) {
    const route = decisionRoute(segment);
    const decision = decisionFor(segment);
    const button = node("button", "segment-row");
    button.type = "button";
    button.dataset.segmentIndex = String(segment.segment_index);
    button.setAttribute("aria-label", `Inspect ${decision} segment ${segment.segment_index} at ${formatTimecode(segment.start_s)}`);

    button.append(node("span", "segment-row-index", String(segment.segment_index).padStart(2, "0")));
    button.append(node("span", `segment-route ${routeClass(route)}`, decision));

    const time = node("span", "segment-time");
    time.append(
      node("strong", "", `${formatTimecode(segment.start_s)} → ${formatTimecode(segment.end_s)}`),
      node("small", "", `${formatDuration(segment.end_s - segment.start_s)} · ${segment.reason ?? "stable"}`),
    );
    button.append(time);

    const confidence = segment.readiness?.[decision] ?? segment.mean_score;
    const score = node("span", "segment-score");
    score.append(node("strong", "", formatPercent(confidence)), node("small", "", "route confidence"));
    button.append(score);

    const reports = Array.isArray(segment.reported_issues) ? segment.reported_issues.length : 0;
    button.append(node("span", "segment-issue-count", `${reports} SIGNAL${reports === 1 ? "" : "S"}`));
    button.addEventListener("click", () => selectSegment(segment, true));
    list.append(button);
  }
  updateSelectedClasses();
}

function renderIssues(parsed) {
  const list = byId("issue-list");
  clear(list);
  const issues = aggregateIssues(parsed.segments.length > 0 ? parsed.segments : parsed.windows);
  if (issues.length === 0) {
    list.append(node("p", "empty-list", "No issue probabilities were recorded."));
    return;
  }
  for (const issue of issues) {
    const row = node("div", "issue-row");
    const heading = node("div", "issue-row-head");
    heading.append(
      node("span", "issue-name", prettyIssue(issue.name)),
      node("span", "issue-value", `${formatPercent(issue.max)} / ${formatPercent(issue.mean)}`),
    );
    const meter = node("div", "issue-meter");
    meter.title = `Maximum ${formatPercent(issue.max, 1)}; mean ${formatPercent(issue.mean, 1)} across ${issue.samples} record${issue.samples === 1 ? "" : "s"}`;
    const maximum = node("span");
    maximum.style.width = `${Math.min(100, issue.max * 100)}%`;
    const mean = node("i");
    mean.style.left = `${Math.min(100, issue.mean * 100)}%`;
    meter.append(maximum, mean);
    row.append(heading, meter);
    list.append(row);
  }
}

function metadataValue(value) {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function renderMetadata(header) {
  const source = header.source ?? {};
  const model = header.model ?? {};
  const sampling = header.sampling ?? {};
  const policy = header.policy ?? {};
  const geometry = source.display_width && source.display_height
    ? `${source.display_width} × ${source.display_height}${source.rotation_degrees ? ` · ${source.rotation_degrees}° display rotation` : ""}`
    : "—";
  const metadata = [
    ["Source file", displayPathLeaf(source.source_path)],
    ["Source SHA-256", source.source_sha256 ?? "not recorded"],
    ["Display geometry", geometry],
    ["Codec / FPS", `${source.codec_name ?? "unknown"} / ${source.fps ? Number(source.fps).toFixed(3) : "unknown"}`],
    ["Model", model.id ?? "not recorded"],
    ["Resolved revision", model.revision ?? "not recorded"],
    ["Processor revision", model.processor_revision ?? "not recorded"],
    ["Generator", header.generator ?? "not recorded"],
    ["Created", header.created_at ?? "not recorded"],
    ["Window sampling", sampling.window_duration_s ? `${sampling.window_duration_s}s / stride ${sampling.stride_s}s / ${sampling.frames_per_window} frames` : "not recorded"],
    ["Keep hysteresis", policy.enter_threshold !== undefined ? `enter ${policy.enter_threshold} / exit ${policy.exit_threshold}` : "not recorded"],
    ["Uncertainty route", policy.uncertainty_threshold !== undefined ? `${policy.uncertainty_route ?? "review"} at ${policy.uncertainty_threshold}` : "not recorded"],
  ];
  const grid = byId("metadata-grid");
  clear(grid);
  for (const [label, value] of metadata) {
    const item = node("div", "metadata-item");
    item.append(node("span", "", label), node("strong", "", metadataValue(value)));
    grid.append(item);
  }
  byId("raw-header").textContent = JSON.stringify(header, null, 2);
}

function renderSelected(segment) {
  const panel = byId("selected-panel");
  if (!segment) {
    panel.hidden = true;
    return;
  }
  const decision = decisionFor(segment);
  const route = decisionRoute(segment);
  panel.hidden = false;
  byId("selected-title").textContent = `Segment ${String(segment.segment_index).padStart(2, "0")}`;
  const routeLabel = byId("selected-route");
  routeLabel.textContent = decision;
  routeLabel.className = routeClass(route);
  byId("selected-time").textContent = `${formatTimecode(segment.start_s)} → ${formatTimecode(segment.end_s)}`;
  byId("selected-reason").textContent = `${formatDuration(segment.end_s - segment.start_s)} · ${segment.reason ?? "stable"}`;

  const scores = byId("selected-scores");
  clear(scores);
  for (const label of READINESS_LABELS) {
    const item = node("div");
    item.append(node("dt", "", label), node("dd", "", formatPercent(segment.readiness?.[label], 1)));
    scores.append(item);
  }

  const issues = byId("selected-issues");
  clear(issues);
  const ranked = Object.entries(segment.issues ?? {}).sort((left, right) => right[1] - left[1]);
  if (ranked.length === 0) {
    issues.append(node("span", "issue-chip", "No issue evidence recorded"));
  } else {
    for (const [name, probability] of ranked) {
      issues.append(node("span", "issue-chip", `${prettyIssue(name)} ${formatPercent(probability)}`));
    }
  }
}

function updateSelectedClasses() {
  for (const element of document.querySelectorAll("[data-segment-index]")) {
    element.classList.toggle(
      "is-selected",
      Number(element.dataset.segmentIndex) === state.selectedSegmentIndex,
    );
  }
}

function selectSegment(segment, seekVideo) {
  state.selectedSegmentIndex = segment.segment_index;
  renderSelected(segment);
  updateSelectedClasses();
  if (seekVideo) seekTo(segment.start_s);
}

function seekTo(seconds) {
  const video = byId("video");
  if (!state.videoUrl) {
    showToast(`Segment starts at ${formatTimecode(seconds)}. Open the matching video to seek.`);
    return;
  }
  const target = Number.isFinite(video.duration)
    ? Math.max(0, Math.min(seconds, Math.max(0, video.duration - 0.01)))
    : Math.max(0, seconds);
  video.currentTime = target;
  byId("video-timecode").textContent = formatTimecode(target);
  showToast(`Video set to ${formatTimecode(target)}.`);
}

function updatePlayhead(seconds) {
  const line = document.getElementById("playhead-line");
  if (!line || !state.parsed) return;
  const duration = Math.max(state.parsed.header.source.duration_s, 0.001);
  const chartWidth = CHART.width - CHART.left - CHART.right;
  const position = CHART.left + (Math.max(0, Math.min(duration, seconds)) / duration) * chartWidth;
  line.setAttribute("x1", String(position));
  line.setAttribute("x2", String(position));
  line.setAttribute("opacity", "0.7");
}

function renderWorkspace() {
  const parsed = state.parsed;
  if (!parsed) return;
  const header = parsed.header;
  const summary = summarizeManifest(parsed);
  const source = header.source;
  const model = header.model ?? {};
  byId("identity-file").textContent = state.manifestFileName;
  byId("identity-model").textContent = model.id
    ? `${model.id} @ ${model.revision ?? model.requested_revision ?? "unresolved revision"}`
    : "Model identity not recorded";
  byId("metric-duration").textContent = formatDuration(summary.duration);
  byId("metric-geometry").textContent = source.display_width && source.display_height
    ? `${source.display_width} × ${source.display_height} · ${source.fps ? `${Number(source.fps).toFixed(2)} fps` : "fps unknown"}`
    : "geometry not recorded";
  byId("metric-windows").textContent = String(summary.windows).padStart(2, "0");
  byId("metric-window-mix").textContent = `${summary.windowDecisions.KEEP} keep / ${summary.windowDecisions.REVIEW} review / ${summary.windowDecisions.REJECT} reject`;
  byId("metric-segments").textContent = String(summary.segments).padStart(2, "0");
  byId("metric-keep").textContent = formatDuration(summary.routedSeconds.KEEP);
  byId("metric-review").textContent = `${formatDuration(summary.routedSeconds.REVIEW)} routed to review`;
  checkVideoDuration();
  renderWarnings();
  renderTimeline(parsed);
  renderSegmentTrack(parsed);
  renderSegments();
  renderIssues(parsed);
  renderMetadata(header);
  updateManifestDownload();
  const selected = parsed.segments.find((segment) => segment.segment_index === state.selectedSegmentIndex)
    ?? parsed.segments[0]
    ?? null;
  state.selectedSegmentIndex = selected?.segment_index ?? null;
  renderSelected(selected);
  updateSelectedClasses();
  byId("workspace").hidden = false;
}

function acceptManifestText(text, fileName, isDemo = false) {
  const parsed = parseManifest(text);
  state.parsed = parsed;
  state.manifestText = text.endsWith("\n") ? text : `${text}\n`;
  state.manifestFileName = safeFileName(fileName, "egosieve-manifest.jsonl");
  state.selectedSegmentIndex = null;
  state.routeFilter = "all";
  for (const button of document.querySelectorAll("[data-route]")) {
    button.setAttribute("aria-pressed", button.dataset.route === "all" ? "true" : "false");
  }
  byId("manifest-file-name").textContent = state.manifestFileName;
  renderWorkspace();
  showNotice(
    isDemo
      ? "Illustrative evidence loaded. It demonstrates the inspector and is not a checkpoint result."
      : `${parsed.windows.length} windows and ${parsed.segments.length} compiled segments loaded locally.`,
  );
  byId("workspace").scrollIntoView({ behavior: "smooth", block: "start" });
}

async function loadManifestFile(file) {
  if (!file) return;
  if (file.size > MAX_MANIFEST_BYTES) {
    showNotice("That manifest is larger than 32 MB. Open a smaller EgoSieve JSONL file.", true);
    return;
  }
  try {
    acceptManifestText(await file.text(), file.name);
  } catch (error) {
    showNotice(error instanceof Error ? error.message : "The manifest could not be parsed.", true);
  }
}

function loadVideoFile(file) {
  if (!file) return;
  if (file.type && !file.type.startsWith("video/")) {
    showNotice("The optional media file must be a video supported by this browser.", true);
    return;
  }
  if (state.videoUrl) URL.revokeObjectURL(state.videoUrl);
  state.videoUrl = URL.createObjectURL(file);
  state.videoFileName = file.name;
  state.videoDurationWarning = null;
  const video = byId("video");
  video.src = state.videoUrl;
  video.hidden = false;
  video.load();
  byId("video-empty").hidden = true;
  byId("video-timecode").hidden = false;
  byId("video-file-name").textContent = file.name;
  byId("video-state").textContent = "LOCAL VIDEO";
  byId("media-footnote").textContent = "Click any segment or timeline position to seek this local file.";
  showToast(`${file.name} opened locally.`);
}

function resetWorkspace() {
  state.parsed = null;
  state.manifestText = "";
  state.manifestFileName = "";
  state.selectedSegmentIndex = null;
  state.routeFilter = "all";
  state.videoDurationWarning = null;
  revokeManifestUrl();
  if (state.videoUrl) URL.revokeObjectURL(state.videoUrl);
  state.videoUrl = null;
  state.videoFileName = "";
  const video = byId("video");
  video.pause();
  video.removeAttribute("src");
  video.load();
  video.hidden = true;
  byId("video-empty").hidden = false;
  byId("video-timecode").hidden = true;
  byId("video-state").textContent = "NO VIDEO";
  byId("manifest-input").value = "";
  byId("video-input").value = "";
  byId("manifest-file-name").textContent = "Drop JSONL here or choose a file";
  byId("video-file-name").textContent = "Drop video here or choose a file";
  byId("media-footnote").textContent = "Segment selections still work without media; their timestamps remain inspectable.";
  byId("workspace").hidden = true;
  hideNotice();
  showToast("Workspace cleared.");
}

function exportVisibleSegments() {
  if (!state.parsed) return;
  const selectedIndexes = new Set(visibleSegments().map((segment) => segment.segment_index));
  const payload = JSON.stringify(buildSegmentExport(state.parsed, selectedIndexes), null, 2) + "\n";
  const url = URL.createObjectURL(new Blob([payload], { type: "application/json;charset=utf-8" }));
  const anchor = document.createElement("a");
  const stem = safeFileName(state.manifestFileName, "egosieve-manifest").replace(/\.(jsonl|ndjson)$/iu, "");
  anchor.href = url;
  anchor.download = `${stem}.${state.routeFilter === "all" ? "segments" : state.routeFilter}.json`;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  showToast(`${selectedIndexes.size} segment${selectedIndexes.size === 1 ? "" : "s"} exported.`);
}

function chartPosition(event) {
  const svg = byId("timeline");
  const bounds = svg.getBoundingClientRect();
  const x = ((event.clientX - bounds.left) / Math.max(bounds.width, 1)) * CHART.width;
  const duration = state.parsed?.header.source.duration_s ?? 0;
  const fraction = (x - CHART.left) / (CHART.width - CHART.left - CHART.right);
  return Math.max(0, Math.min(duration, fraction * duration));
}

function nearestWindow(seconds) {
  if (!state.parsed || state.parsed.windows.length === 0) return null;
  let nearest = null;
  let nearestDistance = Infinity;
  for (const windowRecord of state.parsed.windows) {
    const midpoint = (windowRecord.start_s + windowRecord.end_s) / 2;
    const distance = Math.abs(midpoint - seconds);
    if (distance < nearestDistance) {
      nearest = windowRecord;
      nearestDistance = distance;
    }
  }
  return nearest;
}

function handleChartPointer(event) {
  if (!state.parsed) return;
  const windowRecord = nearestWindow(chartPosition(event));
  const tooltip = byId("chart-tooltip");
  if (!windowRecord?.readiness) {
    tooltip.hidden = true;
    return;
  }
  const stageBounds = byId("timeline-stage").getBoundingClientRect();
  tooltip.textContent = [
    `${formatTimecode(windowRecord.start_s)} → ${formatTimecode(windowRecord.end_s)}`,
    ...READINESS_LABELS.map((label) => `${label} ${formatPercent(windowRecord.readiness[label], 1)}`),
    `UNCERTAINTY ${formatPercent(windowRecord.routing_uncertainty ?? windowRecord.uncertainty, 1)}`,
  ].join("\n");
  tooltip.style.whiteSpace = "pre-line";
  tooltip.style.left = `${Math.min(stageBounds.width - 175, Math.max(8, event.clientX - stageBounds.left + 14))}px`;
  tooltip.style.top = `${Math.max(8, event.clientY - stageBounds.top - 88)}px`;
  tooltip.hidden = false;
}

function demoManifestText() {
  const duration = 42;
  const probabilities = [
    [0.82, 0.13, 0.05],
    [0.88, 0.09, 0.03],
    [0.52, 0.41, 0.07],
    [0.24, 0.63, 0.13],
    [0.72, 0.23, 0.05],
    [0.91, 0.07, 0.02],
    [0.86, 0.1, 0.04],
    [0.68, 0.28, 0.04],
    [0.37, 0.55, 0.08],
    [0.18, 0.32, 0.5],
  ];
  const issueNames = [
    "acting_hand_not_visible",
    "low_hand_activity",
    "camera_instability",
    "blur",
    "exposure",
    "scene_cut",
    "duplicate_frames",
  ];
  const windows = probabilities.map((values, index) => {
    const start = index * 4;
    const end = Math.min(duration, start + 6);
    const readiness = Object.fromEntries(READINESS_LABELS.map((label, labelIndex) => [label, values[labelIndex]]));
    const issues = Object.fromEntries(issueNames.map((name, issueIndex) => [
      name,
      Math.min(0.92, 0.03 + (((index + 2) * (issueIndex + 3)) % 11) * 0.035),
    ]));
    if (index === 3) issues.blur = 0.68;
    if (index === 8) issues.duplicate_frames = 0.57;
    return {
      record_type: "window",
      schema_version: 1,
      window_index: index,
      start_s: start,
      end_s: end,
      source_start_s: start,
      source_end_s: end,
      timestamps_s: Array.from({ length: 12 }, (_, frame) => start + ((frame + 0.5) * (end - start)) / 12),
      decision: READINESS_LABELS[values.indexOf(Math.max(...values))],
      readiness,
      score: readiness.KEEP,
      uncertainty: 0.2 + index * 0.025,
      routing_uncertainty: Math.max(0.2 + index * 0.025, readiness.REVIEW),
      issues,
      reported_issues: Object.entries(issues).filter(([, value]) => value >= 0.35).map(([name]) => name),
      boundary: [],
    };
  });
  const makeSegment = (segment_index, start_s, end_s, route, window_indices, readiness, issues, reported_issues = []) => ({
    record_type: "segment",
    schema_version: 1,
    segment_index,
    segment_id: `sha256:example-${segment_index}`,
    start_s,
    end_s,
    source_start_s: start_s,
    source_end_s: end_s,
    duration_s: end_s - start_s,
    route,
    decision: { keep: "KEEP", review: "REVIEW", discard: "REJECT" }[route],
    reason: route === "review" ? "uncertain" : "stable",
    window_indices,
    mean_score: readiness.KEEP,
    min_score: Math.max(0, readiness.KEEP - 0.08),
    max_score: Math.min(1, readiness.KEEP + 0.08),
    readiness,
    issues,
    reported_issues,
  });
  const baseIssues = Object.fromEntries(issueNames.map((name) => [name, 0.12]));
  const segments = [
    makeSegment(0, 0, 10, "keep", [0, 1], { KEEP: 0.85, REVIEW: 0.11, REJECT: 0.04 }, { ...baseIssues, low_hand_activity: 0.08 }),
    makeSegment(1, 10, 16, "review", [2, 3], { KEEP: 0.38, REVIEW: 0.52, REJECT: 0.1 }, { ...baseIssues, blur: 0.68, camera_instability: 0.41 }, ["blur", "camera_instability"]),
    makeSegment(2, 16, 30, "keep", [4, 5, 6, 7], { KEEP: 0.79, REVIEW: 0.17, REJECT: 0.04 }, { ...baseIssues, exposure: 0.29 }),
    makeSegment(3, 30, 36, "review", [7, 8], { KEEP: 0.39, REVIEW: 0.53, REJECT: 0.08 }, { ...baseIssues, duplicate_frames: 0.57 }, ["duplicate_frames"]),
  ];
  const header = {
    record_type: "manifest",
    schema: "egosieve.video-compilation",
    schema_version: 1,
    created_at: "2026-08-23T12:00:00Z",
    generator: "egosieve static-inspector example",
    example: "illustrative interface data; not a checkpoint result",
    source: {
      source_path: "example/assembly-pass.mp4",
      source_sha256: "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
      source_size_bytes: 18420000,
      duration_s: duration,
      start_time_s: 0,
      width: 1920,
      height: 1080,
      display_width: 1920,
      display_height: 1080,
      rotation_degrees: 0,
      fps: 29.97,
      frame_count: 1259,
      codec_name: "h264",
    },
    model: {
      id: "itspublu/EgoSieve-S",
      revision: "illustrative-example",
      requested_revision: "illustrative-example",
      processor_revision: "illustrative-example",
      readiness_labels: READINESS_LABELS,
    },
    counts: {
      windows: windows.length,
      unique_samples: 84,
      segments: segments.length,
      window_decisions: { KEEP: 7, REVIEW: 2, REJECT: 1 },
      segment_decisions: { KEEP: 2, REVIEW: 2, REJECT: 0 },
    },
    sampling: {
      window_duration_s: 6,
      stride_s: 4,
      frames_per_window: 12,
      window_count: windows.length,
      unique_sample_count: 84,
    },
    policy: {
      enter_threshold: 0.7,
      exit_threshold: 0.5,
      merge_gap_s: 0.5,
      min_duration_s: 1,
      uncertainty_threshold: 0.5,
      uncertainty_route: "review",
      short_segment_route: "discard",
      include_discard: false,
    },
    issue_labels: issueNames,
    issue_reporting_thresholds: Object.fromEntries(issueNames.map((name) => [name, 0.35])),
  };
  return [header, ...windows, ...segments].map((record) => JSON.stringify(record)).join("\n") + "\n";
}

function configureDropZone(zoneId, handler) {
  const zone = byId(zoneId);
  for (const eventName of ["dragenter", "dragover"]) {
    zone.addEventListener(eventName, (event) => {
      event.preventDefault();
      event.dataTransfer.dropEffect = "copy";
      zone.classList.add("is-dragging");
    });
  }
  for (const eventName of ["dragleave", "drop"]) {
    zone.addEventListener(eventName, (event) => {
      event.preventDefault();
      zone.classList.remove("is-dragging");
    });
  }
  zone.addEventListener("drop", (event) => handler(event.dataTransfer.files?.[0] ?? null));
}

function init() {
  byId("manifest-input").addEventListener("change", (event) => loadManifestFile(event.target.files?.[0]));
  byId("video-input").addEventListener("change", (event) => loadVideoFile(event.target.files?.[0]));
  configureDropZone("manifest-drop", loadManifestFile);
  configureDropZone("video-drop", loadVideoFile);
  byId("demo-button").addEventListener("click", () => {
    try {
      acceptManifestText(demoManifestText(), "illustrative.egosieve.jsonl", true);
    } catch (error) {
      showNotice(error instanceof Error ? error.message : "The example could not be loaded.", true);
    }
  });
  byId("reset-button").addEventListener("click", resetWorkspace);
  byId("copy-cli").addEventListener("click", () => copyText(byId("cli-command").textContent, "CLI command copied."));
  byId("copy-segment").addEventListener("click", () => {
    const segment = state.parsed?.segments.find((row) => row.segment_index === state.selectedSegmentIndex);
    if (segment) copyText(JSON.stringify(segment, null, 2), "Segment JSON copied.");
  });
  byId("download-segments").addEventListener("click", exportVisibleSegments);

  for (const button of document.querySelectorAll("[data-route]")) {
    button.addEventListener("click", () => {
      state.routeFilter = button.dataset.route;
      for (const peer of document.querySelectorAll("[data-route]")) {
        peer.setAttribute("aria-pressed", peer === button ? "true" : "false");
      }
      renderSegments();
    });
  }

  const video = byId("video");
  video.addEventListener("loadedmetadata", () => {
    checkVideoDuration();
    renderWarnings();
  });
  video.addEventListener("timeupdate", () => {
    byId("video-timecode").textContent = formatTimecode(video.currentTime);
    updatePlayhead(video.currentTime);
  });
  video.addEventListener("error", () => showNotice("This browser could not decode the selected video. The manifest remains available.", true));

  const timeline = byId("timeline");
  timeline.addEventListener("pointermove", handleChartPointer);
  timeline.addEventListener("pointerleave", () => {
    byId("chart-tooltip").hidden = true;
  });
  timeline.addEventListener("click", (event) => {
    if (state.parsed) seekTo(chartPosition(event));
  });

  window.addEventListener("beforeunload", () => {
    revokeManifestUrl();
    if (state.videoUrl) URL.revokeObjectURL(state.videoUrl);
  });
}

init();
