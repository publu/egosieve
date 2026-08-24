import assert from "node:assert/strict";
import test from "node:test";

import {
  ManifestValidationError,
  aggregateIssues,
  buildSegmentExport,
  decisionFor,
  formatDuration,
  parseManifest,
  summarizeManifest,
} from "./manifest.mjs";

function fixture(overrides = {}) {
  const header = {
    record_type: "manifest",
    schema: "egosieve.video-compilation",
    schema_version: 1,
    created_at: "2026-08-23T12:00:00Z",
    generator: "egosieve test",
    source: {
      source_path: "source.mp4",
      source_sha256: "f".repeat(64),
      duration_s: 12,
      display_width: 1280,
      display_height: 720,
      fps: 30,
    },
    model: { id: "example/EgoSieve-S", revision: "a".repeat(40) },
    counts: { windows: 2, segments: 2 },
    ...overrides,
  };
  const windows = [
    {
      record_type: "window",
      schema_version: 1,
      window_index: 0,
      start_s: 0,
      end_s: 6,
      decision: "KEEP",
      readiness: { KEEP: 0.8, REVIEW: 0.15, REJECT: 0.05 },
      issues: { blur: 0.1, exposure: 0.2 },
      reported_issues: [],
      boundary: [],
    },
    {
      record_type: "window",
      schema_version: 1,
      window_index: 1,
      start_s: 6,
      end_s: 12,
      decision: "REVIEW",
      readiness: { KEEP: 0.25, REVIEW: 0.65, REJECT: 0.1 },
      issues: { blur: 0.7, exposure: 0.3 },
      reported_issues: ["blur"],
      boundary: [],
    },
  ];
  const segments = [
    {
      record_type: "segment",
      schema_version: 1,
      segment_index: 0,
      start_s: 0,
      end_s: 6,
      route: "keep",
      decision: "KEEP",
      reason: "stable",
      window_indices: [0],
      readiness: { KEEP: 0.8, REVIEW: 0.15, REJECT: 0.05 },
      issues: { blur: 0.1, exposure: 0.2 },
      reported_issues: [],
    },
    {
      record_type: "segment",
      schema_version: 1,
      segment_index: 1,
      start_s: 6,
      end_s: 12,
      route: "review",
      decision: "REVIEW",
      reason: "uncertain",
      window_indices: [1],
      readiness: { KEEP: 0.25, REVIEW: 0.65, REJECT: 0.1 },
      issues: { blur: 0.7, exposure: 0.3 },
      reported_issues: ["blur"],
    },
  ];
  return [header, ...windows, ...segments].map((record) => JSON.stringify(record)).join("\n");
}

test("parses a versioned manifest and preserves sorted evidence", () => {
  const parsed = parseManifest(fixture());
  assert.equal(parsed.header.schema_version, 1);
  assert.deepEqual(parsed.windows.map((row) => row.window_index), [0, 1]);
  assert.deepEqual(parsed.segments.map((row) => row.segment_index), [0, 1]);
  assert.deepEqual(parsed.warnings, []);
});

test("summarizes routing and issue evidence", () => {
  const summary = summarizeManifest(parseManifest(fixture()));
  assert.deepEqual(summary.windowDecisions, { KEEP: 1, REVIEW: 1, REJECT: 0 });
  assert.deepEqual(summary.routedSeconds, { KEEP: 6, REVIEW: 6, REJECT: 0 });
  assert.equal(summary.issues[0].name, "blur");
  assert.equal(summary.issues[0].max, 0.7);
  assert.ok(Math.abs(summary.issues[0].mean - 0.4) < 1e-12);
});

test("rejects malformed JSON, schemas, intervals, and probabilities", () => {
  assert.throws(() => parseManifest("{oops"), /Line 1: invalid JSON/u);
  assert.throws(
    () => parseManifest(fixture({ schema: "unknown" })),
    (error) => error instanceof ManifestValidationError && /unsupported schema/u.test(error.message),
  );
  assert.throws(
    () => parseManifest(fixture().replace('"end_s":6', '"end_s":0')),
    /end_s must be greater/u,
  );
  assert.throws(
    () => parseManifest(fixture().replace('"blur":0.1', '"blur":1.1')),
    /issues\.blur must be between/u,
  );
});

test("reports count and reference mismatches without hiding the evidence", () => {
  const text = fixture({ counts: { windows: 8, segments: 9 } }).replace(
    '"window_indices":[1]',
    '"window_indices":[1,99]',
  );
  const parsed = parseManifest(text);
  assert.equal(parsed.warnings.length, 3);
  assert.match(parsed.warnings[0], /declares 8 windows/u);
  assert.match(parsed.warnings[2], /missing window indexes: 99/u);
});

test("exports only selected segments with source and model identity", () => {
  const parsed = parseManifest(fixture());
  const exported = buildSegmentExport(parsed, new Set([1]));
  assert.equal(exported.schema, "egosieve.segment-selection/v1");
  assert.equal(exported.source.source_sha256, "f".repeat(64));
  assert.equal(exported.model.revision, "a".repeat(40));
  assert.deepEqual(exported.segments.map((row) => row.segment_index), [1]);
});

test("formatters and route fallback remain deterministic", () => {
  assert.equal(formatDuration(65.25), "1:05.3");
  assert.equal(formatDuration(3661.2), "1:01:01.2");
  assert.equal(decisionFor({ route: "discard" }), "REJECT");
  assert.deepEqual(aggregateIssues([]), []);
});
