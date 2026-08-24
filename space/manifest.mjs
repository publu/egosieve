export const MANIFEST_SCHEMA = "egosieve.video-compilation";
export const MANIFEST_VERSION = 1;
export const READINESS_LABELS = Object.freeze(["KEEP", "REVIEW", "REJECT"]);
export const ROUTE_TO_DECISION = Object.freeze({
  keep: "KEEP",
  review: "REVIEW",
  discard: "REJECT",
});

export class ManifestValidationError extends Error {
  constructor(message, line = null) {
    super(line === null ? message : `Line ${line}: ${message}`);
    this.name = "ManifestValidationError";
    this.line = line;
  }
}

function isObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function finiteNumber(value, name, line, { min = -Infinity, max = Infinity } = {}) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new ManifestValidationError(`${name} must be a finite number`, line);
  }
  if (value < min || value > max) {
    throw new ManifestValidationError(`${name} must be between ${min} and ${max}`, line);
  }
  return value;
}

function validateInterval(record, line) {
  const start = finiteNumber(record.start_s, "start_s", line, { min: 0 });
  const end = finiteNumber(record.end_s, "end_s", line, { min: 0 });
  if (end <= start) {
    throw new ManifestValidationError("end_s must be greater than start_s", line);
  }
}

function validateProbabilityMap(value, name, line, warnings, requireReadiness = false) {
  if (!isObject(value)) {
    throw new ManifestValidationError(`${name} must be an object`, line);
  }
  if (requireReadiness) {
    for (const label of READINESS_LABELS) {
      if (!Object.hasOwn(value, label)) {
        throw new ManifestValidationError(`${name} is missing ${label}`, line);
      }
    }
  }
  const entries = Object.entries(value);
  for (const [key, probability] of entries) {
    finiteNumber(probability, `${name}.${key}`, line, { min: 0, max: 1 });
  }
  if (requireReadiness) {
    const total = READINESS_LABELS.reduce((sum, label) => sum + value[label], 0);
    if (Math.abs(total - 1) > 0.02) {
      warnings.push(`Line ${line}: readiness probabilities sum to ${total.toFixed(3)}, not 1.`);
    }
  }
}

function validateHeader(header, line) {
  if (!isObject(header)) {
    throw new ManifestValidationError("the first non-empty line must be an object", line);
  }
  if (header.record_type !== "manifest") {
    throw new ManifestValidationError("the first record must have record_type \"manifest\"", line);
  }
  if (header.schema !== MANIFEST_SCHEMA) {
    throw new ManifestValidationError(`unsupported schema ${JSON.stringify(header.schema)}`, line);
  }
  if (header.schema_version !== MANIFEST_VERSION) {
    throw new ManifestValidationError(
      `unsupported schema version ${JSON.stringify(header.schema_version)}; expected ${MANIFEST_VERSION}`,
      line,
    );
  }
  if (!isObject(header.source)) {
    throw new ManifestValidationError("manifest.source must be an object", line);
  }
  finiteNumber(header.source.duration_s, "source.duration_s", line, { min: 0 });
}

function validateWindow(record, line, warnings) {
  if (!Number.isInteger(record.window_index) || record.window_index < 0) {
    throw new ManifestValidationError("window_index must be a non-negative integer", line);
  }
  validateInterval(record, line);
  if (record.decision !== undefined && !READINESS_LABELS.includes(record.decision)) {
    throw new ManifestValidationError("window decision must be KEEP, REVIEW, or REJECT", line);
  }
  if (record.readiness !== undefined) {
    validateProbabilityMap(record.readiness, "readiness", line, warnings, true);
  }
  if (record.issues !== undefined) {
    validateProbabilityMap(record.issues, "issues", line, warnings);
  }
  if (record.boundary !== undefined && !Array.isArray(record.boundary)) {
    throw new ManifestValidationError("boundary must be an array", line);
  }
}

function validateSegment(record, line, warnings) {
  if (!Number.isInteger(record.segment_index) || record.segment_index < 0) {
    throw new ManifestValidationError("segment_index must be a non-negative integer", line);
  }
  validateInterval(record, line);
  if (!Object.hasOwn(ROUTE_TO_DECISION, record.route)) {
    throw new ManifestValidationError("segment route must be keep, review, or discard", line);
  }
  if (record.decision !== undefined && !READINESS_LABELS.includes(record.decision)) {
    throw new ManifestValidationError("segment decision must be KEEP, REVIEW, or REJECT", line);
  }
  if (record.readiness !== undefined) {
    validateProbabilityMap(record.readiness, "readiness", line, warnings, true);
  }
  if (record.issues !== undefined) {
    validateProbabilityMap(record.issues, "issues", line, warnings);
  }
  if (!Array.isArray(record.window_indices) || record.window_indices.length === 0) {
    throw new ManifestValidationError("segment.window_indices must be a non-empty array", line);
  }
}

function declaredCount(header, key) {
  const value = header.counts?.[key];
  return Number.isInteger(value) && value >= 0 ? value : null;
}

/** Parse and validate an EgoSieve v1 JSONL manifest without mutating it. */
export function parseManifest(text) {
  if (typeof text !== "string" || text.trim() === "") {
    throw new ManifestValidationError("manifest is empty");
  }
  const records = [];
  const sourceLines = [];
  for (const [offset, rawLine] of text.split(/\r?\n/u).entries()) {
    if (rawLine.trim() === "") continue;
    const line = offset + 1;
    let record;
    try {
      record = JSON.parse(rawLine);
    } catch (error) {
      throw new ManifestValidationError(`invalid JSON (${error.message})`, line);
    }
    if (!isObject(record)) {
      throw new ManifestValidationError("record must be a JSON object", line);
    }
    records.push(record);
    sourceLines.push(line);
  }
  if (records.length === 0) {
    throw new ManifestValidationError("manifest is empty");
  }

  const header = records[0];
  validateHeader(header, sourceLines[0]);
  const windows = [];
  const segments = [];
  const warnings = [];
  const windowIndexes = new Set();
  const segmentIndexes = new Set();

  for (let index = 1; index < records.length; index += 1) {
    const record = records[index];
    const line = sourceLines[index];
    if (record.schema_version !== MANIFEST_VERSION) {
      throw new ManifestValidationError("record schema_version does not match the header", line);
    }
    if (record.record_type === "window") {
      validateWindow(record, line, warnings);
      if (windowIndexes.has(record.window_index)) {
        throw new ManifestValidationError(`duplicate window_index ${record.window_index}`, line);
      }
      windowIndexes.add(record.window_index);
      windows.push(record);
    } else if (record.record_type === "segment") {
      validateSegment(record, line, warnings);
      if (segmentIndexes.has(record.segment_index)) {
        throw new ManifestValidationError(`duplicate segment_index ${record.segment_index}`, line);
      }
      segmentIndexes.add(record.segment_index);
      segments.push(record);
    } else {
      throw new ManifestValidationError(`unknown record_type ${JSON.stringify(record.record_type)}`, line);
    }
  }

  windows.sort((left, right) => left.start_s - right.start_s || left.window_index - right.window_index);
  segments.sort((left, right) => left.start_s - right.start_s || left.segment_index - right.segment_index);

  const declaredWindows = declaredCount(header, "windows");
  if (declaredWindows !== null && declaredWindows !== windows.length) {
    warnings.push(`Header declares ${declaredWindows} windows; ${windows.length} were found.`);
  }
  const declaredSegments = declaredCount(header, "segments");
  if (declaredSegments !== null && declaredSegments !== segments.length) {
    warnings.push(`Header declares ${declaredSegments} segments; ${segments.length} were found.`);
  }
  const maxEnd = Math.max(0, ...windows.map((row) => row.end_s), ...segments.map((row) => row.end_s));
  if (maxEnd > header.source.duration_s + 0.05) {
    warnings.push(`Evidence extends to ${maxEnd.toFixed(2)}s, beyond the declared source duration.`);
  }
  for (const segment of segments) {
    const unknown = segment.window_indices.filter((value) => !windowIndexes.has(value));
    if (unknown.length > 0) {
      warnings.push(`Segment ${segment.segment_index} references missing window indexes: ${unknown.join(", ")}.`);
    }
  }

  return Object.freeze({
    records: Object.freeze(records),
    header,
    windows: Object.freeze(windows),
    segments: Object.freeze(segments),
    warnings: Object.freeze(warnings),
  });
}

export function decisionFor(record) {
  return record.decision ?? ROUTE_TO_DECISION[record.route] ?? "REVIEW";
}

export function aggregateIssues(records) {
  const buckets = new Map();
  for (const record of records) {
    if (!isObject(record.issues)) continue;
    for (const [name, probability] of Object.entries(record.issues)) {
      if (typeof probability !== "number" || !Number.isFinite(probability)) continue;
      const bucket = buckets.get(name) ?? { name, max: 0, sum: 0, count: 0, reports: 0 };
      bucket.max = Math.max(bucket.max, probability);
      bucket.sum += probability;
      bucket.count += 1;
      if (record.reported_issues?.includes(name)) bucket.reports += 1;
      buckets.set(name, bucket);
    }
  }
  return [...buckets.values()]
    .map((bucket) => ({
      name: bucket.name,
      max: bucket.max,
      mean: bucket.sum / bucket.count,
      reports: bucket.reports,
      samples: bucket.count,
    }))
    .sort((left, right) => right.max - left.max || left.name.localeCompare(right.name));
}

export function summarizeManifest(parsed) {
  const duration = parsed.header.source.duration_s;
  const windowDecisions = Object.fromEntries(READINESS_LABELS.map((label) => [label, 0]));
  for (const window of parsed.windows) {
    const decision = decisionFor(window);
    if (Object.hasOwn(windowDecisions, decision)) windowDecisions[decision] += 1;
  }
  const routedSeconds = Object.fromEntries(READINESS_LABELS.map((label) => [label, 0]));
  for (const segment of parsed.segments) {
    const decision = decisionFor(segment);
    if (Object.hasOwn(routedSeconds, decision)) {
      routedSeconds[decision] += segment.end_s - segment.start_s;
    }
  }
  return {
    duration,
    windows: parsed.windows.length,
    segments: parsed.segments.length,
    windowDecisions,
    routedSeconds,
    issues: aggregateIssues(parsed.segments.length > 0 ? parsed.segments : parsed.windows),
  };
}

export function formatDuration(seconds) {
  if (typeof seconds !== "number" || !Number.isFinite(seconds) || seconds < 0) return "—";
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const remainder = seconds % 60;
  if (hours > 0) {
    return `${hours}:${String(minutes).padStart(2, "0")}:${remainder.toFixed(1).padStart(4, "0")}`;
  }
  return `${minutes}:${remainder.toFixed(1).padStart(4, "0")}`;
}

export function buildSegmentExport(parsed, selectedIndexes = null) {
  const selected = selectedIndexes === null
    ? parsed.segments
    : parsed.segments.filter((segment) => selectedIndexes.has(segment.segment_index));
  return {
    schema: "egosieve.segment-selection/v1",
    exported_at: new Date().toISOString(),
    source: parsed.header.source,
    model: parsed.header.model ?? null,
    manifest: {
      schema: parsed.header.schema,
      schema_version: parsed.header.schema_version,
      created_at: parsed.header.created_at ?? null,
      generator: parsed.header.generator ?? null,
    },
    segments: selected,
  };
}
