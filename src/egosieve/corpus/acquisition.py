"""Explicit, provenance-preserving acquisition of public corpus files.

Nothing in this module downloads data on import.  A caller must select a
known source, an immutable repository commit for repository-hosted files,
every individual source path, and the source's exact license identifier.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

CORPUS_MANIFEST_SCHEMA = "egosieve.corpus-acquisition/v1"
CORPUS_MANIFEST_NAME = "corpus-manifest.json"
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CHUNK_SIZE = 1024 * 1024


class CorpusAcquisitionError(ValueError):
    """Raised when selection, acquisition, or integrity validation fails."""


@dataclass(frozen=True)
class CorpusSource:
    """A source whose publisher and data terms have been recorded explicitly."""

    key: str
    name: str
    repository_id: str
    source_url: str
    license_id: str
    license_url: str
    license_declaration_url: str
    attribution: str
    fixed_download_urls: tuple[tuple[str, str], ...] = ()

    def resolve_url(self, revision: str, repository_path: str) -> str:
        for fixed_path, fixed_url in self.fixed_download_urls:
            if repository_path == fixed_path:
                return fixed_url
        encoded_path = quote(repository_path, safe="/")
        return (
            f"https://huggingface.co/datasets/{self.repository_id}/resolve/"
            f"{revision}/{encoded_path}"
        )


EGO_TACTILE = CorpusSource(
    key="ego-tactile",
    name="Ego-Tactile Manipulation",
    repository_id="OpenGraphLabs-Research/ego-tactile-manipulation",
    source_url="https://huggingface.co/datasets/OpenGraphLabs-Research/ego-tactile-manipulation",
    license_id="CC-BY-4.0",
    license_url="https://creativecommons.org/licenses/by/4.0/legalcode",
    license_declaration_url=(
        "https://huggingface.co/datasets/OpenGraphLabs-Research/ego-tactile-manipulation#license"
    ),
    attribution=(
        "OpenGraph Labs, Ego-Tactile Manipulation (2026), "
        "https://huggingface.co/datasets/"
        "OpenGraphLabs-Research/ego-tactile-manipulation"
    ),
)

HOLOASSIST_ANNOTATION_PATH = "data-annotation-trainval-v1_1.json"
HOLOASSIST_ANNOTATION_RELEASE = "v1_1"
HOLOASSIST_ANNOTATION_URL = (
    "https://hl2data.z5.web.core.windows.net/holoassist-data-release/"
    "data-annotation-trainval-v1_1.json"
)
HOLOASSIST_SCHEMA_URL = "https://holoassist.github.io/data_links/README.html"
HOLOASSIST_AUDIT_URL = (
    "https://openaccess.thecvf.com/content/ICCV2023/papers/"
    "Wang_HoloAssist_an_Egocentric_Human_Interaction_Dataset_for_"
    "Interactive_AI_Assistants_ICCV_2023_paper.pdf"
)
HOLOASSIST_MIRROR_URL = "https://huggingface.co/datasets/lmms-lab/EgoIT-99K"
HOLOASSIST_MIRROR_REVISION = "a57f1f2078a7b01ea87014050fdb3afe169e54f1"
_HOLOASSIST_VIDEO_RE = re.compile(r"^HoloAssist/video/[A-Za-z0-9][A-Za-z0-9._-]*\.mp4$")

HOLOASSIST = CorpusSource(
    key="holoassist",
    name="HoloAssist",
    # This repository is a byte-transport mirror for selected RGB videos.  The
    # publisher annotations and data terms are recorded independently below.
    repository_id="lmms-lab/EgoIT-99K",
    source_url="https://holoassist.github.io/",
    license_id="CDLA-Permissive-2.0",
    license_url="https://cdla.dev/permissive-2-0/",
    license_declaration_url="https://holoassist.github.io/#download",
    attribution=(
        "Wang et al., HoloAssist: an Egocentric Human Interaction Dataset for "
        "Interactive AI Assistants in the Real World (ICCV 2023), "
        "https://holoassist.github.io/"
    ),
    fixed_download_urls=((HOLOASSIST_ANNOTATION_PATH, HOLOASSIST_ANNOTATION_URL),),
)

SOURCES: dict[str, CorpusSource] = {
    EGO_TACTILE.key: EGO_TACTILE,
    HOLOASSIST.key: HOLOASSIST,
}


def get_source(key: str) -> CorpusSource:
    """Return a reviewed source profile by its CLI key."""

    try:
        return SOURCES[key]
    except KeyError as error:
        choices = ", ".join(sorted(SOURCES))
        raise CorpusAcquisitionError(
            f"unknown corpus source {key!r}; choose one of: {choices}"
        ) from error


def validate_revision(revision: str) -> str:
    """Require a full lowercase Git commit instead of a mutable branch or tag."""

    if not isinstance(revision, str) or not _COMMIT_RE.fullmatch(revision):
        raise CorpusAcquisitionError(
            "revision must be a full 40-character lowercase hexadecimal commit"
        )
    return revision


def validate_repository_path(value: str) -> str:
    """Validate that *value* selects one exact, relative repository file."""

    if not isinstance(value, str) or not value:
        raise CorpusAcquisitionError("repository file path must be a non-empty string")
    if "\\" in value or "?" in value or "#" in value:
        raise CorpusAcquisitionError(f"repository file path is not an exact path: {value!r}")
    if any(character in value for character in "*[]{}"):
        raise CorpusAcquisitionError(
            f"repository file path must not contain a glob expression: {value!r}"
        )
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise CorpusAcquisitionError(f"repository file path is unsafe: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute():
        raise CorpusAcquisitionError(f"repository file path must be relative: {value!r}")
    return path.as_posix()


def _holoassist_path_origin(path: str) -> str:
    if path == HOLOASSIST_ANNOTATION_PATH:
        return "publisher_annotations"
    if _HOLOASSIST_VIDEO_RE.fullmatch(path):
        return "media_mirror"
    raise CorpusAcquisitionError(
        "HoloAssist selections must be the exact official annotation filename or an exact "
        "HoloAssist/video/<recording>.mp4 mirror path"
    )


def _validate_profile_selection(profile: CorpusSource, selected: Sequence[str]) -> None:
    if profile.key != HOLOASSIST.key:
        return
    origins = [_holoassist_path_origin(path) for path in selected]
    if HOLOASSIST_ANNOTATION_PATH not in selected:
        raise CorpusAcquisitionError(
            f"HoloAssist acquisition requires explicit selection of {HOLOASSIST_ANNOTATION_PATH!r}"
        )
    if "media_mirror" not in origins:
        raise CorpusAcquisitionError(
            "HoloAssist acquisition requires at least one exact HoloAssist/video/*.mp4 path"
        )


def _selected_local_path(profile: CorpusSource, selected_path: str) -> PurePosixPath:
    if profile.key == HOLOASSIST.key and selected_path == HOLOASSIST_ANNOTATION_PATH:
        return PurePosixPath("files", "annotations", selected_path)
    return PurePosixPath("files") / PurePosixPath(selected_path)


def _source_repository(profile: CorpusSource, revision: str) -> dict[str, Any]:
    if profile.key == HOLOASSIST.key:
        if revision != HOLOASSIST_MIRROR_REVISION:
            raise CorpusAcquisitionError(
                "HoloAssist media mirror revision must match the reviewed exact commit "
                f"{HOLOASSIST_MIRROR_REVISION}"
            )
        return {
            "type": "huggingface_dataset_mirror",
            "id": profile.repository_id,
            "source_url": HOLOASSIST_MIRROR_URL,
            "revision": revision,
            "role": "media-transport-only",
            # The pinned mirror card has no license declaration.  The data
            # terms below therefore come only from the original publisher.
            "license_declaration_url": None,
            "publisher_byte_equivalence_verified": False,
        }
    return {
        "type": "huggingface_dataset",
        "id": profile.repository_id,
        "revision": revision,
    }


def _annotation_release() -> dict[str, str]:
    return {
        "type": "publisher_direct_file",
        "version": HOLOASSIST_ANNOTATION_RELEASE,
        "filename": HOLOASSIST_ANNOTATION_PATH,
        "download_url": HOLOASSIST_ANNOTATION_URL,
        "schema_url": HOLOASSIST_SCHEMA_URL,
        "audit_url": HOLOASSIST_AUDIT_URL,
    }


def _file_entry(
    profile: CorpusSource,
    *,
    selected_path: str,
    local_path: PurePosixPath,
    download_url: str,
    size: int,
    digest: str,
) -> dict[str, Any]:
    common = {
        "local_path": local_path.as_posix(),
        "download_url": download_url,
        "size_bytes": size,
        "sha256": digest,
    }
    if profile.key == HOLOASSIST.key:
        return {
            "selection_path": selected_path,
            "origin": _holoassist_path_origin(selected_path),
            **common,
        }
    return {"repository_path": selected_path, **common}


def sha256_file(path: str | Path) -> str:
    """Return the lowercase SHA-256 digest of a local file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(
    url: str,
    destination: Path,
    *,
    opener: Callable[[Request], Any],
) -> tuple[int, str]:
    request = Request(url, headers={"User-Agent": "egosieve-corpus/0.1"})
    digest = hashlib.sha256()
    size = 0
    try:
        response = opener(request)
    except OSError as error:
        raise CorpusAcquisitionError(f"download failed for {url}: {error}") from error
    try:
        with destination.open("xb") as handle:
            while True:
                chunk = response.read(_CHUNK_SIZE)
                if not chunk:
                    break
                if not isinstance(chunk, bytes):
                    raise CorpusAcquisitionError(f"download returned non-byte content for {url}")
                handle.write(chunk)
                digest.update(chunk)
                size += len(chunk)
    except CorpusAcquisitionError:
        raise
    except Exception as error:
        raise CorpusAcquisitionError(f"download failed for {url}: {error}") from error
    finally:
        close = getattr(response, "close", None)
        if callable(close):
            close()
    return size, digest.hexdigest()


def fetch_selected_files(
    source: CorpusSource | str,
    *,
    revision: str,
    repository_paths: Sequence[str],
    output_dir: str | Path,
    accepted_license: str,
    opener: Callable[[Request], Any] = urlopen,
) -> dict[str, Any]:
    """Download exactly *repository_paths* into a new, verified corpus directory.

    The operation is staged beside ``output_dir`` and renamed only after every
    selected file and the manifest have been written.  Existing output is
    never overwritten.  HoloAssist's one publisher-direct annotation filename
    is interpreted as a source selection rather than a mirror repository path.
    """

    if isinstance(source, str):
        profile = get_source(source)
    else:
        profile = get_source(source.key)
        if source != profile:
            raise CorpusAcquisitionError(
                "corpus source object does not match its reviewed source profile"
            )
    revision = validate_revision(revision)
    repository_metadata = _source_repository(profile, revision)
    if accepted_license != profile.license_id:
        raise CorpusAcquisitionError(
            f"license acknowledgement must be exactly {profile.license_id!r}"
        )
    if isinstance(repository_paths, str | bytes) or not repository_paths:
        raise CorpusAcquisitionError("select at least one repository file with an exact path")
    selected = [validate_repository_path(path) for path in repository_paths]
    if len(selected) != len(set(selected)):
        raise CorpusAcquisitionError("repository file selections must not contain duplicates")
    _validate_profile_selection(profile, selected)

    output = Path(output_dir)
    if output.exists():
        raise CorpusAcquisitionError(f"output directory already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))

    try:
        file_entries: list[dict[str, Any]] = []
        for repository_path in selected:
            local_path = _selected_local_path(profile, repository_path)
            destination = staging.joinpath(*local_path.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            url = profile.resolve_url(revision, repository_path)
            size, digest = _download(url, destination, opener=opener)
            file_entries.append(
                _file_entry(
                    profile,
                    selected_path=repository_path,
                    local_path=local_path,
                    download_url=url,
                    size=size,
                    digest=digest,
                )
            )

        manifest: dict[str, Any] = {
            "schema": CORPUS_MANIFEST_SCHEMA,
            "source": {
                "profile": profile.key,
                "name": profile.name,
                "source_url": profile.source_url,
                "repository": repository_metadata,
                "license": {
                    "id": profile.license_id,
                    "url": profile.license_url,
                    "declaration_url": profile.license_declaration_url,
                    "attribution": profile.attribution,
                },
            },
            "selection_policy": "explicit-files-only",
            "integrity_algorithm": "sha256",
            "files": file_entries,
        }
        if profile.key == HOLOASSIST.key:
            manifest["source"]["annotation_release"] = _annotation_release()
        (staging / CORPUS_MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CorpusAcquisitionError(f"{path} must be an object")
    return value


def _require_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise CorpusAcquisitionError(f"{path} must be a non-empty string")
    return value


def verify_corpus(
    corpus_dir: str | Path,
    *,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    """Validate a corpus manifest and, by default, every selected file digest."""

    root = Path(corpus_dir)
    manifest_path = root / CORPUS_MANIFEST_NAME
    try:
        manifest_value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise CorpusAcquisitionError(f"missing corpus manifest: {manifest_path}") from error
    except json.JSONDecodeError as error:
        raise CorpusAcquisitionError(f"invalid JSON in corpus manifest: {error.msg}") from error
    manifest = _require_mapping(manifest_value, "manifest")
    if manifest.get("schema") != CORPUS_MANIFEST_SCHEMA:
        raise CorpusAcquisitionError(f"manifest.schema must be {CORPUS_MANIFEST_SCHEMA!r}")
    source = _require_mapping(manifest.get("source"), "manifest.source")
    profile = get_source(_require_string(source.get("profile"), "manifest.source.profile"))
    if source.get("name") != profile.name or source.get("source_url") != profile.source_url:
        raise CorpusAcquisitionError("manifest source metadata does not match its reviewed profile")
    repository = _require_mapping(source.get("repository"), "manifest.source.repository")
    if repository.get("id") != profile.repository_id:
        raise CorpusAcquisitionError("manifest repository does not match its reviewed profile")
    revision = validate_revision(_require_string(repository.get("revision"), "revision"))
    if dict(repository) != _source_repository(profile, revision):
        raise CorpusAcquisitionError("manifest repository does not match its reviewed profile")
    if profile.key == HOLOASSIST.key:
        annotation_release = _require_mapping(
            source.get("annotation_release"), "manifest.source.annotation_release"
        )
        if dict(annotation_release) != _annotation_release():
            raise CorpusAcquisitionError(
                "manifest annotation release does not match the reviewed HoloAssist profile"
            )
    license_info = _require_mapping(source.get("license"), "manifest.source.license")
    for name in ("id", "url", "declaration_url", "attribution"):
        _require_string(license_info.get(name), f"manifest.source.license.{name}")
    expected_license = {
        "id": profile.license_id,
        "url": profile.license_url,
        "declaration_url": profile.license_declaration_url,
        "attribution": profile.attribution,
    }
    if dict(license_info) != expected_license:
        raise CorpusAcquisitionError(
            "manifest license metadata does not match its reviewed profile"
        )
    if manifest.get("selection_policy") != "explicit-files-only":
        raise CorpusAcquisitionError("manifest selection policy must be 'explicit-files-only'")
    if manifest.get("integrity_algorithm") != "sha256":
        raise CorpusAcquisitionError("manifest integrity algorithm must be 'sha256'")

    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise CorpusAcquisitionError("manifest.files must be a non-empty array")
    seen: set[str] = set()
    total_size = 0
    verified_files: list[dict[str, Any]] = []
    for index, entry_value in enumerate(entries):
        entry = _require_mapping(entry_value, f"manifest.files[{index}]")
        path_field = "selection_path" if profile.key == HOLOASSIST.key else "repository_path"
        repository_path = validate_repository_path(
            _require_string(entry.get(path_field), f"manifest.files[{index}].{path_field}")
        )
        expected_local = _selected_local_path(profile, repository_path).as_posix()
        local_path = _require_string(entry.get("local_path"), f"manifest.files[{index}].local_path")
        if local_path != expected_local:
            raise CorpusAcquisitionError(
                f"manifest.files[{index}].local_path must be {expected_local!r}"
            )
        if repository_path in seen:
            raise CorpusAcquisitionError(
                f"duplicate repository path in manifest: {repository_path}"
            )
        seen.add(repository_path)
        if profile.key == HOLOASSIST.key:
            expected_origin = _holoassist_path_origin(repository_path)
            if entry.get("origin") != expected_origin:
                raise CorpusAcquisitionError(
                    f"manifest.files[{index}].origin must be {expected_origin!r}"
                )
        download_url = _require_string(
            entry.get("download_url"), f"manifest.files[{index}].download_url"
        )
        expected_url = profile.resolve_url(revision, repository_path)
        if download_url != expected_url:
            raise CorpusAcquisitionError(
                f"download URL for {repository_path} does not match the pinned source revision"
            )
        size = entry.get("size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise CorpusAcquisitionError(
                f"manifest.files[{index}].size_bytes must be a non-negative integer"
            )
        expected_hash = _require_string(entry.get("sha256"), f"manifest.files[{index}].sha256")
        if not _SHA256_RE.fullmatch(expected_hash):
            raise CorpusAcquisitionError(f"invalid SHA-256 for {repository_path}")
        path = root.joinpath(*PurePosixPath(local_path).parts)
        if not path.is_file():
            raise CorpusAcquisitionError(f"selected corpus file is missing: {path}")
        actual_size = path.stat().st_size
        if actual_size != size:
            raise CorpusAcquisitionError(
                f"size mismatch for {repository_path}: expected {size}, found {actual_size}"
            )
        if verify_hashes:
            actual_hash = sha256_file(path)
            if actual_hash != expected_hash:
                raise CorpusAcquisitionError(
                    f"SHA-256 mismatch for {repository_path}: "
                    f"expected {expected_hash}, found {actual_hash}"
                )
        total_size += size
        verified_files.append(dict(entry))

    _validate_profile_selection(profile, sorted(seen))

    return {
        "manifest": dict(manifest),
        "manifest_path": str(manifest_path),
        "file_count": len(verified_files),
        "total_size_bytes": total_size,
        "files": verified_files,
    }


__all__ = [
    "CORPUS_MANIFEST_NAME",
    "CORPUS_MANIFEST_SCHEMA",
    "EGO_TACTILE",
    "HOLOASSIST",
    "HOLOASSIST_ANNOTATION_PATH",
    "HOLOASSIST_ANNOTATION_RELEASE",
    "HOLOASSIST_ANNOTATION_URL",
    "HOLOASSIST_AUDIT_URL",
    "HOLOASSIST_MIRROR_URL",
    "HOLOASSIST_MIRROR_REVISION",
    "HOLOASSIST_SCHEMA_URL",
    "SOURCES",
    "CorpusAcquisitionError",
    "CorpusSource",
    "fetch_selected_files",
    "get_source",
    "sha256_file",
    "validate_repository_path",
    "validate_revision",
    "verify_corpus",
]
