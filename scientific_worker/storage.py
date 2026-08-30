"""Artifact storage for the scientific worker — GCS with a local dev mode.

Canonical layout (one trusted bucket, TOPOSCOUT_GCS_BUCKET):

    runs/<run_id>/input/<filename>
    runs/<run_id>/attempt1/{mask.png, overlay.jpg, prob.npy, prob.json}
    runs/<run_id>/attempt2/{mask.png, overlay.jpg}
    runs/<run_id>/report/report.json

The container filesystem is never persistent state; the probability map is
cached in the bucket (attempt1/prob.npy) so attempt 2 reuses the exact map.

Trust rules:
- gs:// input URIs must live inside the trusted bucket (no arbitrary fetches).
- Local paths are accepted only when TOPOSCOUT_SW_ALLOW_LOCAL_IO=1 (dev/tests);
  artifacts then go under TOPOSCOUT_SW_LOCAL_ARTIFACTS with the same layout.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

import numpy as np

BUCKET_ENV = "TOPOSCOUT_GCS_BUCKET"
LOCAL_IO_ENV = "TOPOSCOUT_SW_ALLOW_LOCAL_IO"
LOCAL_ARTIFACTS_ENV = "TOPOSCOUT_SW_LOCAL_ARTIFACTS"
DEFAULT_LOCAL_ARTIFACTS = "demo_outputs/worker_local"

ALLOWED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")  # mirrors runstate/schemas


class StorageError(Exception):
    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


def bucket_name() -> str | None:
    return os.environ.get(BUCKET_ENV, "").strip() or None


def local_io_allowed() -> bool:
    return os.environ.get(LOCAL_IO_ENV, "").strip() == "1"


def _local_root() -> Path:
    return Path(os.environ.get(LOCAL_ARTIFACTS_ENV, "").strip() or DEFAULT_LOCAL_ARTIFACTS)


def _bucket():
    from google.cloud import storage as gcs  # imported lazily; not needed in local mode
    name = bucket_name()
    if not name:
        raise StorageError("bucket_not_configured", f"set {BUCKET_ENV}")
    return gcs.Client().bucket(name)


def run_prefix(run_id: str) -> str:
    return f"runs/{run_id}"


def _require_trusted_gs(uri: str) -> str:
    """Return the object key of a gs:// URI, enforcing the trusted bucket."""
    name = bucket_name()
    prefix = f"gs://{name}/"
    if not name or not uri.startswith(prefix):
        raise StorageError("untrusted_image_uri",
                           f"image_uri must start with gs://{name or '<unconfigured>'}/")
    return uri[len(prefix):]


def load_image_bgr(image_uri: str) -> "np.ndarray":
    """Fetch the input image (trusted bucket or dev-mode local path) as BGR."""
    import cv2

    if image_uri.startswith("gs://"):
        key = _require_trusted_gs(image_uri)
        suffix = Path(key).suffix.lower()
        if suffix not in ALLOWED_IMAGE_SUFFIXES:
            raise StorageError("unsupported_image_type", suffix)
        blob = _bucket().blob(key)
        if not blob.exists():
            raise StorageError("input_not_found", image_uri)
        with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
            blob.download_to_filename(tmp.name)
            img = cv2.imread(tmp.name)
    else:
        if not local_io_allowed():
            raise StorageError("local_paths_disabled",
                               "only gs:// URIs are accepted (set TOPOSCOUT_SW_ALLOW_LOCAL_IO=1 for dev)")
        p = Path(image_uri)
        if p.suffix.lower() not in ALLOWED_IMAGE_SUFFIXES:
            raise StorageError("unsupported_image_type", p.suffix)
        if not p.is_file():
            raise StorageError("input_not_found", image_uri)
        img = cv2.imread(str(p))
    if img is None:
        raise StorageError("unreadable_image", image_uri)
    return img


def _put_bytes(key: str, data: bytes, content_type: str) -> str:
    if bucket_name() and not local_io_allowed():
        blob = _bucket().blob(key)
        blob.upload_from_string(data, content_type=content_type)
        return f"gs://{bucket_name()}/{key}"
    path = _local_root() / key
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return str(path)


def _get_bytes(key: str) -> bytes | None:
    if bucket_name() and not local_io_allowed():
        blob = _bucket().blob(key)
        if not blob.exists():
            return None
        return blob.download_as_bytes()
    path = _local_root() / key
    return path.read_bytes() if path.is_file() else None


def save_attempt_artifacts(run_id: str, attempt: int, mask: "np.ndarray",
                           overlay_bgr: "np.ndarray", prob: "np.ndarray | None",
                           image_shape: tuple[int, int]) -> dict[str, str | None]:
    """Persist mask/overlay (+ prob map on attempt 1) under the canonical layout."""
    import cv2

    prefix = f"{run_prefix(run_id)}/attempt{attempt}"
    ok, mask_png = cv2.imencode(".png", mask.astype(np.uint8) * 255)
    assert ok
    ok, overlay_jpg = cv2.imencode(".jpg", overlay_bgr)
    assert ok
    mask_uri = _put_bytes(f"{prefix}/mask.png", mask_png.tobytes(), "image/png")
    overlay_uri = _put_bytes(f"{prefix}/overlay.jpg", overlay_jpg.tobytes(), "image/jpeg")

    prob_uri = None
    if prob is not None and attempt == 1:
        import io
        buf = io.BytesIO()
        np.save(buf, prob.astype(np.float32))
        prob_uri = _put_bytes(f"{prefix}/prob.npy", buf.getvalue(), "application/octet-stream")
        _put_bytes(f"{prefix}/prob.json",
                   json.dumps({"shape": list(image_shape), "dtype": "float32"}).encode(),
                   "application/json")
    return {"mask_uri": mask_uri, "overlay_uri": overlay_uri, "prob_uri": prob_uri}


def load_cached_prob(run_id: str, expected_shape: tuple[int, int]) -> "np.ndarray | None":
    """Exact attempt-1 probability map for this run, if present and shape-valid."""
    import io

    meta_raw = _get_bytes(f"{run_prefix(run_id)}/attempt1/prob.json")
    prob_raw = _get_bytes(f"{run_prefix(run_id)}/attempt1/prob.npy")
    if meta_raw is None or prob_raw is None:
        return None
    try:
        meta = json.loads(meta_raw)
        if meta.get("shape") != list(expected_shape):
            return None
        return np.load(io.BytesIO(prob_raw))
    except Exception:
        return None


def validate_run_input(run_id: str, image_uri: str) -> str:
    """Enforce the canonical per-run input contract; return the object key or
    (dev mode) the staged local path.

    Cloud mode: image_uri MUST be gs://<trusted bucket>/runs/<run_id>/input/<file>
    with a single, allowed-suffix filename — any other bucket, any other run's
    prefix, or any nested path is rejected. Dev mode (TOPOSCOUT_SW_ALLOW_LOCAL_IO=1,
    no bucket): the path must live under <local root>/runs/<run_id>/input/.
    """
    if not RUN_ID_RE.match(run_id or ""):
        raise StorageError("invalid_run_id", repr(run_id))

    if image_uri.startswith("gs://"):
        name = bucket_name()
        prefix = f"gs://{name}/runs/{run_id}/input/"
        if not name or not image_uri.startswith(prefix):
            raise StorageError(
                "untrusted_image_uri",
                f"image_uri must be {('gs://' + name) if name else 'gs://<unconfigured>'}"
                f"/runs/{run_id}/input/<file>")
        fname = image_uri[len(prefix):]
        if not fname or "/" in fname:
            raise StorageError("untrusted_image_uri", "input object must be a single filename")
        if Path(fname).suffix.lower() not in ALLOWED_IMAGE_SUFFIXES:
            raise StorageError("unsupported_image_type", Path(fname).suffix)
        return f"runs/{run_id}/input/{fname}"

    if not local_io_allowed():
        raise StorageError("local_paths_disabled",
                           "only canonical gs:// input URIs are accepted in cloud mode")
    p = Path(image_uri)
    canonical_dir = (_local_root() / "runs" / run_id / "input").resolve()
    try:
        resolved = p.resolve(strict=True)
    except OSError:
        raise StorageError("input_not_found", image_uri)
    if resolved.parent != canonical_dir:
        raise StorageError("untrusted_image_uri",
                           f"dev-mode input must live under {canonical_dir}")
    if resolved.suffix.lower() not in ALLOWED_IMAGE_SUFFIXES:
        raise StorageError("unsupported_image_type", resolved.suffix)
    return str(resolved)


def download_run_input(run_id: str, image_uri: str, dest_dir: str | Path | None = None) -> Path:
    """Fetch the validated canonical run input to a run-specific tmp directory."""
    key_or_path = validate_run_input(run_id, image_uri)

    if not image_uri.startswith("gs://"):
        return Path(key_or_path)  # dev mode: already a validated local file

    dest = Path(dest_dir) if dest_dir else Path(tempfile.gettempdir()) / "toposcout_runs" / run_id
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / Path(key_or_path).name
    blob = _bucket().blob(key_or_path)
    if not blob.exists():
        raise StorageError("input_not_found", image_uri)
    blob.download_to_filename(str(target))
    return target


def find_input(run_id: str) -> tuple[str, bytes] | None:
    """Return (name, bytes) of the run's canonical input from DURABLE storage
    (bucket, or the local mirror in dev mode) — never a staging directory, so
    it survives restarts and other instances."""
    if not RUN_ID_RE.match(run_id or ""):
        return None
    prefix = f"{run_prefix(run_id)}/input/"
    if bucket_name() and not local_io_allowed():
        blobs = list(_bucket().list_blobs(prefix=prefix, max_results=2))
        if not blobs:
            return None
        return blobs[0].name, blobs[0].download_as_bytes()
    d = _local_root() / "runs" / run_id / "input"
    files = sorted(p for p in d.glob("*") if p.is_file()) if d.is_dir() else []
    if not files:
        return None
    return files[0].name, files[0].read_bytes()


def upload_input(run_id: str, local_path: str | Path) -> str:
    """Agent-side helper: place the original image under runs/<run_id>/input/."""
    p = Path(local_path)
    if p.suffix.lower() not in ALLOWED_IMAGE_SUFFIXES:
        raise StorageError("unsupported_image_type", p.suffix)
    content_type = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    return _put_bytes(f"{run_prefix(run_id)}/input/{p.name}", p.read_bytes(), content_type)


def save_report(run_id: str, report: dict) -> str:
    return _put_bytes(f"{run_prefix(run_id)}/report/report.json",
                      json.dumps(report, indent=2).encode(), "application/json")
