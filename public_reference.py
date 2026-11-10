"""Aggregate reference distributions; no individual reference faces or row links."""

from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parent / "bank"
SCHEMA = "public-reference-cdf-v1"


def _metadata_at(root):
    path = root / "public_reference.json"
    if path.is_symlink() or path.stat().st_size > 2_000_000:
        raise ValueError("Invalid public reference metadata")
    record = json.loads(path.read_text())
    if (record.get("schema") != SCHEMA
            or record.get("minimum_distinct_identity_clusters") != 150
            or record.get("minimum_cdf_count") != 50
            or not isinstance(record.get("cohorts"), dict)):
        raise ValueError("Unsupported public reference schema")
    data = root / "derived.npz"
    if data.is_symlink() or data.stat().st_size > 40_000_000:
        raise ValueError("Invalid public reference data")
    if hashlib.sha256(data.read_bytes()).hexdigest() != record.get("sha256"):
        raise ValueError("Public reference checksum differs")
    return record


def _arrays_at(root, record):
    published_members = set()
    for cohort, info in record["cohorts"].items():
        if info.get("status") != "available":
            continue
        count = info.get("distinct_identity_clusters")
        rows = info.get("source_rows")
        members = info.get("members", [cohort])
        if (type(count) is not int or type(rows) is not int
                or not 150 <= count <= rows <= 200_000
                or not isinstance(members, list) or not members
                or any(not isinstance(m, str) or not re.fullmatch(r"[a-z0-9_]{1,80}", m)
                       for m in members)
                or len(set(members)) != len(members)
                or published_members.intersection(members)):
            raise ValueError("Invalid or overlapping public reference cohorts")
        published_members.update(members)
    with zipfile.ZipFile(root / "derived.npz") as bundle:
        entries = bundle.infolist()
        if (len(entries) > 20_000 or sum(e.file_size for e in entries) > 150_000_000
                or any(e.file_size > 20_000_000 for e in entries)):
            raise ValueError("Public reference archive exceeds resource bounds")
    with np.load(root / "derived.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    for key, array in arrays.items():
        parts = key.split("|")
        cohort = parts[0]
        info = record["cohorts"].get(cohort, {})
        if (info.get("status") != "available"
                or info.get("distinct_identity_clusters", 0) < 150
                or not re.fullmatch(r"[a-z0-9_]{1,80}", cohort)
                or array.dtype.kind not in "fi"
                or not np.isfinite(array).all()):
            raise ValueError("Invalid aggregate reference entry")
        if len(parts) == 2 and parts[1] == "_mean":
            if array.shape != (468, 2):
                raise ValueError("Invalid aggregate mean shape")
        elif len(parts) == 3 and parts[2] in {"x", "n"}:
            if array.ndim != 1 or not 1 <= len(array) <= 200_000:
                raise ValueError("Invalid empirical distribution")
            if parts[2] == "x":
                counts = arrays.get("|".join(parts[:2]) + "|n")
                if (counts is None or counts.shape != array.shape
                        or counts.dtype.kind not in "iu" or np.any(counts < 1)
                        or np.any(counts > 200_000)
                        or not 50 <= int(counts.sum()) <= 200_000
                        or int(counts.sum()) > info["source_rows"]
                        or np.any(np.diff(array) <= 0)):
                    raise ValueError("Invalid empirical distribution counts")
            elif "|".join(parts[:2]) + "|x" not in arrays:
                raise ValueError("Unpaired empirical distribution")
        else:
            raise ValueError("Individual reference records are not supported")
        array.setflags(write=False)
    return arrays


@lru_cache(maxsize=1)
def metadata():
    return _metadata_at(ROOT)


@lru_cache(maxsize=1)
def _arrays():
    return _arrays_at(ROOT, metadata())


def validate_directory(root):
    root = Path(root)
    if any(root.glob("*/measurements.npz")):
        raise ValueError("Individual reference records are forbidden")
    _arrays_at(root, _metadata_at(root))


def pool(cohort, metric):
    """Expand one independent ECDF; its order carries no reference row identity."""
    arrays = _arrays()
    key = f"{cohort}|{metric}"
    values, counts = arrays.get(key + "|x"), arrays.get(key + "|n")
    return None if values is None else np.repeat(values, counts)


def metric_names(cohort):
    prefix = cohort + "|"
    return sorted(key.split("|")[1] for key in _arrays()
                  if key.startswith(prefix) and key.endswith("|x")
                  and not key.split("|")[1].startswith("_"))


def average_reference(cohort):
    info = metadata()["cohorts"].get(cohort, {})
    mean = _arrays().get(cohort + "|_mean")
    distances = pool(cohort, "_typicality")
    if mean is None or distances is None or info.get("status") != "available":
        raise ValueError("No eligible aggregate typicality reference")
    return mean, distances, int(info["source_rows"])
