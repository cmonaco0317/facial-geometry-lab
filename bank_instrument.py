#!/usr/bin/env python3
"""The instrument fingerprint — DERIVED from the import graph, never enumerated.

WHY THIS IS NOT A LIST OF FILENAMES
-----------------------------------
`OBJECTIVE-ACCURACY.md` §10: *"Each guard must DERIVE its coverage, never
enumerate it."*  That rule was written because on 2026-08-28 `selfcheck_tiers.py`
enumerated exactly the four surfaces `metric_tiers.py` named, the front page was
the fifth, and the ledger stayed green over the contradiction for eight days.

A bank is stale the moment ANY module that participates in measuring it changes.
Listing those modules by hand has the identical failure mode: the list is correct
until somebody adds a module, and adding modules is what improving this
instrument means.  So the fingerprint is taken by importing the measurement
entry point and hashing every project-local module that actually ended up in
`sys.modules` as a result -- which is a fact about what ran, not a claim about
what someone remembered to write down.

⭐ The model weights are hashed too.  `face_landmarker.task` is the largest single
determinant of every number here and it is not a `.py` file.

⛔ A bank whose fingerprint does not match the instrument measuring the subject
is a HARD RED, not a warning.  That is §5.3's rule mechanised.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def local_modules_loaded() -> dict[str, str]:
    """Every project-local module currently imported, and its content hash."""
    out = {}
    for name, mod in list(sys.modules.items()):
        path = getattr(mod, "__file__", None)
        if not path:
            continue
        path = os.path.abspath(path)
        # project-local means inside this directory and NOT inside a venv
        if not path.startswith(HERE + os.sep):
            continue
        if os.sep + "venv" in path or f"{os.sep}site-packages{os.sep}" in path:
            continue
        if not path.endswith(".py"):
            continue
        out[os.path.relpath(path, HERE)] = _sha256(path)[:16]
    return dict(sorted(out.items()))


def package_versions() -> dict[str, str]:
    """The measured versions of the packages that decide pixel-level results."""
    out = {}
    for name in ("mediapipe", "cv2", "numpy"):
        try:
            m = __import__(name)
            out[name] = getattr(m, "__version__", "unknown")
        except Exception:  # noqa: BLE001
            out[name] = "absent"
    out["python"] = sys.version.split()[0]
    return out


def model_hashes() -> dict[str, str]:
    """Model weights are part of the instrument and are not .py files."""
    out = {}
    for fn in sorted(os.listdir(HERE)):
        if fn.endswith((".task", ".tflite")):
            out[fn] = _sha256(os.path.join(HERE, fn))[:16]
    return out


def fingerprint() -> dict:
    """The whole instrument, as one record plus one comparable digest."""
    mods = local_modules_loaded()
    pkgs = package_versions()
    models = model_hashes()
    blob = repr((mods, pkgs, models)).encode()
    return {
        "digest": hashlib.sha256(blob).hexdigest()[:24],
        "modules": mods,
        "n_modules": len(mods),
        "packages": pkgs,
        "models": models,
    }


def compare(a: dict, b: dict) -> dict:
    """What differs between two fingerprints, in the terms a reader can act on."""
    diff = {
        "match": a.get("digest") == b.get("digest"),
        "changed": [],
        "added": [],
        "removed": [],
        "packages": [],
        "models": [],
    }
    am, bm = a.get("modules", {}), b.get("modules", {})
    for k in sorted(set(am) | set(bm)):
        if k not in am:
            diff["added"].append(k)
        elif k not in bm:
            diff["removed"].append(k)
        elif am[k] != bm[k]:
            diff["changed"].append(k)
    for k in sorted(set(a.get("packages", {})) | set(b.get("packages", {}))):
        if a.get("packages", {}).get(k) != b.get("packages", {}).get(k):
            diff["packages"].append(
                f"{k}: {a.get('packages', {}).get(k)} -> {b.get('packages', {}).get(k)}"
            )
    for k in sorted(set(a.get("models", {})) | set(b.get("models", {}))):
        if a.get("models", {}).get(k) != b.get("models", {}).get(k):
            diff["models"].append(k)
    return diff


if __name__ == "__main__":
    import json
    import analyze_averageness2  # noqa: F401  -- load the measurement path
    import facial_catalogue  # noqa: F401

    fp = fingerprint()
    print(json.dumps({k: v for k, v in fp.items() if k != "modules"}, indent=2))
    print(f"modules hashed: {fp['n_modules']}")


# --------------------------------------------------------------------------
# The MEASUREMENT fingerprint — taken in a clean subprocess.
#
# ⭐ WHY A SUBPROCESS RATHER THAN sys.modules IN-PROCESS.  `fingerprint()` hashes
# whatever happens to be loaded, which in a consumer's process includes the
# consumer.  A bank built before `bank_reference.py` existed would then never
# match any process that imports it, and a guard that is always red is a guard
# that gets switched off.  So the measurement fingerprint is taken by a fresh
# interpreter that imports ONLY the entry points below and nothing else.
#
# ⭐ ENUMERATING ENTRY POINTS IS NOT ENUMERATING COVERAGE.  §10's rule is that a
# guard must DERIVE what it covers.  These four names are the things that
# measure a bank face; everything they pull in transitively is discovered by
# importing them, so adding a module to the measurement path is picked up
# automatically and a stale hand-maintained list cannot form.

MEASUREMENT_ENTRY_POINTS = (
    "analyze_averageness2",  # detection, pose, the frontal gate
    "averageness_band",  # the at-rest / mouth-posture gate
    "facial_catalogue",  # the ratio definitions
    "dimorphism_bank_probe",  # the dimorphism feature definitions
)

_MEASUREMENT_CACHE: dict | None = None


def measurement_fingerprint(force: bool = False) -> dict:
    global _MEASUREMENT_CACHE
    if _MEASUREMENT_CACHE is not None and not force:
        return _MEASUREMENT_CACHE
    import json
    import subprocess

    code = (
        "import sys, json;"
        f"sys.path.insert(0, {HERE!r});"
        + "".join(f"import {m};" for m in MEASUREMENT_ENTRY_POINTS)
        + "import bank_instrument as BI;"
        "print('@@' + json.dumps(BI.fingerprint()))"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    line = [item for item in r.stdout.splitlines() if item.startswith("@@")]
    if not line:
        raise RuntimeError(f"measurement fingerprint failed:\n{r.stderr[-1500:]}")
    fp = json.loads(line[0][2:])
    # ⛔ bank_instrument itself is the observer, not the instrument. Hashing it
    # would make every edit to this file invalidate every bank ever built.
    fp["modules"].pop("bank_instrument.py", None)
    import hashlib as _h

    fp["digest"] = _h.sha256(
        repr((fp["modules"], fp["packages"], fp["models"])).encode()
    ).hexdigest()[:24]
    fp["n_modules"] = len(fp["modules"])
    fp["entry_points"] = list(MEASUREMENT_ENTRY_POINTS)
    _MEASUREMENT_CACHE = fp
    return fp


def _fingerprint_from_recorded_modules(recorded: dict) -> dict:
    """Rehash a packaged instrument whose private calibration inputs are absent."""
    modules = recorded.get("modules")
    entry_points = recorded.get("entry_points")
    if (
        not isinstance(modules, dict)
        or not modules
        or recorded.get("n_modules") != len(modules)
        or entry_points != list(MEASUREMENT_ENTRY_POINTS)
    ):
        raise ValueError("recorded instrument inventory is invalid")
    root = os.path.realpath(HERE)
    current_modules = {}
    for relative, expected_hash in sorted(modules.items()):
        if (
            not isinstance(relative, str)
            or not isinstance(expected_hash, str)
            or os.path.isabs(relative)
            or os.path.normpath(relative) != relative
            or relative.startswith(f"..{os.sep}")
            or not relative.endswith(".py")
        ):
            raise ValueError("recorded instrument inventory is invalid")
        lexical = os.path.join(HERE, relative)
        path = os.path.realpath(lexical)
        try:
            contained = os.path.commonpath((root, path)) == root
        except ValueError:
            contained = False
        if not contained or os.path.islink(lexical):
            raise ValueError("recorded instrument inventory is invalid")
        current_modules[relative] = _sha256(path)[:16]
    packages = package_versions()
    models = model_hashes()
    digest = hashlib.sha256(
        repr((current_modules, packages, models)).encode()
    ).hexdigest()[:24]
    return {
        "digest": digest,
        "modules": current_modules,
        "n_modules": len(current_modules),
        "packages": packages,
        "models": models,
        "entry_points": list(MEASUREMENT_ENTRY_POINTS),
    }


def _verified_packaged_manifest(recorded: dict) -> bool:
    """Return true only for an exact, hash-verified installed runtime payload."""
    manifest_path = os.path.join(HERE, "manifest.json")
    if not os.path.isfile(manifest_path) or os.path.islink(manifest_path):
        return False
    with open(manifest_path, encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict):
        raise ValueError("packaged runtime manifest is invalid")
    entries = manifest.get("files")
    measurement_modules = manifest.get("measurement_modules")
    runtime_assets = manifest.get("runtime_assets")
    recorded_modules = recorded.get("modules") if isinstance(recorded, dict) else None
    models = recorded.get("models") if isinstance(recorded, dict) else None
    if (
        manifest.get("version") != 1
        or not isinstance(entries, list)
        or not isinstance(measurement_modules, list)
        or not isinstance(runtime_assets, dict)
        or not isinstance(recorded_modules, dict)
        or len(measurement_modules) != len(set(measurement_modules))
        or set(measurement_modules) != set(recorded_modules)
    ):
        raise ValueError("packaged runtime manifest is invalid")
    if not isinstance(models, dict) or any(
        not isinstance(name, str)
        or not isinstance(digest, str)
        or os.path.basename(name) != name
        or not name.endswith((".task", ".tflite"))
        or len(digest) != 16
        or any(character not in "0123456789abcdef" for character in digest)
        for name, digest in models.items()
    ):
        raise ValueError("recorded model inventory is invalid")
    if any(
        not isinstance(name, str)
        or os.path.basename(name) != name
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        for name, digest in runtime_assets.items()
    ):
        raise ValueError("packaged runtime asset inventory is invalid")
    if any(
        not runtime_assets.get(name, "").startswith(digest)
        for name, digest in models.items()
    ):
        raise ValueError("packaged model inventory differs from the bank")

    expected: dict[str, tuple[int, str]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("packaged runtime manifest is invalid")
        relative = entry.get("path")
        size = entry.get("size")
        digest = entry.get("sha256")
        if (
            not isinstance(relative, str)
            or not relative
            or os.path.isabs(relative)
            or os.path.normpath(relative) != relative
            or relative.startswith(f"..{os.sep}")
            or relative == "manifest.json"
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or relative in expected
        ):
            raise ValueError("packaged runtime manifest is invalid")
        expected[relative] = (size, digest)

    if any(
        not isinstance(relative, str)
        or os.path.isabs(relative)
        or os.path.normpath(relative) != relative
        or relative.startswith(f"..{os.sep}")
        or not relative.endswith(".py")
        or relative not in expected
        for relative in measurement_modules
    ):
        raise ValueError("packaged measurement module inventory is invalid")

    allowed_files = set(expected) | set(runtime_assets) | {"manifest.json"}
    allowed_directories = {
        parent
        for relative in allowed_files
        for parent in (
            os.path.dirname(relative[:index])
            for index, character in enumerate(relative, start=1)
            if character == os.sep
        )
        if parent
    }
    actual_files: set[str] = set()
    for directory, names, filenames in os.walk(HERE, followlinks=False):
        directory_relative = os.path.relpath(directory, HERE)
        if directory_relative == ".":
            directory_relative = ""
        for name in names:
            path = os.path.join(directory, name)
            relative = (
                os.path.join(directory_relative, name) if directory_relative else name
            )
            if os.path.islink(path) or relative not in allowed_directories:
                raise ValueError("packaged runtime manifest is invalid")
        for name in filenames:
            path = os.path.join(directory, name)
            relative = (
                os.path.join(directory_relative, name) if directory_relative else name
            )
            if os.path.islink(path) or relative not in allowed_files:
                raise ValueError("packaged runtime manifest is invalid")
            actual_files.add(relative)
    if actual_files != allowed_files:
        raise ValueError("packaged runtime manifest is invalid")
    for relative, (size, digest) in expected.items():
        path = os.path.join(HERE, relative)
        if os.path.getsize(path) != size or _sha256(path) != digest:
            raise ValueError("packaged runtime manifest is invalid")
    for relative, digest in runtime_assets.items():
        if _sha256(os.path.join(HERE, relative)) != digest:
            raise ValueError("packaged runtime asset differs from its manifest")
    return True


def check_against(recorded: dict) -> tuple[bool, str]:
    """HARD RED when the instrument that would measure a subject is not the one
    that measured the bank.  §5.3 — this is the guard against the defect this
    project has hit four separate times: one label over two quantities."""
    try:
        packaged = _verified_packaged_manifest(recorded)
    except (OSError, TypeError, ValueError):
        return False, "the recorded bank instrument record is invalid"
    if packaged:
        try:
            now = _fingerprint_from_recorded_modules(recorded)
        except (OSError, TypeError, ValueError):
            return False, "the recorded bank instrument record is invalid"
    else:
        now = measurement_fingerprint()
    if recorded.get("digest") == now["digest"]:
        return True, "instrument matches the bank"
    d = compare(recorded, now)
    bits = []
    if d["changed"]:
        bits.append(f"changed: {', '.join(d['changed'][:6])}")
    if d["added"]:
        bits.append(f"added to the measurement path: {', '.join(d['added'][:6])}")
    if d["removed"]:
        bits.append(f"no longer in it: {', '.join(d['removed'][:6])}")
    if d["packages"]:
        bits.append(f"packages: {'; '.join(d['packages'])}")
    if d["models"]:
        bits.append(f"model weights: {', '.join(d['models'])}")
    return False, (
        "the bank was measured by a DIFFERENT instrument than this one "
        "(" + " | ".join(bits or ["no detail available"]) + "). Rebuild "
        "with build_bank.py, or the percentile compares two quantities."
    )
