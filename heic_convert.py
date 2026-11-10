#!/usr/bin/env python3
"""Turn an iPhone camera roll into something the pipeline can read.

WHY THIS EXISTS
---------------
iPhones shoot HEIC by default and OpenCV cannot decode it. `capture_check.read_image`
already worked around that for ONE photo by converting through a temporary file, so
the intake checker could see a HEIC -- but `fit_identity_stage.set_paths` globs
`png/jpg/jpeg` only. A friend who dropped a folder of camera-roll photos in
therefore got an intake report that listed their photos and a fit that silently
used none of them, or refused the set for having too few views.

The instruction the README used to give was "convert HEIC with sips". That is a
step a person has to know to take, and forgetting it produces a WRONG SET rather
than an error, which is the failure mode this project cares about most.

WHAT IT DOES, AND WHAT IT DELIBERATELY DOES NOT
-----------------------------------------------
Converts `<stem>.heic` to `<stem>.png` NEXT TO the original, using macOS `sips` --
the same decoder `capture_check` already trusts, so a photograph reads identically
whether it arrived as HEIC or as PNG.

* Originals are never moved, renamed or deleted. The user's photographs are the
  one thing here that cannot be regenerated.
* A photograph that already has a sibling image (`IMG_1.jpg` beside `IMG_1.HEIC`)
  is SKIPPED, not converted. Writing a second copy would hand the fitter the same
  view twice, and this project measured that angular DIVERSITY rather than photo
  count is what determines the fit -- a duplicate is worse than a missing photo
  because it looks like coverage.
* The output is verified to be a real PNG before it is accepted, because a
  truncated file that exists is harder to diagnose than one that does not.
* Nothing is retried silently: a failure is returned and named.
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess

HEIC_EXTS = (".heic", ".heif", ".HEIC", ".HEIF")
IMAGE_EXTS = ("png", "jpg", "jpeg", "PNG", "JPG", "JPEG")
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
MIN_PNG_BYTES = 1024


def find_heic(directory: str) -> list[str]:
    """Every HEIC/HEIF in the directory itself (not in subdirectories)."""
    out: list[str] = []
    for ext in HEIC_EXTS:
        out.extend(glob.glob(os.path.join(directory, f"*{ext}")))
    return sorted(set(out))


def _sibling_image(path: str) -> str | None:
    """An already-usable image with the same stem, if one is sitting there."""
    stem = os.path.splitext(path)[0]
    for ext in IMAGE_EXTS:
        cand = f"{stem}.{ext}"
        if os.path.exists(cand):
            return cand
    return None


def _is_real_png(path: str) -> bool:
    try:
        if os.path.getsize(path) < MIN_PNG_BYTES:
            return False
        with open(path, "rb") as fh:
            return fh.read(8) == PNG_MAGIC
    except OSError:
        return False


def convert_one(src: str) -> tuple[str, str | None]:
    """Convert one HEIC. Returns (status, path) with status in
    converted / skipped-existing / failed."""
    existing = _sibling_image(src)
    if existing is not None:
        return "skipped-existing", existing
    dest = os.path.splitext(src)[0] + ".png"
    r = subprocess.run(
        ["sips", "-s", "format", "png", src, "--out", dest],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0 or not _is_real_png(dest):
        # Leave nothing half-written: a 0-byte PNG beside a good HEIC would be
        # picked up as a photograph by every later glob in this pipeline.
        if os.path.exists(dest):
            os.remove(dest)
        return "failed", None
    return "converted", dest


def convert_dir(directory: str, verbose: bool = True) -> dict:
    """Convert every HEIC in a photo folder in place. Idempotent."""
    found = find_heic(directory)
    result: dict = {
        "found": len(found),
        "converted": [],
        "skipped": [],
        "failed": [],
        "tool": None,
    }
    if not found:
        return result
    if shutil.which("sips") is None:
        result["tool"] = "missing"
        result["failed"] = [os.path.basename(p) for p in found]
        if verbose:
            print(
                f"!! {len(found)} HEIC photo(s) found and no `sips` on this machine.\n"
                "   `sips` is macOS-only. Convert them to PNG or JPEG with any tool\n"
                "   (Preview, an online converter, ImageMagick) and put the results\n"
                "   in the same folder, then run this again."
            )
        return result
    result["tool"] = "sips"
    for src in found:
        status, path = convert_one(src)
        name = os.path.basename(src)
        if status == "converted":
            result["converted"].append(os.path.basename(path))
        elif status == "skipped-existing":
            result["skipped"].append(name)
        else:
            result["failed"].append(name)
    if verbose:
        bits = []
        if result["converted"]:
            bits.append(f"{len(result['converted'])} converted to PNG")
        if result["skipped"]:
            bits.append(f"{len(result['skipped'])} already had a usable copy")
        if result["failed"]:
            bits.append(f"{len(result['failed'])} FAILED")
        print(f"     HEIC: {len(found)} found — {', '.join(bits)} (originals kept)")
        for name in result["failed"]:
            print(f"       !! could not convert {name}")
    return result


def usable_photos(directory: str) -> list[str]:
    """The photographs a run would actually fit, after conversion.

    Mirrors `fit_identity_stage.set_paths`: png/jpg/jpeg, minus the lateral. It is
    duplicated rather than imported because importing that module pulls in the
    whole fitting stack, and `--check` must stay cheap enough to run on a folder
    before anything is installed or measured.
    """
    keep = []
    for ext in IMAGE_EXTS:
        for p in glob.glob(os.path.join(directory, f"*.{ext}")):
            if os.path.splitext(os.path.basename(p))[0].lower() == "profile":
                continue
            keep.append(p)
    return sorted(set(keep))


def find_profile(directory: str) -> str | None:
    for ext in IMAGE_EXTS:
        hits = glob.glob(os.path.join(directory, f"profile.{ext}"))
        if hits:
            return hits[0]
    return None


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        raise SystemExit("usage: heic_convert.py <photo-directory>")
    convert_dir(sys.argv[1])
