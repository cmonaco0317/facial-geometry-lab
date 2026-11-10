#!/usr/bin/env python3
"""Public implementation; private development notes omitted."""

from __future__ import annotations

import math
import os
import shutil

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# A face tilted further than this is not "upright". Chosen well above real head
# tilt in a casual photograph (a person leaning their head reaches maybe 20 deg)
# and far below the 90 deg that orientation loss produces, so the two cases
# cannot be confused. Deliberately NOT tight: this gate decides whether to ROTATE
# BY 90 DEGREES, not whether the photograph is any good -- capture_check already
# judges quality, and a 25 deg head tilt must reach it rather than be rotated.
ROLL_TOLERANCE_DEG = 30.0

# Where originals go when a file has to be replaced by its rotated self. Leading
# underscore keeps it out of every `*.png` glob in the project, which are all
# top-level and non-recursive (heic_convert.usable_photos,
# fit_identity_stage.set_paths).
ORIGINALS_DIR = "_originals_orientation"

IMAGE_EXTS = ("png", "jpg", "jpeg", "PNG", "JPG", "JPEG")

_landmarker = None


def _get_landmarker():
    """One landmarker for the whole pass. Building it costs ~1s."""
    global _landmarker
    if _landmarker is None:
        import mediapipe as mp
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision

        import fa_common as F

        _landmarker = (
            vision.FaceLandmarker.create_from_options(
                vision.FaceLandmarkerOptions(
                    base_options=mpp.BaseOptions(model_asset_path=F.MODEL_PATH),
                    num_faces=1,
                )
            ),
            mp,
        )
    return _landmarker


def measure_roll(img) -> float | None:
    """Roll in degrees from the inter-canthal line, or None if no face is found.

    The eye line rather than this project's usual midline-axis fit: the midline
    fit is the better instrument on an upright face, but it is being asked here
    to describe a face that may be lying on its side, which is outside what it
    was built for. Two points that are unambiguous at any rotation are the right
    tool for deciding rotation, and nothing downstream consumes this number --
    it only chooses a multiple of 90.
    """
    import fa_common as F

    lm, mp = _get_landmarker()
    h, w = img.shape[:2]
    res = lm.detect(
        mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        )
    )
    if not res.face_landmarks:
        return None
    pts = np.array([[p.x * w, p.y * h] for p in res.face_landmarks[0]])
    left, right = pts[F.L_CANTHUS_LAT], pts[F.R_CANTHUS_LAT]
    return math.degrees(math.atan2(right[1] - left[1], right[0] - left[0]))


def _wrap180(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def assess(img) -> dict:
    """How many 90-degree turns make this image upright?

    Returns k (0-3, counter-clockwise turns as np.rot90 applies them), the roll
    before and after, and whether the result is actually upright. k == 0 with
    ok == True is the common case and costs one detection.
    """
    roll0 = measure_roll(img)
    if roll0 is not None and abs(_wrap180(roll0)) <= ROLL_TOLERANCE_DEG:
        return {
            "k": 0,
            "roll_before": round(roll0, 2),
            "roll_after": round(roll0, 2),
            "ok": True,
            "face_found": True,
            "rotated": False,
        }

    best = None
    for k in (1, 2, 3):
        cand = np.rot90(img, k)
        r = measure_roll(np.ascontiguousarray(cand))
        if r is None:
            continue
        if best is None or abs(_wrap180(r)) < abs(_wrap180(best[1])):
            best = (k, r)

    if best is None:
        # No rotation produced a detectable face. If the original had one, keep
        # it and let the quality gate speak; if not, this is simply a photograph
        # with no findable face and intake will say so.
        return {
            "k": 0,
            "roll_before": None if roll0 is None else round(roll0, 2),
            "roll_after": None if roll0 is None else round(roll0, 2),
            "ok": roll0 is None,
            "face_found": roll0 is not None,
            "rotated": False,
            "note": "no rotation yielded a detectable face",
        }

    k, r = best
    upright = abs(_wrap180(r)) <= ROLL_TOLERANCE_DEG
    return {
        "k": k if upright else 0,
        "roll_before": None if roll0 is None else round(roll0, 2),
        "roll_after": (
            round(r, 2) if upright else (None if roll0 is None else round(roll0, 2))
        ),
        "best_rotation_roll": round(r, 2),
        "ok": upright,
        "face_found": True,
        "rotated": upright,
        "note": (
            None
            if upright
            else (
                f"the face is tilted {abs(_wrap180(roll0 or 0)):.0f} deg and no 90-degree "
                f"rotation makes it upright (best would still be "
                f"{abs(_wrap180(r)):.0f} deg)"
            )
        ),
    }


def normalize_dir(directory: str, verbose: bool = True) -> dict:
    """Make every photograph in a folder upright. Originals are preserved.

    Mirrors `heic_convert.convert_dir`: same folder, same non-recursive globs,
    same promise that nothing the user handed us is destroyed. A file that needs
    rotating is MOVED into `_originals_orientation/` and its rotated self written
    in its place under the original name -- the name has to survive, because
    `fit_identity_stage` identifies the lateral by the basename `profile`.
    """
    paths = []
    for ext in IMAGE_EXTS:
        import glob

        paths.extend(glob.glob(os.path.join(directory, f"*.{ext}")))
    paths = sorted(set(paths))

    out = {"checked": 0, "rotated": [], "refused": [], "no_face": [], "errors": []}
    if not paths:
        return out

    for p in paths:
        img = cv2.imread(p)
        if img is None:
            out["errors"].append((os.path.basename(p), "unreadable"))
            continue
        out["checked"] += 1
        try:
            a = assess(img)
        except Exception as exc:  # noqa: BLE001 - report, never silently skip
            out["errors"].append((os.path.basename(p), f"{type(exc).__name__}: {exc}"))
            continue

        if not a["face_found"]:
            out["no_face"].append(os.path.basename(p))
            continue
        if not a["ok"]:
            out["refused"].append((os.path.basename(p), a.get("note") or "not upright"))
            continue
        if a["k"] == 0:
            continue

        keep = os.path.join(directory, ORIGINALS_DIR)
        os.makedirs(keep, exist_ok=True)
        dest = os.path.join(keep, os.path.basename(p))
        if not os.path.exists(dest):
            shutil.move(p, dest)
        else:
            # Already preserved by an earlier pass; the working file is a
            # derivative and may be discarded.
            os.remove(p)
        rot = np.ascontiguousarray(np.rot90(cv2.imread(dest), a["k"]))
        cv2.imwrite(p, rot)
        out["rotated"].append(
            (os.path.basename(p), a["k"], a["roll_before"], a["roll_after"])
        )

    if verbose and (out["rotated"] or out["refused"] or out["errors"]):
        print("=" * 78)
        print("ORIENTATION — photographs are put upright before anything measures")
        print("=" * 78)
        for name, k, before, after in out["rotated"]:
            # ⭐ `before` IS LEGITIMATELY None, and the case matters most.
            # A photograph lying on its side can be one the detector cannot find
            # a face in AT ALL until it is turned upright -- so the frame with no
            # "before" reading is the frame that most needed rotating, and it is
            # the strongest evidence the correction was right. The first version
            # of this line assumed two floats and crashed on exactly that frame,
            # taking the whole intake check down with it. Found 2026-08-19 by a
            # real 15-photograph set, not by a synthetic one.
            b = "no face found" if before is None else f"{before:+.1f}°"
            a_ = "no face found" if after is None else f"{after:+.1f}°"
            print(f"  ROTATED  {name:<26} {k * 90:>3}°   roll {b} -> {a_}")
        for name, why in out["refused"]:
            print(f"  REFUSED  {name:<26} {why}")
        for name, why in out["errors"]:
            print(f"  ERROR    {name:<26} {why}")
        if out["rotated"]:
            print(
                f"\n  {len(out['rotated'])} photograph(s) had lost their orientation tag "
                f"and were lying on their side.\n"
                f"  Originals are untouched in {ORIGINALS_DIR}/. This is measured from the "
                f"face itself,\n  not from metadata, so it works on photographs whose EXIF "
                f"was stripped in transit."
            )
        print()
    return out


def warn_if_not_upright(paths: list[str], verbose: bool = True) -> list[dict]:
    """Public implementation; private development notes omitted."""
    out = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        try:
            a = assess(img)
        except Exception:  # noqa: BLE001
            continue
        a["file"] = os.path.basename(p)
        out.append(a)
        if verbose and a["face_found"] and (a["rotated"] or not a["ok"]):
            roll = a["roll_before"]
            print(
                f"!! {os.path.basename(p)} is not upright (face roll {roll:+.1f}°). "
                f"The two-photo path does NOT rotate;"
            )
            print(
                "   its inputs are the published baseline. Re-export it upright, or "
                "use --set, which does correct orientation."
            )
    return out


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        raise SystemExit("usage: orientation.py <photo-directory>")
    r = normalize_dir(sys.argv[1])
    print(
        f"checked {r['checked']}, rotated {len(r['rotated'])}, "
        f"refused {len(r['refused'])}, no face {len(r['no_face'])}, "
        f"errors {len(r['errors'])}"
    )
