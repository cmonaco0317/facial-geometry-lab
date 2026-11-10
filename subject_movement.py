#!/usr/bin/env python3
"""How much do THIS person's own photographs disagree with each other?

THE DEFECT THIS NARROWS
-----------------------
Every bracket and every precision word in this report is widened by a movement
term measured on ONE person across three sittings on one camera, applied
globally. `analyze_advanced.movement_bracket` takes no subject argument;
`analyze_harmony.band` looks its margin up by axis name alone; the catalogue
divides population spread by a single per-row movement. Nothing anywhere is
per-subject, and `the-report-says-whose-error-bars-it-is-using` discloses that
rather than fixing it, because fixing it was thought to need a second person
photographed on two different days.

⭐ IT DOES NOT. One thing IS measurable from the photographs already in hand:
split the subject's own set into TWO DISJOINT HALVES that EACH independently
pass the real intake gate, fit each alone, and read how far each shipped ratio
moves between two admissible answers about the same face at the same instant.
That is the subject's OWN number. `stranger_movement_probe.py` already performs
exactly this operation; nothing consumed it at report time.

WHAT THE SCALAR IS, AND WHAT REMAINS BORROWED
---------------------------------------------
    scalar = median over ratios of  (this subject's split-half movement)
                                    ---------------------------------------
                                    (reference subject's, SAME operation)

    margin_for_this_subject = shipped_margin * max(1.0, scalar)

So the SIZE of the disagreement becomes the subject's own and only the
RELATIONSHIP between same-instant and cross-sitting movement stays borrowed.
Today 100% of the margin belongs to somebody else.

⛔ WHY IT ONLY EVER WIDENS. The clamp at 1.0 is not caution for its own sake. A
scalar below 1 would NARROW a stranger's bracket — claiming more precision for a
person than for the one subject this instrument was actually calibrated on, on
the evidence of two sittings. Margins here are a FLOOR and the project's own
rule is that a floor can only grow. Narrowing is also the flattering direction,
which is the direction this project refuses by default.

⛔ WHY NOT cross-sitting / split-half. That was the first design and it is
CONFOUNDED. Cross-sitting movement is measured between FULL-set fits (7-9 views,
RMS 6.1-6.7px); split-half between HALF-set fits (4-5 views, RMS up to 11.3px).
Dividing them measures fit quality, not sittings: median 0.96x, spread 43.6x,
and BELOW 1.0 on 10 of 18 ratios, which cannot happen if one truly floors the
other. `stranger_movement_probe` had already recorded that only same-operation
ratios survive, and the first draft of this module ignored it.

WHAT THIS STILL IS NOT
----------------------
  * NOT cross-sitting reproducibility. Both halves come from ONE sitting, so
    expression, lighting, camera distance, hair and time of day are held
    perfectly constant — and those are exactly what a real second sitting
    varies. This is a floor, and the report says so where it prints.
  * NOT available on every capture. It needs >= 8 usable views splitting into
    two halves that BOTH clear 52 deg. The two-photo path, the single-frame
    fallback and small sets get scalar 1.0 and the borrowed margin, unchanged
    and disclosed.
  * NOT free of the reference. The denominator is 2 sittings of one person, and
    the aggregation matters: the shipped 1.30x stranger multiplier is 0.93x or
    2.79x on the same data depending on which sitting is used. `movement_reference`
    carries all three and this module uses the median, with the endpoints
    reported so the sensitivity travels with the number.

Outputs: subject_movement.json
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

OUT = os.path.join(HERE, "subject_movement.json")

# A set with more frames than this is not searched exhaustively for the best
# split: `best_split` is O(2^n) and 29-photo sittings exist on disk. The widest
# frames carry the split, so the search runs on the N most angularly spread.
MAX_SEARCH_FRAMES = 14

# Below this many ratios in common the median is not worth quoting and the
# scalar falls back to 1.0 rather than resting on a handful of rows.
MIN_RATIOS = 8


def _neutral(reason: str, **extra) -> dict:
    """Scalar 1.0 — the borrowed margin, unchanged, and the reason recorded.

    ⛔ A feature that silently does nothing looks identical to one that is
    broken, so every no-op path states which one it is.
    """
    out = {
        "_what": "how much this subject's own photographs disagree, vs the "
        "reference subject, under one identical operation",
        "scalar": 1.0,
        "applied": False,
        "reason": reason,
        "borrowed": True,
    }
    out.update(extra)
    return out


def measure(set_dir: str) -> dict:
    import fit_identity_stage as FIS  # noqa: PLC0415
    import stranger_movement_probe as SMP  # noqa: PLC0415

    try:
        import movement_reference as REF  # noqa: PLC0415
    except ImportError:
        return _neutral(
            "no movement_reference.py — regenerate it with "
            "stranger_movement_probe.py --emit"
        )

    paths = SMP.set_paths(set_dir) if hasattr(SMP, "set_paths") else None
    if paths is None:
        exts = ("png", "jpg", "jpeg", "PNG", "JPG", "JPEG")
        import glob  # noqa: PLC0415

        paths = sorted(
            p
            for e in exts
            for p in glob.glob(os.path.join(set_dir, f"*.{e}"))
            if not os.path.basename(p).lower().startswith("profile.")
        )
    if not paths:
        return _neutral(f"no photographs in {set_dir}")

    frames = [f for f in FIS.detect_frames(paths) if f.get("yaw") is not None]
    frames = [f for f in frames if abs(f["yaw"]) <= 75.0]
    if len(frames) < 2 * SMP.MIN_VIEWS:
        return _neutral(
            f"only {len(frames)} usable views, and two independently "
            f"gate-passing halves need {2 * SMP.MIN_VIEWS}",
            n_views=len(frames),
        )

    # Keep the angular extremes: a split is decided by spread, and the widest
    # frames are what let both halves clear 52 deg.
    if len(frames) > MAX_SEARCH_FRAMES:
        frames = sorted(frames, key=lambda f: abs(f["yaw"]))[:MAX_SEARCH_FRAMES]

    split = SMP.best_split(frames)
    if split is None:
        return _neutral(
            "no two disjoint halves of this set both clear the 52 deg floor",
            n_views=len(frames),
        )
    _score, A, B, sa, sb = split
    ra, rms_a = SMP.ratios_from(A)
    rb, rms_b = SMP.ratios_from(B)

    common = sorted(set(ra) & set(rb) & set(REF.SPLIT_HALF))
    ratios, per = [], {}
    for k in common:
        mine = abs(float(ra[k]) - float(rb[k]))
        theirs = float(REF.SPLIT_HALF[k]["median"])
        if theirs <= 0:
            continue
        per[k] = {"mine": mine, "reference": theirs, "ratio": mine / theirs}
        ratios.append(mine / theirs)
    if len(ratios) < MIN_RATIOS:
        return _neutral(
            f"only {len(ratios)} ratios in common with the reference, need "
            f"{MIN_RATIOS}",
            n_views=len(frames),
        )

    raw = float(np.median(ratios))
    # ⛔ THE CLAMP. Only ever widen — see the module docstring.
    scalar = max(1.0, raw)
    return {
        "_what": "how much this subject's own photographs disagree, vs the "
        "reference subject, under one identical operation",
        "_floor_not_measurement": (
            "Both halves come from ONE sitting, so expression, lighting, camera "
            "distance and posture are held constant. This is a FLOOR on "
            "cross-sitting movement, never a measurement of it."
        ),
        "_clamp": (
            "Only widens. A raw scalar below 1.0 would narrow this subject's "
            "brackets — claiming more precision for them than for the one person "
            "this instrument was calibrated on, on two sittings of evidence."
        ),
        "scalar": scalar,
        "raw_scalar": raw,
        "clamped": scalar != raw,
        "applied": scalar > 1.0,
        "borrowed": False,
        "n_ratios": len(ratios),
        "n_views": len(frames),
        "split": {
            "A": [f["file"] for f in A],
            "B": [f["file"] for f in B],
            "spread_A_deg": round(sa, 1),
            "spread_B_deg": round(sb, 1),
            "rms_px": [round(rms_a, 2), round(rms_b, 2)],
        },
        "reference": {
            "n_sittings": REF.N_SITTINGS,
            "sittings": REF.SITTINGS,
            "aggregation": "median across the reference subject's sittings",
            "sensitivity": (
                "The reference rests on 2 sittings of one person. On this same "
                "data the shipped stranger multiplier is 1.30x by median, 0.93x "
                "using the steadier sitting and 2.79x using the other."
            ),
        },
        "per_ratio": per,
    }


def load() -> dict:
    """The scalar as consumers should read it: absent file means 1.0."""
    try:
        with open(OUT, encoding="utf-8") as fh:
            d = json.load(fh)
        s = float(d.get("scalar", 1.0))
        return d if np.isfinite(s) and s >= 1.0 else _neutral("unreadable scalar")
    except (OSError, ValueError, TypeError):
        return _neutral("not measured for this capture")


def scalar() -> float:
    return float(load().get("scalar", 1.0))


def current_set_dir() -> str | None:
    """The set THIS capture was built from, or None. Answered from the bytes.

    ⛔⛔ THE DEFECT THIS CLOSES WAS IN THIS FILE, ON ITS FIRST DAY. `main` read
    `fit_diagnostics.json` for a set directory and trusted it. That file PERSISTS
    across runs, so after `--restore-two-photo` the two-photo baseline rebuilt
    with a scalar measured on a PREVIOUS SET RUN'S photographs -- a different
    capture entirely. It was invisible because the scalar happened to clamp to
    1.0; had that set been shakier, a two-photo report would have carried a
    widening measured on photographs it never saw.

    ⭐ This is the stale-intermediate class `fa_common.assert_fresh` exists for,
    and `run_pipeline.frontal_is_set_derived` already answers exactly this
    question the only way that cannot be fooled: HASH the reference photograph
    the fit recorded and compare it to the frontal on disk. A marker file or a
    flag can be stale; bytes cannot.
    """
    import hashlib

    try:
        with open(os.path.join(HERE, "fit_diagnostics.json"), encoding="utf-8") as fh:
            meta = json.load(fh)["_meta"]
        ref = os.path.join(meta["set_dir"], meta["reference_photo"])
        frontal = os.path.join(HERE, "frontal.png")
        if not (os.path.exists(ref) and os.path.exists(frontal)):
            return None

        def h(path):
            d = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    d.update(chunk)
            return d.hexdigest()

        return meta["set_dir"] if h(ref) == h(frontal) else None
    except (OSError, ValueError, KeyError):
        return None


def main() -> int:
    set_dir = sys.argv[1] if len(sys.argv) > 1 else current_set_dir()

    if not set_dir or not os.path.isdir(set_dir):
        res = _neutral(
            "this capture is not a photo set built by the fit — the two-photo "
            "baseline and the single-frame fallback measure one photograph, and "
            "one photograph cannot be split into two halves"
        )
    else:
        res = measure(set_dir)

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
        fh.write("\n")

    if res.get("applied"):
        print(
            f"  this subject's photographs disagree {res['scalar']:.2f}x as much "
            f"as the reference subject's, over {res['n_ratios']} ratios"
        )
        print("  every movement margin is widened by that factor")
    elif res.get("borrowed"):
        print(f"  scalar 1.0 (borrowed margin, unchanged) — {res['reason']}")
    else:
        print(
            f"  raw {res.get('raw_scalar', 1.0):.2f}x, clamped to 1.00 — this "
            "subject's photographs agree at least as well as the reference's, "
            "and a margin is a floor"
        )
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
