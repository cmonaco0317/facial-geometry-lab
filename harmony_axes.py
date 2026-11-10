#!/usr/bin/env python3
"""§7.4 — the harmony magnitudes, computed from LANDMARKS so a cohort can have
a distribution of them, and grade the subject against THAT instead of against a
published ideal.

WHY THIS REPLACEMENT IS NOT COSMETIC
------------------------------------
Every harmony axis today grades a magnitude against two hand-chosen cutoffs, and
those cutoffs come from published "ideals" that **disagree with each other more
than a face moves**: the nasolabial angle spans 80-120 degrees across nine
sources against 9.2 degrees of measured cross-session movement, a ratio of
4.35x.  Grading a face against one of nine disagreeing opinions is not a
measurement.  §7.4 replaces the opinion with a distribution: *"ideal" stops being
a number someone published and becomes a percentile.*

⛔ ONE DEFINITION, NOT TWO.  Everything here routes through
`analyze_frontal.compute` -- the same function the report uses -- or copies the
`analyze_extended` definition verbatim with its line noted.  Two copies of a
vocabulary drift from each other, and this project has already paid for that
five times.  `selfcheck_harmony_axes.py` asserts these magnitudes reproduce
`analyze_harmony`'s on the subject's own face, so drift goes red rather than
quiet.

⛔ WHAT CANNOT BE DONE AND IS REFUSED RATHER THAN APPROXIMATED.
`Profile balance` needs a profile photograph and the bank has none; the
mentolabial term of `Lower-face / chin balance` needs the same.  Neither gets a
cohort distribution, and neither is quietly replaced by a frontal stand-in.
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import analyze_frontal as AF  # noqa: E402
import fa_common as F  # noqa: E402

# Axis id -> (label, what a LARGER magnitude means). Deviations are SIGNED where
# the direction is meaningful, because a cohort percentile needs a direction --
# "further from the middle" is not the same claim as "bigger".
AXES = {
    "vertical_balance": "Vertical balance (lower third − upper third)",
    "horizontal_balance": "Horizontal balance (central gap − eye-width fifth)",
    "midface_proportion": "Midface proportion (eye-line→lip / IPD)",
    "nose_mouth_eye": "Nose–mouth–eye proportion",
    "lip_balance": "Lip balance (upper : lower vermilion)",
    "lower_face_split": "Lower-face split (philtrum : lip–chin)",
}

REFUSED_ON_THE_BANK = {
    "profile_balance": "needs a profile photograph; the bank is frontal-only",
    "mentolabial": "needs a profile photograph; the lower-face axis therefore "
                   "ships without its mentolabial term on the ranked path",
}


def magnitudes(P468: np.ndarray) -> dict[str, float]:
    """Every frontal harmony magnitude, from one 468-point mesh.

    ⚠️ `analyze_frontal.compute` needs the 478-point array with iris landmarks;
    the bank stores 468. `fa_common.with_mesh_iris` supplies eye-ring centroids,
    which the project measured to reproduce the detected iris IPD to 1.4%.
    """
    P = F.with_mesh_iris(np.asarray(P468, dtype=np.float64))
    out: dict[str, float] = {}
    try:
        m = AF.compute(P)
    except Exception:  # noqa: BLE001
        return out

    thirds = {t["zone"].split()[0].lower(): t for t in m["thirds"]["items"]}
    if "upper" in thirds and "lower" in thirds:
        out["vertical_balance"] = float(thirds["lower"]["pct"] - thirds["upper"]["pct"])

    fifths = m["fifths"]["items"]
    eye = [x for x in fifths if "eye (" in x["zone"]]
    inter = [x for x in fifths if "Intercanthal" in x["zone"]]
    if eye and inter:
        out["horizontal_balance"] = float(
            inter[0]["pct"] - sum(x["pct"] for x in eye) / len(eye))

    phi = {r["ratio"]: r for r in m["phi"]}
    if "Nose width / avg eye width" in phi and "Mouth width / nose width" in phi:
        ne = float(phi["Nose width / avg eye width"]["actual"])
        mn = float(phi["Mouth width / nose width"]["actual"])
        # analyze_harmony axis 4: the two relations combined into one read.
        out["nose_mouth_eye"] = float(max(abs(ne - 1.0),
                                          abs((mn - 1.618) / 1.618) * 1.618))
    if "Lower face (philtrum / lip-chin)" in phi:
        out["lower_face_split"] = float(phi["Lower face (philtrum / lip-chin)"]["actual"])

    # analyze_extended.py:468-472, copied verbatim so the two cannot drift.
    try:
        ipd = F.dist(P, F.L_IRIS, F.R_IRIS)
        mc_y = (P[F.L_CANTHUS_MED][1] + P[F.R_CANTHUS_MED][1]) / 2
        midface_h = abs(P[13][1] - mc_y)
        upper_verm = F.vdist(P, F.LABRALE_SUP, 13)
        lower_verm = F.vdist(P, 14, F.LABRALE_INF)
        if ipd > 0:
            out["midface_proportion"] = float(midface_h / ipd)
        if lower_verm > 0:
            out["lip_balance"] = float(upper_verm / lower_verm)
    except Exception:  # noqa: BLE001
        pass
    return out


def percentile_in(value: float, pool: np.ndarray) -> float | None:
    """Where a subject sits among a cohort — the replacement for a published ideal."""
    pool = pool[np.isfinite(pool)]
    if pool.size < 50 or value is None or not np.isfinite(value):
        return None
    return float(100.0 * (pool < value).mean())


if __name__ == "__main__":
    import json
    z = np.load(os.path.join(HERE, "bank", "m_white_20_29", "measurements.npz"))
    g = magnitudes(z["landmarks"][0])
    print(json.dumps({k: round(v, 4) for k, v in g.items()}, indent=2))
    print("refused on the bank:", json.dumps(REFUSED_ON_THE_BANK, indent=2))
