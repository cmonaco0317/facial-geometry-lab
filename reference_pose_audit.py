"""Is the reference bank held to the same pose standard as the subject? READ ONLY.

THE ASYMMETRY THAT PROMPTED THIS
--------------------------------
The subject's averageness percentile is gated hard: `pose_gate.averageness_yaw_deg`
withholds the figure past |yaw| 2 deg and refuses it past 5, because on a constant
face this metric ran 4.2%-84.7% with head turn. That gate is measured and correct.

The REFERENCE faces it is scored against were admitted by a different rule --
`is_frontal()`, |yaw| <= 14 deg -- and the project's own measurement says a 12.9
deg frame reads 13.9% where the same face at 0.4 deg reads 84.7%. So a reference
face admitted at 12.9 deg is a badly distorted shape sitting in the pool the
subject is ranked against.

**The subject is judged to 2 degrees against references admitted at 14.** Nobody
has ever looked at what that does, because `ref_landmarks2.npz` stores landmarks
and the fetch-time pose values were discarded -- and the source photographs were
deleted afterwards to keep the privacy promise, so they cannot be re-measured.

RECOVERING POSE FROM THE CACHE, HONESTLY
----------------------------------------
Yaw leaves a signature in 2D landmarks: it compresses the far half of the face
horizontally and not the near half. The signed half-width imbalance about the
facial midline is therefore a monotone proxy for yaw.

A proxy introduced to audit a gate is worthless unless the proxy itself is
checked, so this module does that FIRST, against the only ground truth available:
the 10-frame pose burst, where FaceLandmarker's rigid-fit yaw is known per frame.
The proxy is used on the reference bank only if it tracks there, and the printed
audit states the validation numbers alongside the result rather than in a
docstring nobody re-reads.

⚠️ This module changes nothing. It measures whether a change is warranted.
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)

import fa_common as F  # noqa: E402

# Face-oval indices are all < 468, so the proxy works on the reference bank
# (468 points, no iris) and on the subject (478) unchanged.
_MIDLINE_IDX = (F.TRICHION, F.SUBNASALE, F.MENTON)


def yaw_proxy(pts):
    """Signed half-width imbalance about the facial midline, in [-1, 1].

    Positive means the +x half is wider. Zero is a face whose two halves project
    equally, which is what a frontal capture of a symmetric face gives. It is a
    PROXY: real facial asymmetry contributes to it too, which is exactly why it
    is validated against measured yaw below before being trusted on anything.
    """
    p = np.asarray(pts, float)
    mid = float(np.mean([p[i][0] for i in _MIDLINE_IDX]))
    left = mid - float(p[:468, 0].min())
    right = float(p[:468, 0].max()) - mid
    total = left + right
    return 0.0 if total < 1e-9 else float((right - left) / total)


def validate_against_burst():
    """Fit the proxy to measured yaw on the constant-face burst. Returns a dict."""
    import glob

    import fa_refine
    import pose_detect as FZ

    rows = []
    for path in sorted(
        glob.glob(os.path.join(HERE, "captures/2026-08-01_pose-burst/*.png"))
    ):
        try:
            _P, M, _wh, _img = FZ.detect_3d(path)
            _pitch, yaw, _roll = FZ.euler(M[:3, :3])
            pts, _d, _m = fa_refine.refined_landmarks(path)
            rows.append((float(yaw), yaw_proxy(pts)))
        except Exception:
            continue
    if len(rows) < 4:
        return {"usable": False, "n": len(rows)}
    yaws = np.array([r[0] for r in rows])
    prox = np.array([r[1] for r in rows])
    r = float(np.corrcoef(yaws, prox)[0, 1])
    # Fit over the range the gate actually operates in. A fit dominated by
    # +-60 deg frames would set a slope the 0-15 deg regime never sees.
    near = np.abs(yaws) <= 25.0
    slope = intercept = float("nan")
    r_near = float("nan")
    if near.sum() >= 3:
        slope, intercept = np.polyfit(prox[near], yaws[near], 1)
        r_near = float(np.corrcoef(yaws[near], prox[near])[0, 1])
    return {
        "usable": True,
        "n": len(rows),
        "r_all": r,
        "n_near": int(near.sum()),
        "r_near": r_near,
        "slope_deg_per_unit": float(slope),
        "intercept_deg": float(intercept),
        "yaw_range": (float(yaws.min()), float(yaws.max())),
    }


def main() -> int:
    print("=" * 96)
    print("REFERENCE-BANK POSE AUDIT — is the bank held to the subject's standard?")
    print("=" * 96)

    print("\n1. Validating the yaw proxy against measured yaw (constant-face burst)")
    v = validate_against_burst()
    if not v.get("usable"):
        print(f"   NOT USABLE — only {v.get('n')} frames resolved. Audit aborted.")
        return 1
    print(
        f"   n={v['n']} frames spanning {v['yaw_range'][0]:+.1f} to {v['yaw_range'][1]:+.1f} deg"
    )
    print(f"   correlation over the full range      r = {v['r_all']:+.3f}")
    print(f"   over |yaw| <= 25 deg (n={v['n_near']}), the regime the gate works in:")
    print(f"                                        r = {v['r_near']:+.3f}")
    print(
        f"   fit: yaw_deg = {v['slope_deg_per_unit']:+.1f} * proxy {v['intercept_deg']:+.1f}"
    )
    if abs(v["r_near"]) < 0.85:
        print(
            "\n   🔴 The proxy does NOT track measured yaw well enough in the gate's own"
        )
        print("      regime. It will not be applied to the reference bank — auditing a")
        print("      gate with an unvalidated instrument would manufacture a finding.")
        return 0

    print("\n2. Applying it to the cached reference bank")
    d = np.load(os.path.join(HERE, "ref_landmarks2.npz"), allow_pickle=False)
    s, b = v["slope_deg_per_unit"], v["intercept_deg"]
    print(
        f"   {'cohort':<16}{'n':>5}{'|yaw| mean':>12}{'p90':>8}{'max':>8}   share past each gate"
    )
    total_past = {}
    for cohort in ("white_10_19", "white_20_29", "broad_10_19"):
        pts = d[cohort]
        est = np.array([abs(s * yaw_proxy(p) + b) for p in pts])
        gates = {g: float((est > g).mean()) for g in (2.0, 5.0, 14.0)}
        total_past[cohort] = (len(pts), est, gates)
        print(
            f"   {cohort:<16}{len(pts):>5}{est.mean():>11.1f}°{np.percentile(est, 90):>7.1f}°"
            f"{est.max():>7.1f}°   >2°:{gates[2.0]:>5.0%}  >5°:{gates[5.0]:>5.0%}  >14°:{gates[14.0]:>5.0%}"
        )

    print()
    print("-" * 96)
    print("WHAT IT MEANS")
    print("-" * 96)
    n_all = sum(v2[0] for v2 in total_past.values())
    past5 = sum((v2[1] > 5.0).sum() for v2 in total_past.values())
    past2 = sum((v2[1] > 2.0).sum() for v2 in total_past.values())
    print(
        f"   The subject's figure is withheld past |yaw| 2° and refused past 5°.\n"
        f"   Of the {n_all} reference faces it is ranked against, an estimated\n"
        f"   {past2} ({past2 / n_all:.0%}) sit past 2° and {past5} ({past5 / n_all:.0%}) past 5°."
    )
    print(
        "\n   Tightening the bank to the subject's own gate is NOT free: it is the same\n"
        "   trade the pose gate makes on the subject, but paid in reference count, and\n"
        "   a percentile over a tiny bank is unstable in its own right. The decision\n"
        "   needs the percentile re-run at each gate, which needs analyze_averageness2\n"
        "   to accept a gate argument. That is the next step, not this file's job."
    )
    print(
        "\n   🔴 THESE SHARES ARE UPPER BOUNDS, and the reason is structural. The proxy\n"
        "   conflates head yaw with genuine facial asymmetry: a perfectly frontal but\n"
        "   asymmetric face returns a non-zero proxy. Taking |yaw| then turns that\n"
        "   scatter into POSITIVE bias, because both signs map to a larger magnitude.\n"
        "   The burst validation cannot detect this — it is ONE face, so its asymmetry\n"
        "   is a constant absorbed by the fit's intercept. Across 243 different faces\n"
        "   it is 243 different constants. So read '81% past 2°' as 'no more than 81%',\n"
        "   and treat the ORDERING (most of the bank is looser than the subject's gate)\n"
        "   as the finding rather than the percentage."
    )
    print(
        "\n   ⚠️ Estimated, not measured. The fetch-time pose values were discarded and\n"
        "   the source photographs deleted, so this proxy is the only route left. It\n"
        "   is validated above; it is not the same thing as having kept the numbers.\n"
        "   The lesson for any future fetch: KEEP THE POSE."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
