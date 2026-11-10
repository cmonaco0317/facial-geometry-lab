"""Evidence-based facial analysis: symmetry, sexual dimorphism, averageness.

Reuses the verified 478-point FaceMesh landmarks from the frontal photo. These
three constructs (unlike the neoclassical canons) have actual support in the
attractiveness literature, though all carry heavy single-2D-photo caveats.

Outputs: advanced_metrics.json, symmetry_map.png,
dimorphism_chart.png
"""

import json
import os

def _degrade(where, exc, effect_detail):
    """Record a flattering fallback. Never raises; see degradation.py."""
    try:
        import degradation as _D  # noqa: PLC0415

        _D.record(where, f"{type(exc).__name__}: {exc}", "more confident")
        _D.record(where + " [effect]", effect_detail, "more confident")
    except Exception:  # noqa: BLE001
        pass


import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import procrustes

import fa_common as F
import reference_values as rv

# The dimorphism ratios are DEFINED ONCE, in the module that measured them, and
# imported here rather than restated. A second copy of `nose width / mouth width`
# living in production is exactly how a band ends up describing a quantity the
# report no longer computes.
import dimorphism_bank_probe as DP
import dimorphism_reference as DR

HERE = os.path.dirname(os.path.abspath(__file__))

# Central (midline) landmarks for fitting the symmetry axis (roll-robust)
MIDLINE = [
    10,
    151,
    9,
    8,
    168,
    6,
    197,
    195,
    5,
    4,
    1,
    19,
    94,
    2,
    164,
    0,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    200,
    199,
    175,
    152,
]

# Bilateral landmark pairs (left, right, region) — counterparts in FaceMesh
PAIRS = [
    (55, 285, "brow"),
    (46, 276, "brow"),
    (105, 334, "brow"),
    (107, 336, "brow"),
    (65, 295, "brow"),
    (52, 282, "brow"),
    (70, 300, "brow"),
    (33, 263, "eye"),
    (133, 362, "eye"),
    (159, 386, "eye"),
    (145, 374, "eye"),
    (157, 384, "eye"),
    (173, 398, "eye"),
    (246, 466, "eye"),
    (7, 249, "eye"),
    (129, 358, "nose"),
    (98, 327, "nose"),
    (97, 326, "nose"),
    (64, 294, "nose"),
    (48, 278, "nose"),
    (219, 439, "nose"),
    (234, 454, "cheek"),
    (116, 345, "cheek"),
    (123, 352, "cheek"),
    (50, 280, "cheek"),
    (101, 330, "cheek"),
    (36, 266, "cheek"),
    (205, 425, "cheek"),
    (172, 397, "jaw"),
    (136, 365, "jaw"),
    (150, 379, "jaw"),
    (149, 378, "jaw"),
    (176, 400, "jaw"),
    (58, 288, "jaw"),
    (93, 323, "jaw"),
    (61, 291, "mouth"),
    (40, 270, "mouth"),
    (39, 269, "mouth"),
    (37, 267, "mouth"),
    (91, 321, "mouth"),
    (84, 314, "mouth"),
    (87, 317, "mouth"),
]


def load_pts():
    """symmetry / dimorphism / template-averageness are shape, so they take the fit."""
    return F.load_landmark_csv("geometry", HERE)


def load_canonical():
    verts = []
    with open(os.path.join(HERE, "canonical_face_model.obj")) as f:
        for line in f:
            if line.startswith("v "):
                _, x, y, z = line.split()[:4]
                verts.append([float(x), float(y), float(z)])
    return np.array(verts)  # (468, 3); x right, y up, z forward


# ----------------------------------------------------------------------------
# 1. SYMMETRY (midline-fit, roll-corrected, size-normalised)
# ----------------------------------------------------------------------------
# Per-metric pose tolerances. NOT one global gate — that is the whole finding.
# Measured 2026-08-01 on a 26-second burst, where the face is necessarily constant
# so every difference between frames is error. The two axes the report calls ROBUST
# both collapse, at different rates and in OPPOSITE directions: yaw inflates measured
# asymmetry and deflates measured averageness.
#
# For scale, the project's own is_frontal() reference filter allows |yaw| <= 14 deg.
# Both gates below are several times tighter, and a frame that passes is_frontal was
# measured 71 percentile points off on averageness. A reference-population filter is
# not a subject-qualification gate.
#
# Deliberately no correction anywhere here. For symmetry, two defensible fits of the
# pose term disagree 6-fold about the pose-free value (0.78 vs 4.50 %IPD from the same
# five points), so subtracting one would invent precision the data cannot support.
_POSE_GATES = {
    "symmetry": {
        "key": "pose_gate.symmetry_yaw_deg",
        "mid_status": "pose_inflated",
        "bound": "UPPER BOUND on the face's asymmetry",
        "direction": "inflates",
        "evidence": (
            "measured inflation at 4 deg was already 6.1x the lowest-yaw frame of a "
            "constant face"
        ),
        "refuse_why": (
            "the reading is dominated by head position and the far side of the face "
            "is mesh-fit rather than observed"
        ),
    },
    "profile": {
        "key": "pose_gate.profile_yaw_deg",
        # The only gate whose ideal is a BAND rather than zero. A frontal metric
        # wants |yaw| -> 0; a profile metric wants |yaw| near the 67 deg the
        # published figures were taken at, and is wrong in BOTH directions away
        # from it. Forcing it through the near-zero logic would have graded a
        # perfect 0-degree frontal as the best possible profile.
        "shape": "band",
        "floor": 50.0,
        "mid_status": "pose_shifted",
        "bound": "reading at a DIFFERENT turn angle than the published figures",
        "direction": "shifts",
        "evidence": (
            "measured slopes of -0.72 deg convexity, +0.29 nasolabial and -0.45 pp "
            "chin projection per degree of turn, all r>0.93"
        ),
        "refuse_why": (
            "below the band it is a turned frontal, and above it the far eye is gone "
            "so the landmarker cannot anchor the silhouette at all"
        ),
    },
    "averageness": {
        "key": "pose_gate.averageness_yaw_deg",
        "mid_status": "pose_deflated",
        "bound": "LOWER BOUND on the face's typicality",
        "direction": "deflates",
        "evidence": (
            "measured 84.7% at 0.4 deg falling to 62.5% at 4.5 deg and 13.9% at "
            "12.9 deg, against a published CI of about +/-10 points"
        ),
        "refuse_why": (
            "the shape being compared to the reference mean is a foreshortened "
            "projection, not the face"
        ),
    },
}



# Strangers-with-repeats required before the noise deconvolution is trusted.
DECONV_MIN_SUBJECTS = 10


def _ranked_asymmetry():
    """Asymmetry as a PERCENTILE, measured the way the bank was measured.

    ⭐ WHY THIS RE-DETECTS INSTEAD OF REUSING THE LANDMARKS ALREADY IN HAND. The
    pipeline's landmarks come from a REFINED detector (478 points, refined eye
    and lip positions); every banked face was landmarked by the plain one (468).
    Ranking a refined reading inside a distribution of plain ones is the
    same-label-two-quantities defect. §5.1's rule is that the subject is
    re-detected with the BANK'S OWN detector, so that is what happens here — the
    cost is one extra detection per run.

    ⛔ AND IT IS A SEPARATE NAME FROM THE SHIPPED FIGURE. `asymmetry_ranked` is
    single-photograph and outer-canthus normalised; the report's
    `mean_asym_pct_ipd` is iris-normalised and may be fit-derived. More precise,
    and NOT comparable to this bank.

    ⚠️ The population is pose-gated at 2 deg, because ungated it ranks how
    square-on people were photographed rather than their faces — see
    symmetry_reference.
    """
    try:
        import numpy as _np  # noqa: PLC0415

        import analyze_averageness2 as _A2  # noqa: PLC0415
        import symmetry_population as _SP  # noqa: PLC0415
        import symmetry_reference as _SR  # noqa: PLC0415

        # ⭐ EVERY IN-GATE FRONTAL, NOT ONE PHOTOGRAPH. A single reading carries
        # sigma=0.85% of IPD of viewpoint noise (measured on strangers), which a
        # tight population turns into tens of percentile points. Taking the
        # median across N in-gate frames cuts that by sqrt(N) — the same argument
        # the multi-view fit makes for every other measurement.
        # ⛔ THE SUBJECT'S OWN CURATED FRONTALS, NOT A GLOB OF captures/. The
        # first version of this globbed captures/*/*.png and swept in
        # 2026-08-14_female-multiface and 2026-08-26_second-subject-pexels —
        # OTHER PEOPLE — averaging a stranger's face into this subject's reading.
        # That is the identity-carried-over defect this project has already hit
        # twice, and it moved the answer from the 21st percentile to the 98th.
        # averageness_session_probe.CANDIDATES is the vetted list: this subject
        # only, dated, and it documents why year-old camera-roll frames are
        # excluded (real facial change would masquerade as measurement error).
        import averageness_session_probe as _ASP  # noqa: PLC0415

        cands = [os.path.join(HERE, rel) for rel, _sess in _ASP.CANDIDATES]
        lmk = _A2.make_landmarker()
        vals = []
        try:
            for c in cands:
                if not os.path.exists(c):
                    continue
                lms, _ang = _A2.landmarks_for(lmk, c)
                if lms is None:
                    continue
                if float(_np.linalg.norm(lms[33] - lms[263])) < 120.0:
                    continue
                y = _SP.yaw_degrees(lms)
                if y is None or y > _SR.YAW_GATE_DEG:
                    continue
                v = _SP.symmetry_468(lms)
                if v is not None and _np.isfinite(v):
                    vals.append(float(v))
        finally:
            lmk.close()
        if not vals:
            return {}
        n_frames = len(vals)
        val = float(_np.median(vals))
        pool = _np.asarray(_SR.VALUES, dtype=float)
        rng = _np.random.default_rng(20260904)
        n = len(pool)
        pct = 100.0 * float(_np.mean(pool > val))

        # ⛔ THE INTERVAL MUST CARRY THE SUBJECT'S OWN MOVEMENT, NOT JUST THE
        # BANK'S SAMPLING ERROR. The first version of this shipped a bootstrap
        # over the cohort alone — 19.6-23.1, about 3.5 points wide. Measured on
        # 10 strangers photographed simultaneously from ~30 calibrated cameras
        # (stranger_reproducibility.py), the SAME FACE moves sigma=0.85% of IPD
        # between two in-gate viewpoints, which maps to tens of percentile
        # points because the gated population is tight (sigma=1.31). The
        # observed percentile swing on those strangers was a median of 47
        # points. A 3.5-point interval understated it by more than 10x.
        #
        # ⭐ WHY IT IS SO SENSITIVE, which is the non-obvious part: the pose gate
        # that makes this population VALID also makes it TIGHT, and a tight
        # population converts a small measurement error into a large percentile
        # move. Validity and precision pulled in opposite directions here.
        sigma_m = 0.0
        try:
            _sr = json.load(open(os.path.join(HERE,
                                              "stranger_reproducibility.json"),
                                 encoding="utf-8"))
            g = _sr.get("by_gate", {}).get(str(_SR.YAW_GATE_DEG)) or \
                _sr.get("by_gate", {}).get(f"{_SR.YAW_GATE_DEG:g}")
            sigma_m = float((g or {}).get("sigma_measurement") or 0.0)
        except Exception:  # noqa: BLE001
            sigma_m = 0.0
        # ⭐⭐ THE POPULATION IS DECONVOLVED, because the bank's spread is not the
        # spread of FACES — it is faces PLUS the same viewpoint noise the subject
        # carries: sigma_observed^2 = sigma_true^2 + sigma_measurement^2, measured
        # at 1.31 and 0.85 giving sigma_true 1.00 (58% of the variance is real).
        # ⛔ Averaging the subject over N frames while ranking against a
        # noise-inflated bank would be Phase E again — a cleaner reading scoring
        # better than the crowd it is compared with, for being cleaner. Shrinking
        # the pool to its true spread compares like with like.
        # 🔴 DECONVOLUTION IS BUILT AND DELIBERATELY OFF, and the reason is the
        # point. Shrinking the pool to its true spread is the statistically
        # correct comparison once sigma_true is known — but sigma_true currently
        # rests on FOUR strangers with two or three viewpoints each. Applied, it
        # moved this subject from roughly the 90th percentile (raw, where he sits
        # just below the population's p10 of 1.08) to the 99.3rd. A correction
        # that strong, in the flattering direction, on that evidence base, is the
        # confident-wrong-number this project exists to refuse.
        #
        # It switches on by itself when the evidence arrives: DECONV_MIN_SUBJECTS
        # strangers with repeats. Nothing to remember, and no judgement call left
        # to a later session.
        sigma_true = 0.0
        n_strangers = 0
        try:
            sigma_true = float((g or {}).get("sigma_true") or 0.0)
            n_strangers = int((g or {}).get("n_subjects") or 0)
        except Exception:  # noqa: BLE001
            sigma_true, n_strangers = 0.0, 0
        sigma_o = float(_np.std(pool, ddof=1))
        deconv_ok = (sigma_true > 0 and sigma_o > 0
                     and n_strangers >= DECONV_MIN_SUBJECTS)
        if deconv_ok:
            shrink = min(1.0, sigma_true / sigma_o)
            centre = float(_np.mean(pool))
            pool_eff = centre + (pool - centre) * shrink
        else:
            shrink, pool_eff = 1.0, pool
        # ⭐ THE SUBJECT'S OWN SPREAD IS PREFERRED, AND THE STRANGER FIGURE IS A
        # FLOOR. These candidates span several sittings on different dates, so
        # their spread includes lighting, camera and expression — everything the
        # simultaneous-camera stranger figure holds constant. Using the stranger
        # sigma alone would understate a cross-sitting reading. ⛔ Never narrower
        # than the stranger floor, because a subject who happens to be steady
        # across three frames has not proved the instrument is.
        sigma_subject = (float(_np.std(vals, ddof=1)) if len(vals) >= 3 else 0.0)
        sigma_per_frame = max(sigma_m, sigma_subject)
        sigma_eff = (sigma_per_frame / (n_frames ** 0.5)
                     if sigma_per_frame > 0 else 0.0)
        pct = 100.0 * float(_np.mean(pool_eff > val))
        draws = []
        for _ in range(2000):
            v = val + (rng.normal(0.0, sigma_eff) if sigma_eff > 0 else 0.0)
            draws.append(100.0 * float(
                _np.mean(pool_eff[rng.integers(0, n, n)] > v)))
        lo, hi = float(_np.percentile(draws, 2.5)), float(_np.percentile(draws, 97.5))
        out = {
            "asymmetry_ranked": round(float(val), 3),
            "asymmetry_ranked_units": "% of outer-canthal distance, single "
                                      "photograph, bank's own plain detector",
            "pct_of_cohort_more_asymmetric": round(pct, 1),
            "pct_ci": [round(lo, 1), round(hi, 1)],
            "population_n": int(n),
            "subject_frames_used": n_frames,
            "subject_movement_sigma_pct_ipd": round(sigma_per_frame, 3),
            "sigma_from_strangers_floor": round(sigma_m, 3),
            "sigma_from_this_subject": round(sigma_subject, 3),
            "subject_effective_sigma": round(sigma_eff, 3),
            "population_deconvolved_shrink": round(shrink, 3),
            "deconvolution_applied": bool(deconv_ok),
            "deconvolution_needs_n_strangers": DECONV_MIN_SUBJECTS,
            "deconvolution_has_n_strangers": n_strangers,
            "deconvolution_note": (
                "the bank's spread is faces PLUS the same viewpoint noise the "
                "subject carries, so shrinking it to sigma_true is the correct "
                "comparison — but on 4 strangers it moved this reading from "
                "~90th to 99.3rd percentile, which is too strong a flattering "
                "correction for that evidence. Switches on at "
                f"{DECONV_MIN_SUBJECTS} strangers with repeats."),
            "interval_sources": ["cohort sampling (bootstrap over faces)",
                                 "subject movement measured on 10 strangers "
                                 "across calibrated viewpoints"],
            "population_yaw_gate_deg": _SR.YAW_GATE_DEG,
            "population_note": (
                "pose-gated: ungated, this bank ranks how square-on people were "
                "photographed, not their faces — one unchanging face reads 0.96% "
                "at 0.4 deg of yaw and 202% at 58 deg"),
        }
        try:
            import run_pipeline as _RP  # noqa: PLC0415

            if _RP.frontal_is_set_derived():
                out["ranked_on_a_set_derived_frontal"] = True
                out["population_note"] += (
                    ". ⚠️ THIS frontal.png is a set run's chosen reference frame, "
                    "so the ranked figure describes that frame rather than a "
                    "casually taken photograph")
        except Exception:  # noqa: BLE001
            pass
        return out
    except Exception as exc:  # noqa: BLE001
        _degrade("analyze_advanced._ranked_asymmetry", exc,
                 "no symmetry percentile; the measured asymmetry is unaffected")
        return {}

def pose_condition(yaw_deg, metric="symmetry"):
    """Grade whether a reading taken at this yaw is about the face or the camera.

    `metric` selects the measured tolerance — they are not interchangeable, and
    assuming they are is the error this function exists to prevent.
    """
    spec = _POSE_GATES[metric]
    if yaw_deg is None:
        return {
            "metric": metric,
            "yaw_deg": None,
            "status": "unknown",
            "note": (
                "Head yaw was not supplied, so this reading carries no pose "
                "condition. It is not established to be a statement about the face."
            ),
        }
    lo_or_max, hi = rv.require(spec["key"]).value
    y = abs(float(yaw_deg))
    if spec.get("shape") == "band":
        floor = spec["floor"]
        if lo_or_max <= y <= hi:
            status, note = "measured", (
                f"|yaw| {y:.1f} deg is inside the {lo_or_max:g}-{hi:g} deg band the "
                "published figures were taken at, so the angle term is small."
            )
        elif floor <= y < lo_or_max:
            status, note = spec["mid_status"], (
                f"|yaw| {y:.1f} deg is turned LESS than the {lo_or_max:g}-{hi:g} deg "
                "band these figures were published from. The values are usable but "
                f"NOT comparable to them — {spec['evidence']}."
            )
        else:
            status, note = "refused", (
                f"|yaw| {y:.1f} deg is outside {floor:g}-{hi:g} deg. At this angle "
                f"{spec['refuse_why']}. Not reportable."
            )
        return {
            "metric": metric,
            "yaw_deg": round(float(yaw_deg), 2),
            "status": status,
            "note": note,
        }
    measured_max, mid_max = lo_or_max, hi
    if y <= measured_max:
        status, note = "measured", (
            f"|yaw| {y:.1f} deg is within the {measured_max:g} deg band where the "
            "measured pose contribution is small next to this metric's own error bar."
        )
    elif y <= mid_max:
        status, note = spec["mid_status"], (
            f"|yaw| {y:.1f} deg exceeds {measured_max:g} deg, and yaw "
            f"{spec['direction']} this metric. Treat the value as a {spec['bound']}, "
            f"not an estimate of it — {spec['evidence']}."
        )
    else:
        status, note = "refused", (
            f"|yaw| {y:.1f} deg exceeds {mid_max:g} deg. At this angle "
            f"{spec['refuse_why']}. Not reportable as a face measurement."
        )
    return {
        "metric": metric,
        "yaw_deg": round(float(yaw_deg), 2),
        "status": status,
        "note": note,
    }


def symmetry(pts, yaw_deg=None):
    mid = pts[MIDLINE]
    c = mid.mean(axis=0)
    # principal axis of the midline points = symmetry axis
    u, s, vt = np.linalg.svd(mid - c)
    axis = vt[0]
    # orient axis to point downward (trichion->menton)
    if np.dot(pts[152] - pts[10], axis) < 0:
        axis = -axis
    perp = np.array([-axis[1], axis[0]])  # rightward normal
    if perp[0] < 0:
        perp = -perp
    roll = np.degrees(np.arctan2(axis[0], axis[1]))  # deviation from vertical

    def coords(p):
        d = pts[p] - c
        return np.dot(d, axis), np.dot(d, perp)  # (s along axis, t across)

    ipd = np.linalg.norm(pts[F.L_IRIS] - pts[F.R_IRIS])
    rows, tL_all, tR_all = [], [], []
    for li, ri, reg in PAIRS:
        sL, tL = coords(li)
        sR, tR = coords(ri)
        asym = np.hypot(sL - sR, tL + tR) / ipd * 100.0  # mirror expects tR = -tL
        rows.append(
            {
                "pair": f"{li}-{ri}",
                "region": reg,
                "asym_pct_ipd": round(asym, 2),
                "vertical_px": round(sL - sR, 1),
            }
        )
        tL_all.append(tL)
        tR_all.append(-tR)
    vals = np.array([r["asym_pct_ipd"] for r in rows])
    regions = {}
    for reg in sorted(set(r["region"] for r in rows)):
        rv = [r["asym_pct_ipd"] for r in rows if r["region"] == reg]
        regions[reg] = round(float(np.mean(rv)), 2)
    # yaw confounder: half-width balance + eye-width ratio
    halfL, halfR = np.mean(np.abs(tL_all)), np.mean(np.abs(tR_all))
    yaw_balance = round(100 * abs(halfL - halfR) / ((halfL + halfR) / 2), 1)
    ew_ratio = F.hdist(pts, F.L_CANTHUS_LAT, F.L_CANTHUS_MED) / F.hdist(
        pts, F.R_CANTHUS_MED, F.R_CANTHUS_LAT
    )
    return {
        "roll_deg": round(roll, 2),
        "pose_condition": pose_condition(yaw_deg, "symmetry"),
        "mean_asym_pct_ipd": round(float(np.mean(vals)), 2),
        "median_asym_pct_ipd": round(float(np.median(vals)), 2),
        "max_asym_pct_ipd": round(float(np.max(vals)), 2),
        "by_region": regions,
        "yaw_balance_pct": yaw_balance,
        "eye_width_ratio": round(ew_ratio, 3),
        "pairs": rows,
        "_axis": axis.tolist(),
        "_perp": perp.tolist(),
        "_c": c.tolist(),
        "_ipd": float(ipd),
    }


# ----------------------------------------------------------------------------
# 2. SEXUAL DIMORPHISM (sex-typicality of documented dimorphic features)
# ----------------------------------------------------------------------------
def _percentile_of(val, pcts):
    """Where this value sits among the reference faces, 0-100."""
    import bisect

    i = bisect.bisect_left(pcts, val)
    if i <= 0:
        return 0.0
    if i >= len(pcts):
        return 100.0
    lo, hi = pcts[i - 1], pcts[i]
    frac = 0.0 if hi == lo else (val - lo) / (hi - lo)
    return round((i - 1) + frac, 1)


def movement_bracket(name, val, lo, hi):
    """(bracket_lo, bracket_hi, resolves, half_width) for one dimorphism feature.

    MODULE-LEVEL AND NOT A CLOSURE, so it can be probed with crafted values at
    and around each band edge instead of only through a whole pipeline run. The
    lesson of the retired `philtrum-verdict-refuses`: a guard that can only be
    exercised by running one photograph ends up asserting that photograph's
    reading, and then inverts when a different capture is on disk.

    THE MARGIN IS THE FULL MEASURED RANGE, NOT HALF OF IT. Half-range asks "could
    the CENTRE of this feature's wander be on the other side of the edge", which
    lets a reading sitting at the top of its own observed range clear the edge and
    assert a side. The question a reader actually needs answered is "could ANOTHER
    capture of this face read on the other side", and the measured answer to that
    is the full range.

    ⚠️ RE-KEYED 2026-08-19 when the ratios were rebuilt. The margins are measured
    per RATIO, so a feature whose formula changes cannot keep its old margin --
    applying a margin measured on `nose width / bizygomatic width` to
    `nose width / mouth width` would be a number describing a different quantity.
    `name` is now the stable feature id from dimorphism_reference, not the printed
    label, so a relabelling cannot silently repoint a margin either.
    """
    ref = DR.FEATURES.get(name)
    half = None if ref is None else ref.get("movement")
    # ⭐ PER-SUBJECT WIDENING. The measured margin belongs to the reference
    # subject; this scales it by how much more THIS person's own photographs
    # disagree under one identical operation (`subject_movement.py`). It only
    # ever widens -- the scalar is clamped at 1.0 -- so a subject whose set
    # cannot be split, or who is steadier than the reference, gets exactly the
    # margin that shipped before and the report says the bar is borrowed.
    if half is not None:
        try:
            import subject_movement as _SM  # noqa: PLC0415

            half = half * _SM.scalar()
        except Exception:  # noqa: BLE001 - no scalar is the shipped behaviour
            pass
    if half is None or lo is None or hi is None:
        # An unmeasured feature RESOLVES rather than refusing. Refusing by
        # default would silence every feature the moment someone adds one, which
        # reads as caution and behaves as a bug.
        return None, None, True, None
    b_lo, b_hi = val - half, val + half
    straddles = (b_lo <= lo <= b_hi) or (b_lo <= hi <= b_hi)
    return round(b_lo, 4), round(b_hi, 4), (not straddles), round(half, 4)


# --------------------------------------------------- race-matched sex evidence
# ⭐ THE BANDS WERE VALIDATED ON POOLED GROUPS AND NEVER PER RACE. Split per race
# on 2026-08-23, White scored 5 of 5 features established and every other group
# 1-4 -- and the cause is SAMPLE SIZE, not effect size. White's own AUCs
# (0.58-0.66) are the WEAKEST of any group; it had 460 faces to prove them with
# against 56-75 for everyone else.
#
# ⛔ AND THE TWO REASONS A LEAN CAN FAIL ARE NOT THE SAME SENTENCE. "This
# measurement is wrong for people like you" and "we have too few faces like
# yours to know" are different claims about a person, and only the first is
# about them at all. race_reference records which one applies.

def _probability_fields(fid, val, half, lo, hi, rd):
    """The sex lean as a probability, ADDED BESIDE the label, never replacing it.

    ⭐ WHY THIS IS SAFE TO ADD. The neutral band is computed as
    off = ln(LEAN_ODDS)*sd^2/delta around the midpoint of the male and female
    means, which under the equal-variance Gaussian model that computation already
    assumes is exactly the contour where log-odds(female|x) = +/- ln(LEAN_ODDS).
    So the band edge IS a 60% probability and this recovers a number the code
    already had rather than introducing a model. Verified on all 10 shipped band
    edges by dimorphism_probability.selfcheck.

    ⛔ THE LABEL ABOVE IS UNCHANGED, INCLUDING ITS REFUSAL. A reader who gets
    "unresolved" now also gets a probability and its interval, so the refusal
    stops being an absence — but nothing that used to resolve resolves
    differently, and no claim about the label's behaviour is weakened.
    """
    try:
        import dimorphism_probability as _DP  # noqa: PLC0415

        band = [lo, hi] if lo is not None and hi is not None else None
        r = _DP.probability(fid, val, half=half,
                            auc=(rd or {}).get("auc"),
                            auc_ci=(rd or {}).get("ci"), cohort_band=band)
        if not r:
            return {}
        return {"p_female": r["p_female"], "p_female_ci": r["ci"],
                "p_female_ci_width": r["width"],
                "p_female_interval_sources": r["interval_sources"]}
    except Exception as exc:  # noqa: BLE001
        # Neutral, not flattering: losing this removes an ADDED number and
        # leaves the label exactly as it was.
        _degrade("analyze_advanced._probability_fields", exc,
                 "no probability printed beside the sex lean; the label is "
                 "unaffected")
        return {}

def race_evidence(fid):
    """(verdict, detail) for this feature on the subject's race, or (None, why)."""
    try:
        import bank_reference as RR
        import subject as SUBJ
    except Exception as exc:  # noqa: BLE001
        # ⛔ The race gate can only ever REFUSE, so losing it can only ever
        # ASSERT: without it, "contradicted" and "underpowered" never fire and a
        # lean prints that would otherwise have been withheld.
        _degrade("analyze_advanced.race_evidence", exc,
                 "no race gate, so no lean is refused for cohort evidence")
        return None, None
    # ⛔ THE TRY ABOVE USED TO COVER ONLY THE IMPORT, and everything below ran
    # unprotected. Found 2026-09-04 by selfcheck_degradation injecting a
    # reference module that imports cleanly and raises on USE: the documented
    # fallback did not fire, the stage CRASHED, and a malformed reference took
    # down a report instead of producing one with a stated limitation. Widening
    # the guard is only safe because the degradation is now recorded — an
    # unrecorded broad except is how a silent flattering failure gets in.
    try:
        cid, why = SUBJ.race_cohort()
        if cid is None:
            return None, why
        # cid is `<sex>_<race>_<band>`; the evidence table is keyed
        # `<race>_<band>` because a separation proven on 20-29 faces is not
        # evidence about 40-49 faces of the same group.
        key = cid[2:]
        per = (RR.DIMORPHISM_BY_RACE or {}).get(key) or {}
        d = per.get(fid)
        if not d:
            pretty = key.replace("_", " ")
            return None, (f"the {pretty} reference bank does not carry this "
                          f"feature")
        return d["verdict"], d
    except Exception as exc:  # noqa: BLE001
        _degrade("analyze_advanced.race_evidence", exc,
                 "race evidence unreadable, so no lean is refused for cohort "
                 "evidence")
        return None, None


def dimorphism(pts, thirds_lower_pct=None):
    """Per-feature sex-typicality, against bands MEASURED on real faces.

    ⭐ REBUILT 2026-08-19, and the rebuild is the result. Every band used to be a
    population midpoint lifted from published anthropometry whose landmark
    definitions are not this pipeline's. Measured against 825 real reference
    faces of known sex, four of the seven features could not tell a man from a
    woman at all, and one -- `nose width / bizygomatic width` -- separated them
    at AUC 0.445, i.e. BACKWARDS, because the bizygomatic denominator is itself
    male-larger and more than cancels the numerator's real dimorphism. The
    brow-to-eyelid band sat at 0.50-0.60 against a real population median of
    0.291 and so labelled 93% of all 825 people, men and women alike, masculine.

    ⛔ THE REPAIR WAS NOT TO MOVE THE BANDS ONTO THE POPULATION MEDIAN. That
    manufactures a 50/50 verdict from a feature carrying no signal. It was to
    re-normalise each ratio so it carries the signal the literature describes --
    the failure mode was denominators that cancel it -- and then to measure
    whether it does, on faces the choice was not made on. See
    dimorphism_bank_probe for the method and dimorphism_reference for the result.

    thirds_lower_pct is accepted and ignored: lower-face height is now measured
    directly from the landmarks rather than passed in as a percentage, so the
    caller cannot supply a value inconsistent with the mesh the other ratios use.
    """
    m = DP.measures(pts)

    out = []
    for fid, label, fn, sex_high in DP.FEATURES:
        ref = DR.FEATURES[fid]
        val = float(fn(m))
        band = ref["band"]
        lo, hi = (None, None) if not band else (band[0], band[1])

        if not ref["separates"]:
            # ⭐ NOT A REFUSAL, AND THE DIFFERENCE MATTERS. The measurement is
            # printed, with its position among real faces. What is withheld is
            # the SEX LABEL, because this instrument has measured that it cannot
            # support one for this ratio. Refusing a value it can measure would
            # be the gate failing; asserting a label it cannot measure would be
            # the confident wrong number this whole project exists to avoid.
            lean, lean_point, resolves, half = None, None, True, None
            b_lo = b_hi = None
            rv, rd, race_reason, cohort_band_used = None, None, None, None
            reason = (
                f"no sex lean is printed for this feature: measured on "
                f"{DR.REFERENCE_LABEL} of known sex it "
                f"separates male from female at AUC {ref['auc_holdout']} on "
                f"held-out faces, where 0.5 is a coin flip"
            )
        else:
            fem_high = ref["female_mean"] > ref["male_mean"]
            # The measured direction must agree with the direction the
            # literature asserts, or the feature would not have been adopted;
            # asserting it here means a regenerated reference that flipped could
            # never quietly invert a printed verdict.
            assert fem_high == (sex_high == "F"), (
                f"{fid}: measured direction disagrees with the asserted one"
            )
            if lo <= val <= hi:
                lean_point = "neutral"
            elif (val > hi) == fem_high:
                lean_point = "feminine"
            else:
                lean_point = "masculine"
            # ⭐ USE THE COHORT'S OWN NEUTRAL BAND when the feature is proven for
            # them. The band decides the LABEL, and one fitted on 10-29 faces
            # sits a median 0.39 SD off centre for a 40-49 cohort — enough to
            # call a neutral face masculine while every AUC still looks healthy.
            # ⛔ ONLY WHEN ESTABLISHED. On an unproven feature the lean is
            # withheld anyway, and those are exactly the cohorts whose band the
            # formula blows up on (median 2.0x wider, worst 5.2x) because it
            # divides by a male-female gap that is near zero.
            _rv0, _rd0 = race_evidence(fid)
            cohort_band_used = None
            if _rv0 == "established" and (_rd0 or {}).get("cohort_band"):
                lo, hi = _rd0["cohort_band"]
                cohort_band_used = [lo, hi]
                if lo <= val <= hi:
                    lean_point = "neutral"
                elif (val > hi) == fem_high:
                    lean_point = "feminine"
                else:
                    lean_point = "masculine"
            b_lo, b_hi, resolves, half = movement_bracket(fid, val, lo, hi)
            lean = lean_point if resolves else "unresolved"
            # THE RACE GATE, applied after the movement bracket and before the
            # label is allowed to stand. It can only ever REFUSE -- same
            # contract as the movement margin, so it cannot pick a side.
            rv, rd = race_evidence(fid)
            race_reason = None
            if rv == "contradicted":
                lean = "unresolved"
                race_reason = (
                    f"this feature does not tell the sexes apart for this "
                    f"subject's reference group -- measured AUC "
                    f"{rd['auc']:.2f} (95% CI {rd['ci'][0]:.2f}-{rd['ci'][1]:.2f}) on "
                    f"{rd['n_male']} male and {rd['n_female']} female faces of "
                    f"that group, where 0.5 is a coin flip and below 0.5 points "
                    f"the wrong way"
                )
            elif rv == "underpowered":
                lean = "unresolved"
                race_reason = (
                    f"⚠️ THIS IS A GAP IN THE REFERENCE BANK, NOT A STATEMENT "
                    f"ABOUT THIS FACE. Measured on the {rd['n_male']} male and "
                    f"{rd['n_female']} female faces this instrument holds for "
                    f"the subject's reference group, the separation is "
                    f"AUC {rd['auc']:.2f} with a 95% interval of "
                    f"{rd['ci'][0]:.2f}-{rd['ci'][1]:.2f} that still contains 0.5 -- so "
                    f"whether it works for this group is unknown rather than "
                    f"known to be poor. More faces of that group would settle it"
                )
            reason = (
                None
                if resolves
                else (
                    f"measured movement of this feature on one unchanged face "
                    f"(+/-{half}) crosses its own band edge, so the side is a "
                    f"property of the photograph rather than the face"
                )
            )

        out.append(
            {
                "metric": label,
                "feature_id": fid,
                "value": round(val, 3),
                "percentile_of_reference": _percentile_of(val, ref["percentiles"]),
                "neutral_low": lo,
                "neutral_high": hi,
                "approx_neutral": None if not band else round((lo + hi) / 2, 3),
                "anchor_kind": "measured_likelihood_band" if band else "no_band",
                "masculine_dir": "high" if sex_high == "M" else "low",
                "separates_the_sexes": ref["separates"],
                "auc_holdout": ref["auc_holdout"],
                "auc_select": ref["auc_select"],
                "auc_by_pairing": ref["auc_by_pairing"],
                "lean": lean,
                "lean_at_point_value": lean_point,
                "lean_resolves": resolves,
                "movement_bracket": None if b_lo is None else [b_lo, b_hi],
                "movement_half_width": half,
                # ⭐ ADDITIVE. The label and its refusal are untouched above; this
                # prints the probability the band was computed FROM, so a reader
                # who gets "unresolved" still gets a number and an interval.
                **_probability_fields(fid, val, half, lo, hi, rd),
                # ⛔ BOTH REASONS, GROUP-LEVEL FIRST — never `reason or race_reason`.
                # That version let the MOVEMENT refusal mask the race one, so a
                # subject whose feature is CONTRADICTED for their group was told
                # only that this photograph's bracket straddles the band edge.
                # They would re-shoot, clear the movement refusal, and still get
                # nothing — never learning the feature points the wrong way for
                # people like them. Capture-level problems are fixable by the
                # reader; group-level ones are not, so the group-level one leads.
                "lean_refusal_reason": (
                    " Also: ".join([x for x in (race_reason, reason) if x])
                    or None
                ),
                "cohort_band_used": cohort_band_used,
                "race_evidence": (
                    None if rv is None else {"verdict": rv, "auc": rd["auc"],
                                             "ci": rd["ci"], "n_male": rd["n_male"],
                                             "n_female": rd["n_female"]}
                ),
                "note": ref["why"],
                "evidence_strength": (
                    f"MEASURED on {DR.REFERENCE_LABEL} of known sex: "
                    f"held-out AUC {ref['auc_holdout']} "
                    f"({'carries a lean' if ref['separates'] else 'NO LEAN — cannot distinguish the sexes'})"
                ),
                "source": ref["source"],
                "cluster": next(c for c in DR.CLUSTERS if fid in c),
            }
        )

    # ⭐ THE TALLY COUNTS SIGNALS, NOT ROWS. Measured on the same 825 faces,
    # face width / eye spacing and jaw width / eye spacing correlate at r=0.83
    # and both anti-correlate with eye width / face width -- they are one
    # underlying thing, face width relative to inter-ocular scale. Counting them
    # as three votes was one signal counted three times, in a sentence that reads
    # as "three separate features agree".
    by_id = {r["feature_id"]: r for r in out}
    cluster_votes = {}
    for cl in DR.CLUSTERS:
        key = "+".join(sorted(cl))
        leans = [by_id[f]["lean"] for f in cl]
        sides = {l for l in leans if l in ("masculine", "feminine")}
        if len(sides) > 1:
            # Members of one cluster landing on OPPOSITE sides is a refusal to
            # assert, not a vote. "neutral" is not an opposing side -- a group
            # where one member leans and the rest sit inside their band still
            # points one way, weakly, and saying "split" there would invent a
            # disagreement that the readings do not contain.
            cluster_votes[key] = "split"
        elif sides:
            cluster_votes[key] = sides.pop()
        elif "unresolved" in leans:
            cluster_votes[key] = "unresolved"
        elif "neutral" in leans:
            cluster_votes[key] = "neutral"
        else:
            cluster_votes[key] = None
    counts = {
        lean: sum(1 for v in cluster_votes.values() if v == lean)
        for lean in ("masculine", "neutral", "feminine", "unresolved", "split")
    }
    n_signals = len([c for c in DR.CLUSTERS if cluster_votes["+".join(sorted(c))]])
    n_leaning = counts["masculine"] + counts["feminine"]

    tally = (
        f"{counts['masculine']} of {n_signals} independent signals lean masculine, "
        f"{counts['feminine']} lean feminine, {counts['neutral']} sit inside their "
        f"measured neutral band"
        + (
            f", and {counts['unresolved']} cannot be resolved — their measured "
            f"movement on an unchanged face crosses their own band edge"
            if counts["unresolved"]
            else ""
        )
        + (f", and {counts['split']} is split across features that move together"
           if counts["split"] else "")
    )

    no_lean = [r["metric"] for r in out if not r["separates_the_sexes"]]

    return {
        "metrics": out,
        "lean_counts": {
            **counts,
            "n_features": len(out),
            "n_independent_signals": n_signals,
            "n_leaning": n_leaning,
        },
        "cluster_votes": cluster_votes,
        "lean_tally": tally,
        "features_without_a_lean": no_lean,
        "n_reference_faces": DR.N_REFERENCE_FACES,
        # ⭐ THE CLASSICAL PROPORTIONS, KEPT AND PUBLISHED SEPARATELY. The
        # sex-typicality ratios above deliberately normalise against eye spacing,
        # because bizygomatic denominators are themselves male-larger and cancel
        # the signal. But the anatomy map and the levers name these proportions
        # by their classical denominators, and reading a differently-normalised
        # value under a classical name prints one ratio under another's label.
        # Computed ONCE here, where the landmarks are, rather than re-derived in
        # each consumer.
        "descriptive_ratios": {
            # classical fWHR: bizygomatic width over BROW-TO-UPPER-LIP height.
            # Not brow-to-hairline -- a different quantity that this file briefly
            # conflated with it.
            "fwhr_bizygo_over_upperlip": round(float(m["bizygo"]) / max(float(m["fwhr_h"]), 1e-9), 3),
            "bigonial_over_bizygomatic": round(float(m["jaw_w"]) / max(float(m["bizygo"]), 1e-9), 3),
            "ala_over_bizygomatic": round(float(m["nose_w"]) / max(float(m["bizygo"]), 1e-9), 3),
            "_note": (
                "descriptive proportions, NOT sex-typicality features. Each was "
                "measured against real faces of known sex and could not "
                "distinguish them under this instrument; they are printed as "
                "shape descriptions and carry no masculine/feminine reading."
            ),
        },
        "counts_note": (
            "lean_counts counts INDEPENDENT SIGNALS, not table rows. Measured on "
            f"{DR.REFERENCE_LABEL}, some of these ratios move together "
            "— face width and jaw width relative to eye spacing correlate at r=0.83 "
            "— so each correlated group casts one vote. It remains a COUNT and not a "
            "score: no net total is emitted, because the signals are not equally "
            "weighted even after grouping."
        ),
        "anchors_note": (
            "⭐ Each neutral band here is MEASURED, not borrowed. It is the range "
            f"in which this pipeline's own reading of {DR.REFERENCE_LABEL} "
            "of known sex cannot say which sex a value came from — "
            "specifically, the range inside which neither sex is "
            f"{DR.LEAN_ODDS}x more likely than the other. Every band was fixed on "
            "the 10–19 reference banks and then checked on the disjoint 20–29 "
            "banks, and on four POOLED race/age pairings. ⚠️ POOLED IS NOT "
            "PER-RACE, and saying otherwise is what this sentence used to do: "
            "measured per race on 2026-08-23 against 1,633 race-matched faces, "
            "each feature is separately marked established, unproven or "
            "contradicted for the subject's own group, and the lean is withheld "
            "unless it was established there. "
            "⛔ THIS REPLACES the published-anthropometry midpoints used until "
            "2026-08-19, and the replacement was forced by measurement: against "
            "these same faces, four of the seven old features could not "
            "distinguish men from women at all, and nose width / bizygomatic "
            "width separated them BACKWARDS at AUC 0.445 because its denominator "
            "is itself male-larger. The old brow band sat at 0.50–0.60 against a "
            "real population median of 0.291, so it called 93% of all 825 people "
            "— men and women alike — masculine. Two features still carry no lean, "
            "and their VALUES are printed anyway with their position among real "
            "faces: withholding a label this instrument cannot support is not the "
            "same as going quiet about a measurement it can make. "
            "⚠️ Male and female distributions overlap heavily even on the features "
            "that do separate — the best of them reaches AUC 0.67, so a lean is a "
            "population tendency and never a classification. The reads are "
            "NON-SUMMABLE: no net score and no overall masculine/feminine verdict "
            "is emitted. The per-feature directions are the finding; there is no "
            "total to read."
        ),
    }


# ----------------------------------------------------------------------------
# 3. AVERAGENESS (Procrustes deviation from the canonical average-face template)
# ----------------------------------------------------------------------------
def averageness(pts, canon):
    subject = pts[:468].copy()
    can = canon[:, :2].copy()
    can[:, 1] = -can[:, 1]  # flip y (canonical y-up -> image y-down)
    mtx1, mtx2, disparity = procrustes(can, subject)
    resid = np.linalg.norm(mtx1 - mtx2, axis=1)  # per-landmark, standardized units
    region_idx = {
        "brow/forehead": [10, 151, 9, 8, 107, 336, 105, 334, 66, 296, 55, 285],
        "eyes": [33, 263, 133, 362, 159, 386, 145, 374, 468, 473],
        "nose": [1, 4, 2, 6, 168, 129, 358, 98, 327, 5, 195],
        "mouth/lips": [0, 17, 13, 14, 61, 291, 40, 270, 84, 314],
        "cheeks": [234, 454, 116, 345, 50, 280, 101, 330, 205, 425],
        "jaw/chin": [152, 172, 397, 136, 365, 150, 379, 200, 199, 175],
    }
    by_region = {
        k: round(float(np.mean([resid[i] for i in v if i < 468])), 4)
        for k, v in region_idx.items()
    }
    return {
        "procrustes_disparity": round(float(disparity), 5),
        "rms_residual": round(float(np.sqrt(np.mean(resid**2))), 5),
        "by_region": dict(sorted(by_region.items(), key=lambda kv: -kv[1])),
        "_resid": resid.tolist(),
    }


# ----------------------------------------------------------------------------
# Visualisations
# ----------------------------------------------------------------------------
def draw_symmetry(pts, sym):
    img = cv2.imread(os.path.join(HERE, "frontal.png"))
    c = np.array(sym["_c"])
    axis = np.array(sym["_axis"])
    p0 = (c - axis * 1400).astype(int)
    p1 = (c + axis * 1400).astype(int)
    cv2.line(img, tuple(p0), tuple(p1), F.BLUE, 3, cv2.LINE_AA)  # symmetry axis
    vmax = max(r["asym_pct_ipd"] for r in sym["pairs"])
    for (li, ri, reg), row in zip(PAIRS, sym["pairs"]):
        a = row["asym_pct_ipd"]
        frac = min(a / 8.0, 1.0)  # 8% IPD -> full red
        col = (int(60), int(200 * (1 - frac)), int(60 + 195 * frac))
        L, R = pts[li].astype(int), pts[ri].astype(int)
        cv2.line(img, tuple(L), tuple(R), col, 2, cv2.LINE_AA)
        for P in (L, R):
            cv2.circle(img, tuple(P), 5, col, -1, cv2.LINE_AA)
    F.label(
        img, "Symmetry map (roll-corrected)", (40, 70), F.WHITE, 1.2, 3, bg=(0, 0, 0)
    )
    F.label(
        img,
        f"mean {sym['mean_asym_pct_ipd']:.1f}% of IPD  |  roll {sym['roll_deg']:+.1f}deg  |  yaw bal {sym['yaw_balance_pct']:.0f}%",
        (40, 125),
        F.WHITE,
        0.8,
        2,
        bg=(0, 0, 0),
    )
    F.label(
        img,
        "green=symmetric  red=asymmetric",
        (40, 175),
        (180, 180, 180),
        0.7,
        2,
        bg=(0, 0, 0),
    )
    cv2.imwrite(os.path.join(HERE, "symmetry_map.png"), img)


# draw_averageness() was removed: it rendered averageness_map.png (~6.4 MB per
# run) which build_report.py never embedded and nothing else read. The
# averageness NUMBERS are unaffected -- Part C3 comes from analyze_averageness2.py,
# and this module's template-averageness figures are still emitted to JSON.


def draw_dimorphism(dim):
    rows = dim["metrics"]
    labels = [r["metric"].split("(")[0].strip() for r in rows]
    # Range design, not a point threshold: a value INSIDE its own neutral band has no
    # lean and plots at exactly 0. Bars show only how far OUTSIDE the band a value
    # sits, in band half-widths, so the only cutoff drawn is the same neutral band
    # that produced r["lean"] — no second (or third) unrelated threshold.
    excess = []
    for r in rows:
        # A feature with NO BAND carries no sex lean at all -- it is measured not
        # to separate the sexes -- so it plots at exactly 0 rather than being
        # given a bar against a band that does not exist.
        if r["neutral_low"] is None:
            excess.append(0.0)
            continue
        half = max((r["neutral_high"] - r["neutral_low"]) / 2.0, 1e-9)
        if r["value"] > r["neutral_high"]:
            e = (r["value"] - r["neutral_high"]) / half
        elif r["value"] < r["neutral_low"]:
            e = (r["value"] - r["neutral_low"]) / half
        else:
            e = 0.0
        excess.append(e if r["masculine_dir"] == "high" else -e)
    # UNRESOLVED GETS ITS OWN COLOUR. Falling through to the neutral grey would
    # draw "we cannot tell" and "it sits inside its band" identically, which is
    # the whole distinction the bracket was added to make.
    # UNRESOLVED and NO-LEAN get their own colours. Falling through to the
    # neutral grey would draw three different statements identically: "it sits
    # inside its band", "we cannot tell from this photograph", and "this ratio
    # cannot tell the sexes apart at all".
    colors = [
        (
            "#bdbdbd"
            if r["neutral_low"] is None
            else "#f57c00"
            if r["lean"] == "unresolved"
            else (
                "#1565c0"
                if r["lean"] == "masculine"
                else "#c2185b" if r["lean"] == "feminine" else "#9e9e9e"
            )
        )
        for r in rows
    ]
    fig, ax = plt.subplots(figsize=(12, 6))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, excess, color=colors, edgecolor="black", linewidth=0.6)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(
        "<-- feminine direction        distance OUTSIDE the neutral band "
        "(band half-widths; 0 = inside its own band, no lean)        masculine direction -->",
        fontsize=9,
    )
    c = dim["lean_counts"]
    ax.set_title(
        "Sexual dimorphism — per-feature sex-typicality  "
        f"({c['masculine']} masculine / {c['neutral']} neutral / {c['feminine']} "
        "feminine — a count, not a score)",
        fontweight="bold",
        fontsize=11,
    )
    m = max(1.0, max(abs(e) for e in excess) * 1.35)
    pad = 0.035 * m
    # zero-length bars would be invisible: mark in-band features explicitly at 0
    zero_y = [y[i] for i, e in enumerate(excess) if e == 0.0]
    if zero_y:
        ax.plot(
            [0] * len(zero_y), zero_y, "D", color="#9e9e9e", ms=5, mec="black", mew=0.6
        )
    for yi, (e, r) in enumerate(zip(excess, rows)):
        suffix = "  (in band)" if e == 0.0 else ""
        ax.text(
            e + (pad if e >= 0 else -pad),
            y[yi],
            f"{r['value']:.2f}{suffix}",
            va="center",
            ha="left" if e >= 0 else "right",
            fontsize=8,
        )
    ax.set_xlim(-m, m)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "dimorphism_chart.png"), dpi=130)


def main():
    # Refuse to measure the previous photograph. See fa_common.assert_fresh.
    F.assert_fresh(
        os.path.join(HERE, "metrics_frontal.json"),
        os.path.join(HERE, "frontal.png"),
        stage=os.path.basename(__file__),
    )
    pts = load_pts()
    canon = load_canonical()
    with open(os.path.join(HERE, "metrics_frontal.json")) as f:
        mf = json.load(f)
    lower_pct = [t for t in mf["thirds"]["items"] if t["zone"].startswith("Lower")][0][
        "pct"
    ]

    # Head yaw is measured once upstream and carried in metrics_frontal.json.
    # Symmetry is meaningless without it — see pose_condition().
    sym = symmetry(pts, yaw_deg=mf.get("pose", {}).get("yaw_deg"))
    dim = dimorphism(pts, lower_pct)
    avg = averageness(pts, canon)

    # THE OVERLAY IS INK ON A PHOTOGRAPH, SO IT USES THE PHOTOGRAPH'S LANDMARKS.
    # In the set path `pts` is a pose-free render of a shape fitted across several
    # photographs -- nobody was ever photographed from that angle, so drawing its
    # axis and pair markers over `frontal.png` would land them beside the features
    # instead of on them: a picture that quietly disagrees with its own caption.
    # The published figures still come from the fit; only the ink moves, and the
    # drawn axis is recomputed from the photograph so the picture is internally
    # consistent rather than half of each. Identical arrays in the two-photo path,
    # where this is a no-op.
    photo_pts = F.load_landmark_csv("photo", HERE)
    if np.allclose(photo_pts, pts):
        draw_symmetry(pts, sym)
    else:
        draw_symmetry(photo_pts, symmetry(photo_pts, yaw_deg=None))
    draw_dimorphism(dim)

    out = {
        "symmetry": {**{k: v for k, v in sym.items() if not k.startswith("_")},
                     **_ranked_asymmetry()},
        "sexual_dimorphism": dim,
        "averageness": {k: v for k, v in avg.items() if not k.startswith("_")},
    }
    with open(os.path.join(HERE, "advanced_metrics.json"), "w") as f:
        json.dump(out, f, indent=2)

    print("=== SYMMETRY ===")
    print(
        f"  roll {sym['roll_deg']:+.1f} deg | yaw-balance {sym['yaw_balance_pct']:.0f}% | eye-width ratio {sym['eye_width_ratio']}"
    )
    print(
        f"  mean asymmetry {sym['mean_asym_pct_ipd']:.1f}% of IPD (median {sym['median_asym_pct_ipd']:.1f}%, max {sym['max_asym_pct_ipd']:.1f}%)"
    )
    pc = sym["pose_condition"]
    print(f"  pose condition: {pc['status'].upper()} — {pc['note']}")
    print(f"  by region: {sym['by_region']}")
    print("=== SEXUAL DIMORPHISM ===  tally:", dim["lean_tally"])
    print("  (a count of features, not a score — the features are not summable)")
    for r in dim["metrics"]:
        print(f"  {r['metric']:<42} {r['value']:.3f}  -> {r['lean']}")
    print("=== AVERAGENESS ===")
    print(
        f"  Procrustes disparity {avg['procrustes_disparity']} (lower = closer to template avg)"
    )
    print(f"  most-deviated regions: {list(avg['by_region'].items())[:3]}")
    print("wrote symmetry_map.png, dimorphism_chart.png, advanced_metrics.json")


if __name__ == "__main__":
    main()
