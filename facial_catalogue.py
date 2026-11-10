#!/usr/bin/env python3
"""Public implementation; private development notes omitted."""

from __future__ import annotations

import numpy as np

import fa_common as F

# ----------------------------------------------------------------------------
# Landmark groups, each corroborated by ordering + magnitude on 825 reference rows (762 distinct faces).
# ----------------------------------------------------------------------------
FOREHEAD_L, FOREHEAD_R = 54, 284  # frontotemporale-ish; 85.4% of bizygomatic
CHIN_L, CHIN_R = 32, 262  # chin protuberance; 33.1% of bizygomatic
L_BROW = [70, 63, 105, 66, 107, 55, 65, 52, 46]
R_BROW = [300, 293, 334, 296, 336, 285, 295, 282, 276]

# ⛔ DROPPED BY THE CORROBORATION CHECK — kept so nobody re-adds them blind.
DROPPED = {
    "nasal_tip_width": (
        "mesh points 45/275 read 27% of alar base; the published tip:alar ratio "
        "is ~65% (22-26mm tip vs 34-40mm alar). These are not the tip-defining "
        "points and no other pair in the 468 mesh clearly is."
    ),
    "nasal_bridge_width": (
        "points 196/419 read 26% of alar base against a published ~38%. Too "
        "narrow to be the dorsum."
    ),
    "nasal_root_width": (
        "points 189/413 read 62% of alar against a published ~48%. Closer than "
        "the other two but still uncorroborated; held out rather than shipped."
    ),
}


def _eye_centres(p):
    return p[F.LEFT_EYE_RING].mean(axis=0), p[F.RIGHT_EYE_RING].mean(axis=0)


def base(p):
    """Every length the catalogue ratios are built from, in pixels."""
    d = {}
    ecL, ecR = _eye_centres(p)
    d["ipd"] = float(np.hypot(*(ecL - ecR)))
    d["bizygomatic"] = F.hdist(p, F.ZYGION_L, F.ZYGION_R)
    d["forehead_w"] = F.hdist(p, FOREHEAD_L, FOREHEAD_R)
    d["jaw_w"] = F.hdist(p, F.JAW_L, F.JAW_R)
    d["chin_w"] = F.hdist(p, CHIN_L, CHIN_R)
    d["intercanthal"] = F.hdist(p, F.L_CANTHUS_MED, F.R_CANTHUS_MED)
    d["biocular"] = F.hdist(p, F.L_CANTHUS_LAT, F.R_CANTHUS_LAT)
    d["eye_w"] = (
        F.hdist(p, F.L_CANTHUS_LAT, F.L_CANTHUS_MED)
        + F.hdist(p, F.R_CANTHUS_MED, F.R_CANTHUS_LAT)
    ) / 2
    d["eye_h"] = (
        abs(p[F.L_EYE_TOP][1] - p[F.L_EYE_BOT][1])
        + abs(p[F.R_EYE_TOP][1] - p[F.R_EYE_BOT][1])
    ) / 2
    d["alar"] = F.hdist(p, F.ALA_L, F.ALA_R)
    d["mouth_w"] = F.hdist(p, F.MOUTH_L, F.MOUTH_R)
    d["lip_h"] = F.vdist(p, F.LABRALE_SUP, F.LABRALE_INF)
    browc = (p[F.BROW_L][1] + p[F.BROW_R][1]) / 2
    d["face_h"] = abs(p[F.TRICHION][1] - p[F.MENTON][1])
    d["upper_third"] = abs(browc - p[F.TRICHION][1])
    d["middle_third"] = abs(p[F.SUBNASALE][1] - browc)
    d["lower_third"] = abs(p[F.MENTON][1] - p[F.SUBNASALE][1])
    d["fwhr_h"] = abs(p[F.LABRALE_SUP][1] - browc)  # classical fWHR denominator
    d["philtrum"] = abs(p[F.SUBNASALE][1] - p[F.LABRALE_SUP][1])
    peakL = min(p[i][1] for i in L_BROW)
    peakR = min(p[i][1] for i in R_BROW)
    d["brow_peak_from_pupil"] = ((ecL[1] - peakL) + (ecR[1] - peakR)) / 2
    # canthal tilt: lateral canthus relative to medial, per side, degrees
    tL = np.degrees(
        np.arctan2(
            p[F.L_CANTHUS_MED][1] - p[F.L_CANTHUS_LAT][1],
            abs(p[F.L_CANTHUS_LAT][0] - p[F.L_CANTHUS_MED][0]),
        )
    )
    tR = np.degrees(
        np.arctan2(
            p[F.R_CANTHUS_MED][1] - p[F.R_CANTHUS_LAT][1],
            abs(p[F.R_CANTHUS_LAT][0] - p[F.R_CANTHUS_MED][0]),
        )
    )
    d["canthal_tilt_deg"] = (tL + tR) / 2
    return d


# (id, printed label, expression, unit, published anchor, claim kind, source,
#  definition_checked)
#
# `unit` "ratio" | "deg" | "mm". An mm row is converted with this pipeline's own
# scale at report time and is only printed when that scale exists.
CATALOGUE = [
    # ---- overall proportion -------------------------------------------------
    (
        "facial_index",
        "Facial index (height / width)",
        lambda d: d["face_h"] / d["bizygomatic"],
        "ratio",
        None,
        None,
        None,
        True,
    ),
    (
        "fwhr",
        "Facial width-to-height ratio (fWHR)",
        lambda d: d["bizygomatic"] / d["fwhr_h"],
        "ratio",
        (1.7, 2.1),
        "measurement",
        "Qoves catalogue: men 1.7-2.1, women 1.7-2.0. ⚠️ Their own page states there "
        "is NO significant sex difference once landmark placement is standardised; "
        "measured here on 825 faces at AUC 0.531 (p=0.32), which agrees.",
        True,
    ),
    (
        "upper_third",
        "Upper third (share of face height)",
        lambda d: d["upper_third"]
        / (d["upper_third"] + d["middle_third"] + d["lower_third"]),
        "ratio",
        (0.30, 0.367),
        "preference",
        "The equal-thirds canon (33.3%). 🔴 DISPUTED FOR THIS INSTRUMENT: measured "
        "against 825 reference rows (762 distinct faces) this band contains 0.0% of them under our "
        "landmarks. Our TRICHION is the apex of the 468-point mesh — roughly "
        "mid-forehead — not the anthropometric hairline the canon assumes, so the "
        "upper third reads systematically short for EVERY face. The canon is not "
        "wrong; it is measuring a taller segment than this instrument can see. "
        "⚠️ It is also bounded by a hairline rather than a skeletal landmark, so "
        "it moves with hairstyle and recession.",
        False,
    ),
    (
        "middle_third",
        "Middle third (share of face height)",
        lambda d: d["middle_third"]
        / (d["upper_third"] + d["middle_third"] + d["lower_third"]),
        "ratio",
        (0.30, 0.367),
        "preference",
        "The equal-thirds canon (33.3%). Measured to contain 74.1% of 825 reference rows (762 distinct faces) "
        "faces under our landmarks — the one third that transfers cleanly.",
        True,
    ),
    (
        "lower_third",
        "Lower third (share of face height)",
        lambda d: d["lower_third"]
        / (d["upper_third"] + d["middle_third"] + d["lower_third"]),
        "ratio",
        (0.30, 0.367),
        "preference",
        "The equal-thirds canon (33.3%). 🔴 Contains only 7.0% of 825 reference rows (762 distinct faces) "
        "under our landmarks: the lower third absorbs the height the short upper "
        "third loses, so the two errors are one error.",
        False,
    ),
    (
        "middle_lower_ratio",
        "Middle : lower third",
        lambda d: d["middle_third"] / d["lower_third"],
        "ratio",
        (0.95, 1.05),
        "measurement",
        "Qoves: Caucasian men 1.00-1.02, women ~1.00; African American 0.93-0.97; "
        "East Asian 0.97-1.02; Arabian 0.88-0.93. Population means, "
        "ancestry-specific. 🔴 Contains 15.5% of 825 reference rows (762 distinct faces) here — it "
        "inherits the same trichion mismatch as the thirds it is built from.",
        False,
    ),
    # ---- widths -------------------------------------------------------------
    (
        "forehead_ratio",
        "Forehead width / face width",
        lambda d: d["forehead_w"] / d["bizygomatic"],
        "ratio",
        None,
        None,
        None,
        True,
    ),
    (
        "jaw_ratio",
        "Jaw width / face width (bigonial : bizygomatic)",
        lambda d: d["jaw_w"] / d["bizygomatic"],
        "ratio",
        (0.72, 0.83),
        "measurement",
        "Derived from Qoves' absolute norms: jaw 100-115mm (men) against facial "
        "width ~139mm.",
        True,
    ),
    (
        "chin_jaw_ratio",
        "Chin width / jaw width",
        lambda d: d["chin_w"] / d["jaw_w"],
        "ratio",
        None,
        None,
        None,
        True,
    ),
    # ---- eyes ---------------------------------------------------------------
    (
        "eye_spacing",
        "Eye spacing (intercanthal / biocular)",
        lambda d: d["intercanthal"] / d["biocular"],
        "ratio",
        (0.30, 0.38),
        "measurement",
        "Qoves: <0.30 close-set, 0.30-0.38 balanced, >0.38 wide-set. Same "
        "normalisation as ours (inner-corner gap over full outer span).",
        True,
    ),
    (
        "eye_almondness",
        "Eye almondness (fissure height / width)",
        lambda d: d["eye_h"] / d["eye_w"],
        "ratio",
        (0.28, 0.35),
        "measurement",
        "Qoves: Caucasian men 0.28-0.33, Caucasian women 0.30-0.35, East Asian "
        "women 0.33-0.38.",
        True,
    ),
    (
        "canthal_tilt",
        "Canthal tilt",
        lambda d: d["canthal_tilt_deg"],
        "deg",
        (0.0, 8.0),
        "measurement",
        "Qoves: men 0 to +4 deg, women +2 to +8 deg; typical 0-10.",
        True,
    ),
    (
        "brow_position",
        "Brow height (pupil to brow peak / face height)",
        lambda d: d["brow_peak_from_pupil"] / d["face_h"],
        "ratio",
        None,
        None,
        "⚠️ Qoves publish this in mm (men 18-22, women 22-27) measured pupil to brow "
        "peak. Tested here for a sex signal on 825 faces and it did NOT clear this "
        "project's pre-registered bar (held-out AUC 0.595 vs a 0.60 bar), so it is "
        "reported as a position among real faces and carries no sex reading.",
        True,
    ),
    # ---- nose and mouth -----------------------------------------------------
    (
        "alar_ratio",
        "Nose width / face width",
        lambda d: d["alar"] / d["bizygomatic"],
        "ratio",
        (0.24, 0.29),
        "measurement",
        "Derived from Qoves' absolute norms: alar 34-40mm (men), 31-37mm (women), "
        "against facial width ~139mm/131mm.",
        True,
    ),
    (
        "naso_oral",
        "Nose width / mouth width",
        lambda d: d["alar"] / d["mouth_w"],
        "ratio",
        None,
        None,
        "The classic lower-face balance. ⭐ Measured here to be the single most "
        "sex-discriminating nose ratio available to this instrument (held-out AUC "
        "0.635), where nose-over-face-width points the WRONG way (0.445).",
        True,
    ),
    (
        "mouth_ratio",
        "Mouth width / face width",
        lambda d: d["mouth_w"] / d["bizygomatic"],
        "ratio",
        (0.36, 0.40),
        "measurement",
        "Derived from Qoves: mouth 50-55mm (men), 45-50mm (women) against facial "
        "width ~139mm/131mm.",
        True,
    ),
    (
        "lip_lower_face",
        "Lip height / lower-face height",
        lambda d: d["lip_h"] / d["lower_third"],
        "ratio",
        None,
        None,
        "⭐ Measured here as the best-separating lip ratio available (held-out AUC "
        "0.638), against 0.546 for the more commonly published lip-height / "
        "mouth-width.",
        True,
    ),
    (
        "philtrum_ratio",
        "Philtrum / lower-face height",
        lambda d: d["philtrum"] / d["lower_third"],
        "ratio",
        None,
        None,
        None,
        True,
    ),
    # ---- absolute, only with a scale ----------------------------------------
    (
        "ipd_mm",
        "Interpupillary distance",
        lambda d: d["ipd"],
        "mm",
        (58.0, 68.0),
        "measurement",
        "Qoves: men 60-68mm, women 58-65mm. ⭐ This pipeline's own IPD lands inside "
        "that range, which is part of why its mm scale is trusted at all.",
        True,
    ),
    (
        "bizygomatic_mm",
        "Face width (bizygomatic)",
        lambda d: d["bizygomatic"],
        "mm",
        (131.0, 139.0),
        "measurement",
        "Qoves: ~139mm men, ~131mm women (Caucasian averages).",
        True,
    ),
    (
        "alar_mm",
        "Nose width (alar base)",
        lambda d: d["alar"],
        "mm",
        (31.0, 40.0),
        "measurement",
        "Qoves: men 34-40mm, women 31-37mm. ⚠️ Their page also quotes '20-21mm at "
        "alar base' for Caucasian females, which cannot be the same measurement; "
        "the 31-40 range is the one consistent with their other figures.",
        True,
    ),
    (
        "mouth_mm",
        "Mouth width",
        lambda d: d["mouth_w"],
        "mm",
        (45.0, 55.0),
        "measurement",
        "Qoves: men 50-55mm, women 45-50mm.",
        True,
    ),
    (
        "intercanthal_mm",
        "Intercanthal width",
        lambda d: d["intercanthal"],
        "mm",
        (30.0, 35.0),
        "measurement",
        "Qoves: ~30-35mm, men slightly wider.",
        True,
    ),
    (
        "philtrum_mm",
        "Philtrum length (Sn to Ls)",
        lambda d: d["philtrum"],
        "mm",
        (20.0, 24.0),
        "measurement",
        "Qoves/Naini/Farkas: men 22 +- 2mm, women 20 +- 2mm. 🔴 DISPUTED FOR THIS "
        "INSTRUMENT: our reading of the same named landmark pair sits ~5mm below "
        "this, while our IPD and bizygomatic both land inside their published "
        "ranges — so the disagreement is definitional, not a scale error, and this "
        "anchor must not be used to grade the subject until it is resolved.",
        False,
    ),
]

CATALOGUE_BY_ID = {c[0]: c for c in CATALOGUE}


def measure_all(p, px_per_mm=None):
    """Every catalogue value for one face. mm rows are None without a scale."""
    d = base(p)
    out = {}
    for cid, label, fn, unit, anchor, kind, src, checked in CATALOGUE:
        try:
            v = float(fn(d))
        except Exception:
            v = None
        if v is not None and unit == "mm":
            v = None if not px_per_mm else v / px_per_mm
        out[cid] = v
    return out
