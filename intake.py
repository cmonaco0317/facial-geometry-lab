#!/usr/bin/env python3
"""P1 — photo-SET intake and qualification.

WHAT CHANGED, AND WHY THIS IS NOT capture_check.py
--------------------------------------------------
`capture_check.py` qualifies ONE photo against the two-ideal-photos model: is
this a good frontal, is this a good profile. The 2026-08-02 re-scope killed that
model. What matters now is whether a SET of casual photos can support a fit, and
that is a property of the set, not of any photo in it.

The evidence is blunt. On synthetic ground truth where the true face is known,
four photos that are individually flawless but all from the SAME angle recover
shape to 26.43% error, while three ordinary spread ones recover to 0.02%
(`selfcheck_multiview.py`). Every photo passes a per-photo check; the set is
worthless. A per-photo checker cannot see this, because the defect does not
exist in any single photo.

So intake reports a per-photo verdict AND a set verdict, and the set verdict is
the one that decides.

NOTHING IS REJECTED OUTRIGHT
----------------------------
A weak photo is down-weighted, not discarded, because the fit consumes weighted
observations. Hard rejection at upload is the most likely place to lose a user,
and a mediocre photo from a NEW angle is often worth more than a perfect
duplicate of one already in the set.

THRESHOLD PROVENANCE — EVERY NUMBER DECLARES WHERE IT CAME FROM
---------------------------------------------------------------
  MEASURED-SIM   derived from intake_calibration.py's synthetic sweep, where the
                 true shape is known. Authoritative for GEOMETRY: how much
                 angular spread and how many views the estimate needs.
  MEASURED-REAL  derived from real-photo measurement on the 2026-08-01 burst.
                 Authoritative for the failure modes the simulation does not
                 model -- detection death near 90deg, IPD collapse, compound pose.
  AUTHOR-CHOSEN  a judgement call. Focus, exposure and expression remain
                 author-chosen, inherited from capture_check.py, and are STILL
                 not calibrated against known-bad captures.

The two measured sources DISAGREE about extreme angles and that disagreement is
real, not noise: the simulation says a wider baseline improves conditioning
(true, it is basic multi-view geometry) and the real photos say extreme frames
arrive degraded in ways geometry does not capture (also true). Each governs the
threshold it can actually see. Neither is allowed to override the other.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

import reference_values as _rv

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import capture_check as CC  # noqa: E402

CALIBRATION = os.path.join(HERE, "intake_calibration.json")

SIM = "MEASURED-SIM"
REAL = "MEASURED-REAL"
AUTHOR = "AUTHOR-CHOSEN"

# --- set-level thresholds --------------------------------------------------
# MEASURED-REAL. Beyond this the simulation's "wider is better" stops holding on
# actual photographs: detection failed outright at ~90deg (IMG_1891), IPD
# collapsed 337 -> 83px so absolute detector error dominates, and real extreme
# frames were compound-degraded. Frames past this still COUNT toward the set but
# earn no spread credit.
MAX_USEFUL_ABS_YAW = 45.0

# MEASURED-REAL. capture_check's PROFILE_YAW_BAND already records detection as
# known-good at 67deg and failing near 90; past this a frame is near-useless.
DEAD_ABS_YAW = 75.0

# MEASURED-SIM defaults, overridden by intake_calibration.json when present.
#
# ⭐ RAISED 26 -> 52 ON 2026-08-19, and the source class changed with it. The 26
# came from intake_calibration.py's synthetic sweep, which is authoritative for
# SHAPE RECOVERY -- how much angle the fit needs to recover geometry when the true
# shape is known. It is NOT authoritative for the DERIVED readings the report
# actually prints, and this module's own docstring already says MEASURED-REAL wins
# for failure modes the simulation does not model.
#
# MEASURED-REAL (capture_gate.min_effective_spread_deg_real): one sitting of one
# face, subsetted to a spread ladder, each rung run end to end so only the angular
# coverage changes. Worst deviation of any harmony axis from the 84.8-degree
# reference:
#     28.8 deg -> 2.48      36.8 deg -> 3.55      52.4 deg -> 0.55      64.9 deg -> 0.47
# The axis that collapses is LEFT-RIGHT BALANCE: 4.87 at 36.8 degrees against 1.32
# at 84.8. That is unsolved head turn leaking into the symmetry reading.
#
# Private development note omitted.
# photos aren't good enough then it makes sense to refuse it, but if the photos are
# good enough then it HAS to be in the report." A set at 38 degrees was passing this
# gate, reading its symmetry axis nearly 4x wrong, and that instability was then
# pooled into the movement margins -- which WITHHELD VERDICTS FROM THE IDEAL
# TWO-PHOTO PAIR. Refusing the thin set is what lets the good ones report.
#
# 52 is the lowest rung TESTED GOOD. 37 is catastrophic and the 37-52 gap is
# untested, so a floor between them would be author-chosen dressed as measured.
# In practice it asks for about 26 degrees of turn each way rather than 13.
DEFAULT_MIN_SPREAD = 52.0
DEFAULT_MIN_VIEWS = 4

# AUTHOR-CHOSEN, inherited from capture_check.py and still uncalibrated.
MIN_IPD_PX = 120.0

# --- FACE AT REST -----------------------------------------------------------
# ⛔ REFUSAL, NOT A DOWN-WEIGHT, and the only place in this file that refuses a
# photograph outright. A small face or an awkward angle costs precision; an
# EXPRESSION changes the thing being measured. A smile moves the mouth, cheeks
# and eye corners at once, so a set with one smiling frame does not fit a
# noisier version of that face -- it fits a different face, and reports the
# average with a straight one.
#
# ⭐ WHICH BLENDSHAPES, AND WHY ONLY THESE THREE. Measured 2026-08-25 over 61
# real frames from 11 capture sets and TWO subjects, against every cutoff in
# `capture_check.EXPRESSION`:
#
#   mouthSmileL/R (cut 0.20)  max observed 0.040   safe at every yaw, both faces
#   jawOpen       (cut 0.15)  max observed 0.068   safe at every yaw, both faces
#   ---------------------------------------------------------------------------
# Private development note omitted.
#                                AT REST, on four separate near-frontal frames.
#                                The cutoff was calibrated on 11 frames of one
#                                person whose relaxed range is 0.15-0.33. It is
#                                a fact about that person's eye aperture, not
#                                about posing, and gating on it would refuse a
#                                stranger's perfectly good photographs.
#   eyeBlinkL/R   (cut 0.40)  ⛔ POSE-CONTAMINATED: <=0.15 near-frontal, up to
#                                0.70 turned. The far eye occludes and the model
#                                reports it part-closed.
#   browDownL/R   (cut 0.25)  ⛔ POSE-CONTAMINATED: <=0.10 near-frontal, up to
#                                0.53 turned.
#   mouthPucker               ⛔ SUBJECT-DEPENDENT AND ALREADY HANDLED. It breaks
# Private development note omitted.
#                                0.769) while the stranger's max is 0.109.
#                                Refusing on it would reject most real sets. It
#                                is gated where it actually does damage --
#                                `averageness_band` withholds the typicality
#                                figure -- which is the honest split: the
#                                photograph is fine for shape, not for that one
#                                reading.
#
# The three that ship have 3-5x of headroom between the worst relaxed frame and
# their cutoff, so a breach means a real expression rather than a different face.
AT_REST_KEYS = ("mouthSmileLeft", "mouthSmileRight", "jawOpen")


def load_calibration() -> dict:
    """Read the derived thresholds, falling back to the recorded defaults."""
    if not os.path.exists(CALIBRATION):
        return {
            "min_spread_deg": DEFAULT_MIN_SPREAD,
            "min_views": DEFAULT_MIN_VIEWS,
            "_source": "defaults — intake_calibration.json not present",
            "by_spread_band": {},
        }
    with open(CALIBRATION, encoding="utf-8") as fh:
        payload = json.load(fh)
    d = payload.get("derived", {})
    spread = d.get("min_spread_deg")
    if spread is None or (isinstance(spread, float) and np.isnan(spread)):
        spread = DEFAULT_MIN_SPREAD
    # smallest view count whose pass rate clears half
    min_views = DEFAULT_MIN_VIEWS
    for k, v in sorted((int(k), v) for k, v in d.get("by_n_views", {}).items()):
        if v.get("pass_rate", 0) >= 0.4:
            min_views = k
            break
    # ⭐ TWO CONSTRAINTS, AND BOTH MUST HOLD. `spread` above is the SIM-derived
    # floor for SHAPE RECOVERY, regenerated into intake_calibration.json by
    # intake_calibration.py. The registry carries a second, stricter floor
    # measured on REAL photographs for the DERIVED READINGS the report prints --
    # a set can be wide enough to recover the geometry and still too narrow for
    # left-right balance to mean anything (4.87 at 36.8 deg against 1.32 at 84.8).
    # Taking the max keeps the generated file authoritative for its own question
    # while never admitting a set the real measurement says is unusable, and it
    # survives regeneration of that file rather than being overwritten by it.
    try:
        real_floor = float(
            _rv.require("capture_gate.min_effective_spread_deg_real").value
        )
        binding = REAL if real_floor > float(spread) else SIM
        spread = max(float(spread), real_floor)
    except Exception:  # noqa: BLE001 - registry absent: fall back to the sim floor
        binding = SIM
    return {
        "min_spread_deg": float(spread),
        "min_spread_source": binding,
        "min_views": int(min_views),
        "_source": payload.get("_command", "intake_calibration.json"),
        "by_spread_band": d.get("by_spread_band", {}),
        "target_shape_error_pct": d.get("target_shape_error_pct"),
    }


# ---------------------------------------------------------------------------
# Per photo
# ---------------------------------------------------------------------------
def assess_photo(path: str) -> dict:
    """Measure one photo. Returns a record with a WEIGHT, not a pass/fail."""
    name = os.path.basename(path)
    rec: dict = {"file": name, "path": path, "weight": 0.0, "notes": []}
    try:
        img = CC.read_image(path)
    except Exception as exc:  # noqa: BLE001
        rec["status"] = "unreadable"
        rec["notes"].append(f"could not open ({type(exc).__name__})")
        return rec
    if img is None:
        rec["status"] = "unreadable"
        rec["notes"].append("could not decode")
        return rec

    try:
        det = CC.detect(path)
    except Exception as exc:  # noqa: BLE001
        rec["status"] = "no-face"
        rec["notes"].append(f"detection failed ({type(exc).__name__})")
        return rec
    if det is None or not det.get("found"):
        rec["status"] = "no-face"
        rec["notes"].append("no face found — too turned, too small, or occluded")
        return rec

    pts = det["pts"]
    # capture_check.detect returns pose as (pitch, yaw, roll) from the
    # FaceLandmarker transformation matrix, or None if it was unavailable.
    pitch, yaw, roll = det.get("pose") or (None, None, None)
    ipd = float(np.linalg.norm(pts[468] - pts[473])) if len(pts) > 473 else float("nan")

    rec.update(
        yaw=None if yaw is None else round(float(yaw), 1),
        pitch=None if pitch is None else round(float(pitch), 1),
        roll=None if roll is None else round(float(roll), 1),
        ipd_px=round(ipd, 1),
    )

    # ⛔ THE ONE OUTRIGHT REFUSAL. Everything below this is a weight; this is a
    # rejection, because an expression changes the face rather than blurring it.
    blend = det.get("blend") or {}
    awake = [
        (k, float(blend[k]), cut)
        for k, cut, _why in CC.EXPRESSION
        if k in AT_REST_KEYS and k in blend and float(blend[k]) > cut
    ]
    if awake:
        worst = max(awake, key=lambda a: a[1] - a[2])
        why = next(w for k, _c, w in CC.EXPRESSION if k == worst[0])
        rec["status"] = "not-at-rest"
        rec["weight"] = 0.0
        rec["at_rest"] = False
        rec["notes"].append(
            f"face is not at rest — {why} ({worst[0]} {worst[1]:.2f} over {worst[2]:.2f}). "
            "Re-take it with a completely relaxed face: lips together, jaw loose, no smile."
        )
        return rec
    rec["at_rest"] = True

    weight = 1.0
    if not np.isnan(ipd) and ipd < MIN_IPD_PX:
        weight *= max(0.15, ipd / MIN_IPD_PX)
        rec["notes"].append(f"face is small in frame (IPD {ipd:.0f}px) — move closer")
    if yaw is not None and abs(yaw) > DEAD_ABS_YAW:
        weight *= 0.1
        rec["notes"].append(
            f"turned {abs(yaw):.0f}° — past where the detector is reliable"
        )
    elif yaw is not None and abs(yaw) > MAX_USEFUL_ABS_YAW:
        weight *= 0.5
        rec["notes"].append(f"turned {abs(yaw):.0f}° — usable but degraded")

    rec["weight"] = round(float(np.clip(weight, 0.0, 1.0)), 3)
    rec["status"] = "good" if weight >= 0.75 else ("weak" if weight > 0.2 else "poor")
    if not rec["notes"]:
        rec["notes"].append("fine")
    return rec


# ---------------------------------------------------------------------------
# Set level — the verdict that decides
# ---------------------------------------------------------------------------
def angular_coverage(yaws: list[float], ipds: list[float] | None = None) -> dict:
    """Spread that actually counts, ignoring credit from unusable frames.

    ⭐ FACE SIZE NOW DISQUALIFIES A FRAME, added 2026-08-19. MIN_IPD_PX has existed
    since this module was written and was never a floor -- it only scaled a WEIGHT.
    So a set in which every photograph was 54-96 px between the pupils, each one
    flagged "face is small in frame — move closer", still reported "9 with a usable
    face" and passed as READY TO MEASURE.

    That is not the soft-gate philosophy in this module's docstring working as
    intended. That philosophy is about ANGLE -- "a mediocre photo from a NEW angle
    is often worth more than a perfect duplicate" -- and it does not extend to a
    face with no detail in it. Measured: at 96 px IPD the Horizontal balance axis
    changes SIDE against the same face at 383 px; at 191 px it does not.

    ⛔ THIS ONLY EVER REFUSES MORE. MIN_IPD_PX keeps the value it has always
    declared (120, AUTHOR-CHOSEN and uncalibrated) rather than a new one invented
    here -- the change is that the number is now enforced instead of decorative.
    The measurement above brackets it rather than setting it: 96 flips, 191 does
    not, and 120 sits between.
    """
    if ipds is None:
        ipds = [float("nan")] * len(yaws)
    usable = [
        y
        for y, d in zip(yaws, ipds)
        if abs(y) <= MAX_USEFUL_ABS_YAW and not (not np.isnan(d) and d < MIN_IPD_PX)
    ]
    if not usable:
        return {
            "effective_spread_deg": 0.0,
            "n_usable": 0,
            "largest_gap_deg": 0.0,
            "yaws": [],
        }
    srt = sorted(usable)
    gaps = [b - a for a, b in zip(srt, srt[1:])] or [0.0]
    return {
        "effective_spread_deg": round(srt[-1] - srt[0], 1),
        "n_usable": len(usable),
        "largest_gap_deg": round(max(gaps), 1),
        "yaws": [round(y, 1) for y in srt],
    }


def missing_advice(cov: dict, photos: list[dict], cal: dict) -> list[str]:
    """Concrete, actionable instructions. Never 'add more photos'."""
    out: list[str] = []
    yaws = cov["yaws"]
    need_spread = cal["min_spread_deg"]

    # ⭐ LEAD WITH THE ACTUAL CAUSE. When frames were disqualified for FACE SIZE,
    # telling someone to add more angles sends them to re-shoot the wrong thing --
    # the photographs they have may already cover every angle needed and simply be
    # taken from too far away. Found 2026-08-19 on a set where all nine frames were
    # 54-96 px between the pupils: the advice opened with "add 4 more usable
    # photo(s) from ANGLES YOU DO NOT ALREADY HAVE" and mentioned the distance
    # fourth.
    # ⭐ AND LEAD WITH REST BEFORE SIZE, for the same reason again. A refused
    # expression removes the frame from the count entirely, so the set fails with
    # "add more usable photo(s)" -- which sends someone off to shoot MORE photos
    # of a face that is doing the wrong thing, and they will keep smiling in them.
    not_at_rest = [p for p in photos if p.get("status") == "not-at-rest"]
    if not_at_rest:
        out.append(
            f"RE-TAKE {len(not_at_rest)} photo(s) WITH THE FACE AT REST — "
            + ", ".join(sorted(p["file"] for p in not_at_rest))
            + ". An expression is not noise that averaging removes: a smile moves "
            "the mouth, cheeks and eye corners at once, so the fit would blend a "
            "different face into the answer. Lips together but not pressed, jaw "
            "loose, no smile, eyes open on the lens."
        )

    too_small = [
        p for p in photos
        if p.get("ipd_px") is not None
        and not np.isnan(p["ipd_px"])
        and p["ipd_px"] < MIN_IPD_PX
    ]
    if too_small and cov["n_usable"] < cal["min_views"]:
        out.append(
            f"MOVE CLOSER — {len(too_small)} of your photo(s) have the face too small "
            f"to measure ({min(p['ipd_px'] for p in too_small):.0f}-"
            f"{max(p['ipd_px'] for p in too_small):.0f} px between the pupils, need "
            f"{MIN_IPD_PX:.0f}+). Head and shoulders should fill the frame at about "
            f"an arm's length. The ANGLES you already have may be fine; it is the "
            f"distance that disqualified them."
        )
    elif cov["n_usable"] < cal["min_views"]:
        out.append(
            f"Add {cal['min_views'] - cov['n_usable']} more usable photo(s) — "
            f"but from ANGLES YOU DO NOT ALREADY HAVE (see below); duplicates of an "
            f"angle you have add almost nothing."
        )
    if cov["effective_spread_deg"] < need_spread:
        # YAW SIGN CONVENTION — VERIFIED, NOT ASSUMED (2026-08-02).
        #
        # Getting this backwards tells a user to turn the wrong way, so it was
        # measured rather than remembered. For each burst frame, the distance
        # from nose tip (1) to each side of the face (234 = subject's right,
        # 454 = subject's left) was compared; the COMPRESSED side is the one
        # rotated away from the camera, i.e. the way the head is turned:
        #
        #     IMG_1892  yaw +20.4  ->  subject's LEFT compressed
        #     IMG_1885  yaw +12.9  ->  subject's LEFT
        #     IMG_1890  yaw +57.8  ->  subject's LEFT
        #     IMG_1893  yaw -17.9  ->  subject's RIGHT
        #     IMG_1888  yaw -15.9  ->  subject's RIGHT
        #     IMG_1889  yaw -62.9  ->  subject's RIGHT
        #
        # => POSITIVE yaw = turned toward the subject's OWN LEFT. 6/6 consistent.
        has_turned_right = any(y < -12 for y in yaws)  # negative yaw
        has_turned_left = any(y > 12 for y in yaws)  # positive yaw
        if not has_turned_right:
            out.append("Add a photo with your head turned to YOUR RIGHT about 20–30°.")
        if not has_turned_left:
            out.append("Add a photo with your head turned to YOUR LEFT about 20–30°.")
        if has_turned_right and has_turned_left:
            out.append(
                f"Your angles span only {cov['effective_spread_deg']:.0f}° — widen it to at "
                f"least {need_spread:.0f}° with a more turned (but not profile) shot."
            )
    elif cov["largest_gap_deg"] > 35.0:
        out.append(
            f"Your angles have a {cov['largest_gap_deg']:.0f}° hole in the middle — "
            f"one roughly front-on photo would close it."
        )

    small = [p for p in photos if "small in frame" in " ".join(p.get("notes", []))]
    if small:
        out.append(
            f"{len(small)} photo(s) have your face small in frame — closer or "
            f"higher-resolution versions of those would carry more weight."
        )
    extreme = [
        p
        for p in photos
        if p.get("yaw") is not None and abs(p["yaw"]) > MAX_USEFUL_ABS_YAW
    ]
    if extreme:
        out.append(
            f"{len(extreme)} photo(s) are turned past {MAX_USEFUL_ABS_YAW:.0f}°. Near-profile "
            f"shots look like they add the most variety but measurably do not — the far "
            f"side of the face is hidden and the detector degrades. Moderate angles are "
            f"worth more."
        )
    return out


def assess_set(paths: list[str]) -> dict:
    cal = load_calibration()
    photos = [assess_photo(p) for p in paths]
    with_face = [p for p in photos if p.get("yaw") is not None and p["weight"] > 0.2]
    cov = angular_coverage(
        [p["yaw"] for p in with_face],
        [p.get("ipd_px", float("nan")) for p in with_face],
    )

    ready = (
        cov["n_usable"] >= cal["min_views"]
        and cov["effective_spread_deg"] >= cal["min_spread_deg"]
    )
    advice = missing_advice(cov, photos, cal)
    return {
        "photos": photos,
        "coverage": cov,
        "calibration": cal,
        "ready": bool(ready),
        "advice": advice,
        "n_submitted": len(photos),
        "n_with_face": len(with_face),
    }


# ---------------------------------------------------------------------------
def cohort_section(age=None, race=None, sex=None) -> None:
    """WHO THE READER WILL BE COMPARED WITH — asked at upload, where it is fixable.

    ⭐ AGE AND RACE ARE NOT DEMOGRAPHIC DECORATION HERE, they select the actual
    bank of faces every percentile is computed in. Without them the report falls
    back to a pooled bank of 825 faces that is 56% aged 10-19 and 56% White, no
    matter who submitted the photographs -- and moving to a matched cohort shifts
    the median catalogue row by 7.5 percentile points and some rows by 15-26. A
    percentile against a population you are not in is not a wider answer; it is
    the answer to a different question.

    ⛔ AND IT IS ASKED, NEVER INFERRED. This pipeline does not estimate age or
    race from a photograph and must not start: a guessed label would silently
    pick a comparison population, which is the same defect as defaulting to one.
    Left blank, the report says it is unmatched rather than guessing.
    """
    import subject as SUBJ

    print()
    print("=" * 78)
    print("WHO YOU WILL BE COMPARED WITH")
    print("=" * 78)
    have_age = age if age is not None else SUBJ.load().get("age")
    have_race = race or SUBJ.race()
    have_sex = sex or SUBJ.cohort_sex()
    bands = getattr(SUBJ, "RACE_COHORT_BANDS", [])
    lo = bands[0][0] if bands else 20
    hi = bands[-1][1] if bands else 29

    missing = []
    if have_age is None:
        missing.append("age")
    if have_race is None:
        missing.append("race group")
    if have_sex is None:
        missing.append("sex")

    print(f"  age        : {have_age if have_age is not None else '— not given'}")
    print(f"  race group : {have_race or '— not given'}")
    print(f"  sex        : {have_sex or '— not given'}")
    print()
    if not missing:
        band = next((b for a0, a1, b in bands if a0 <= have_age <= a1), None)
        if band:
            print(f"  ✓ Your percentiles will be measured against real "
                  f"{have_race.replace('_', ' ')} {have_sex} faces aged "
                  f"{band.replace('_', '-')} — your own cohort.")
        else:
            print(f"  ⚠️ Matched reference faces exist only for ages {lo}-{hi}, and you "
                  f"are {have_age}.")
            print(f"     Your percentiles will come from a pooled bank of 825 faces that "
                  f"is 56% aged")
            print(f"     10-19 and 56% White. The report will SAY it is unmatched rather "
                  f"than pretend.")
    else:
        print(f"  ⚠️ Missing {', '.join(missing)} — so your percentiles will come from a "
              f"pooled bank")
        print( "     of 825 faces that is 56% aged 10-19 and 56% White, whoever you are.")
        print( "     Matching moves the typical measurement by about 7.5 percentile points,")
        print( "     and some by 15-26. It also decides whether a masculine/feminine read is")
        print( "     shown at all: a feature has to be proven on faces like yours first.")
        print()
        print( "     Add them and re-run:")
        print(f"       ./venv/bin/python intake.py <photos> --age 34 --race Black --save-profile")
        print(f"     Race groups: {', '.join(SUBJ.RACE_NAMES)}")
        print( "     Nothing is guessed from your photographs — left blank, the report says")
        print( "     plainly that the comparison is unmatched.")


def render(result: dict) -> None:
    cal = result["calibration"]
    print("=" * 78)
    print("PHOTO SET INTAKE")
    print("=" * 78)
    print(f"  {'file':<26}{'yaw':>8}{'pitch':>8}{'IPD':>8}{'weight':>9}  notes")
    for p in result["photos"]:
        yaw = f"{p['yaw']:.1f}" if p.get("yaw") is not None else "--"
        pit = f"{p['pitch']:.1f}" if p.get("pitch") is not None else "--"
        ipd = (
            f"{p['ipd_px']:.0f}"
            if p.get("ipd_px") == p.get("ipd_px") and "ipd_px" in p
            else "--"
        )
        print(
            f"  {p['file'][:25]:<26}{yaw:>8}{pit:>8}{ipd:>8}{p['weight']:>9.2f}  "
            f"{p['notes'][0]}"
        )

    cov = result["coverage"]
    print()
    print("=" * 78)
    print("SET COVERAGE — this is the verdict that decides")
    print("=" * 78)
    print(f"  photos submitted            : {result['n_submitted']}")
    print(
        f"  with a usable face          : {cov['n_usable']}   (need >= {cal['min_views']}, {SIM})"
    )
    print(
        f"  effective angular spread    : {cov['effective_spread_deg']:.0f}°"
        # NAME THE SOURCE THAT IS ACTUALLY BINDING. The spread floor is the max of
        # a SIM-derived shape-recovery floor and a REAL-measured derived-reading
        # floor, and printing "MEASURED-SIM" while the real one binds is the
        # provenance mislabel this project audits for everywhere else.
        f"   (need >= {cal['min_spread_deg']:.0f}°, {cal.get('min_spread_source', SIM)})"
    )
    print(f"  largest gap between angles  : {cov['largest_gap_deg']:.0f}°")
    print(f"  angles present              : {cov['yaws']}")
    print(
        f"\n  'Effective' spread ignores frames past {MAX_USEFUL_ABS_YAW:.0f}° ({REAL}): they still\n"
        f"  count as photos but earn no spread credit, because on real photographs the\n"
        f"  far side is hidden and the detector degrades."
    )

    print()
    print("=" * 78)
    print(
        "VERDICT: "
        + (
            "READY TO MEASURE"
            if result["ready"]
            else "NOT YET — here is what is missing"
        )
    )
    print("=" * 78)
    if result["advice"]:
        for a in result["advice"]:
            print(f"  • {a}")
    else:
        print("  • Nothing missing. This set can support a fit.")
    if not result["ready"]:
        print(
            "\n  Why this is not pedantry: four photos that each look perfect but share\n"
            "  one angle recover shape to 26.4% error, where three ordinary spread ones\n"
            "  reach 0.02% (selfcheck_multiview.py, synthetic ground truth). The defect\n"
            "  lives in the SET, so no per-photo check can see it."
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+", help="photo files, or a directory of them")
    ap.add_argument("--json", metavar="PATH", help="also write the full result as JSON")
    # These select the COMPARISON POPULATION every percentile is computed in.
    ap.add_argument("--age", type=int, help="your age — picks the matched reference band")
    ap.add_argument("--race", help="your race group — see the list intake prints")
    ap.add_argument("--save-profile", action="store_true",
                    help="write --age/--race into subject.json for the report")
    args = ap.parse_args()

    files: list[str] = []
    for p in args.paths:
        if os.path.isdir(p):
            files.extend(
                sorted(
                    f
                    for ext in ("png", "jpg", "jpeg", "heic", "HEIC", "JPG", "PNG")
                    for f in glob.glob(os.path.join(p, f"*.{ext}"))
                )
            )
        else:
            files.append(p)
    files = sorted(set(files))
    if not files:
        raise SystemExit("no photos found")

    result = assess_set(files)
    render(result)
    if args.save_profile and (args.age is not None or args.race):
        import subject as SUBJ

        prof = {}
        if os.path.exists(SUBJ.CONFIG):
            try:
                with open(SUBJ.CONFIG, encoding="utf-8") as fh:
                    prof = json.load(fh)
            except ValueError:
                prof = {}
        if args.age is not None:
            prof["age"] = args.age
        if args.race:
            slug = args.race.strip().lower().replace(" ", "_")
            if slug not in SUBJ.RACE_SLUGS:
                raise SystemExit(
                    f"unknown race group {args.race!r}; one of: "
                    + ", ".join(SUBJ.RACE_NAMES)
                )
            prof["race"] = SUBJ.RACE_SLUGS[slug]
        with open(SUBJ.CONFIG, "w", encoding="utf-8") as fh:
            json.dump(prof, fh, indent=2)
        print(f"\n  wrote {os.path.basename(SUBJ.CONFIG)}")
    cohort_section(args.age, args.race, None)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
        print(f"\n  wrote {args.json}")
    return 0 if result["ready"] else 2


if __name__ == "__main__":
    sys.exit(main())
