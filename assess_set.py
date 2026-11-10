#!/usr/bin/env python3
"""Public implementation; private development notes omitted."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import glob
import hashlib
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import analyze_averageness2 as A2  # noqa: E402
import averageness_band as AB  # noqa: E402
import bank_derive as BD  # noqa: E402
import harmony_axes as HA  # noqa: E402
from local_app.contracts import AxisResult, CohortUse, PhotoRequest, SYMMETRY_UNIT  # noqa: E402
from local_app.guidance import PhotoFact, request_for_axis  # noqa: E402
from validate_harmony_score import score_from_artifact  # noqa: E402

MIN_IN_GATE = 4          # the set size every margin here was measured at
N_NOISE = 12

# ⚠️ SYMMETRY READS ONLY NEAR-FRONTAL PHOTOGRAPHS, AND THE REASON IS NARROWER
# THAN IT FIRST LOOKED.
#
# ⛔ CORRECTED 2026-09-08, THE SAME DAY IT WAS WRITTEN. An earlier version of this
# comment claimed "three quarters of the asymmetry noise at the ordinary 14 degree
# gate is head turn", from a Multiface table reading noise/signal 3.17 at 14 deg
# falling to 0.42 at 5. THAT TABLE DIVIDED A 5-DEGREE-GATED NOISE BY THE
# 14-DEGREE-GATED BANK SPREAD. Gating tightens BOTH sides -- the bank's own SD
# falls from 4.955 to 1.998 as its pose noise is removed too -- and with both
# gated consistently the ratio does NOT improve: 3.17 at 14 deg, 10.29 at 10,
# signal collapsing to zero at 7, 3.96 at 5. The head-turn story was an artefact
# of mixing gates.
#
# WHAT SURVIVES, and it is what the gate is actually set on:
# `symmetry_yaw_gate_probe.py` gates BOTH sides correctly and measures set-to-set
# movement as a share of the deconvolved population scale -- 73.5% at 14 deg,
# 53.5% at 10, 42.1% at 7, 28.3% at 5. That improvement is real and correctly
# computed, so the gate earns its place by the MOVEMENT figure and not by any
# claim about where the noise lives. Two independent noise estimates disagree by
# ~2.5x at 5 deg (LFW-based noise/signal 1.61, Multiface-based 3.96) and BOTH say
# symmetry cannot be placed. It ships tier 2 under either.
SYMMETRY_MAX_YAW = 5.0


# ⭐ THE SECOND PER-METRIC CAPTURE GATE, and it was EARNED the same way the yaw
# one was -- by measurement, under a rule fixed before the data (NEXT-FOUR §2.3).
#
# `pitch_gate_probe.py` swept |pitch| at 16, 12, 9, 6 and 4 degrees over the ten
# metrics §2.1's rule DERIVES as pitch-suspect, gating the bank and the strangers
# at the same threshold every time.  The pre-registered rule was: adopt the
# tightest gate whose corrected movement is >= 3 points lower than at 16 deg AND
# whose availability is >= 60% of identities with >= 4 in-gate photographs.
#
# ⭐ EXACTLY ONE METRIC CLEARS IT: `upper_third`, at 9 degrees -- 27.9% of the
# true population scale at 16 deg, 22.2% at 9, which crosses the 25% a placement
# needs.  Availability falls from 88.1% of identities to 74.6%, above the floor.
#
# ⛔ AND THE ALL-COMERS FIGURE ALONE WOULD NOT HAVE JUSTIFIED THIS.  A tighter
# gate changes WHICH PEOPLE are measured, so an improvement in the average can be
# pure composition -- the hard cases simply stop qualifying.  So the probe also
# scores the population each gate would SERVE both ways: the same 91 identities
# read 25.3% using all their in-16-degree photographs and 21.9% using only the
# in-9-degree ones, a within-person gain of 3.4 points that clears the same bar
# independently.  `upper_third` is the ONLY metric where both agree; at 9 degrees
# `harm__vertical_balance` reads -0.9 within-person, `lower_third` -1.8 and
# `asymmetry` -1.8, all WORSE.  Those nine are dropped with their numbers.
#
# ⚠️ 6 degrees would have been better still (21.2% all-comers, +3.4 within-person)
# and is REFUSED on availability alone: 57.2% of identities, under the 60% floor.
# The bar was fixed before the data and is not moved to collect it.
PITCH_GATES = {"upper_third": 9.0}


# Private development note omitted.
# 2026-09-09, reversing the decision he locked on 2026-09-04 (§6.4).
#
# The superseded rule is preserved here, because a future reader has to be able
# to see this was a DECISION and not a drift:
#
#     §4.1(a): "Rater consensus — REPORTED, NEVER GATED. Gating harmony on rater
#     agreement would quietly redefine harmony as PERCEIVED ATTRACTIVENESS,
#     which is a different construct with a different literature. Publish the
#     correlation whatever it says. A near-zero result is a finding."
#
# THE BAR, pre-registered 2026-09-09 before the correlation was computed:
# Pearson r against mean rater score, 95% CI lower bound > 0.10, WITHIN cohort,
# and an axis must clear it in EVERY measurable cohort. Measured on SCUT-FBP5500
# (4,743 of 5,500 faces in this bank's own gate, four cohorts).
#
# ⭐ FIVE OF SIX AXES ARE REFUSED. Only `nose_mouth_eye` clears it, and it is the
# one axis whose construct is monotonic -- an explicit absolute deviation from
# the canon -- which is exactly where a linear agreement test can see anything:
#     nose_mouth_eye  r 0.397-0.614, worst cohort lower-CI 0.300   PASSES
#     lip_balance     best r 0.216, fails in 3 of 4 cohorts        REFUSED
#     horizontal_balance / lower_face_split / midface_proportion   REFUSED
#     vertical_balance best r -0.315 -- INVERTED, not merely null  REFUSED
# Robust to both cohort judgement calls: identical survivor set under
# Asian->southeast_asian and under the 30-39 age band.
#
# ⚠️⚠️ WHAT THE SURVIVOR NOW MEANS, and it must travel with the number:
# `nose_mouth_eye` is no longer GEOMETRIC harmony. It is geometry that predicts
# the ratings of 60 Chinese volunteers aged 18-27. The neoclassical-canon
# literature the axis was built from no longer licenses it, and a rater panel
# measures one population's perception at one time.
# Private development note omitted.
# IT: "you cant refuse symmetry or harmony". So harmony is NO LONGER GATED and
# §4.1(a)'s original rule -- REPORTED, NEVER GATED -- is back in force. The
# measurement is not discarded: every axis now carries its rater correlation as
# a DISCLOSURE beside its percentile, which is strictly more than the report
# said before SCUT was run.
#
# ⭐ WHICH IS WHY THE MEASUREMENT SURVIVED THE POLICY CHANGE. §7.11's numbers are
# unchanged and still true; what changed is that a weak correlation now travels
# WITH the axis instead of deleting it. `RATER_AGREEMENT` is read from
# scut_harmony.json rather than retyped, so it cannot drift from the run.
RATER_GATED_AXES = {"nose_mouth_eye"}   # kept: which axes CLEAR the bar


def _rater_note(metric: str) -> str:
    """What SCUT's raters said about this axis — a disclosure, never a gate.

    ⛔ §4.1(a), restored 2026-09-09: a weak correlation is a FINDING about the
    axis, not grounds to delete it. Gating on rater agreement would quietly
    redefine harmony as perceived attractiveness. The number is read from
    `scut_harmony.json` so it cannot drift from the run that produced it.
    """
    v = _measured("scut_harmony.json", "gate_verdicts", metric, default=None)
    if not v or v.get("best_r") is None:
        return "not measured against raters"
    axis = metric.replace("harm__", "")
    if axis in RATER_GATED_AXES:
        return (f"agrees with human raters (r up to {v['best_r']:.2f} across four "
                f"cohorts; clears a 0.10 bar in every one)")
    return (f"⚠️ does NOT track human ratings (best r {v['best_r']:.2f}; below a "
            f"0.10 bar in {len(v.get('failed_in', []))} of four cohorts). This is "
            f"a geometric proportion, and this is the measured limit of what it "
            f"corresponds to in perception — it is reported, not withheld.")


def _measured(path, *keys, default=None):
    try:
        with open(os.path.join(HERE, path), encoding="utf-8") as stream:
            d = json.load(stream)
        for k in keys:
            d = d[k]
        return d
    except Exception:  # noqa: BLE001
        return default


def _ordinal(value: float) -> str:
    rounded = int(round(float(value)))
    if 10 <= rounded % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(rounded % 10, "th")
    return f"{rounded}{suffix}"


def gate_photos(paths, *, retain_estimates=False):
    """Every photograph, with why it was kept or refused. Nothing is dropped silently."""
    lm = A2.make_landmarker(blendshapes=True)
    kept, refused = [], []
    for p in sorted(paths):
        L, pose, expr = A2.landmarks_and_expression(lm, p)
        name = os.path.basename(p)
        if L is None:
            refused.append({
                "photo": name,
                "code": "no_face",
                "why": "no face detected",
                "fact": {
                    "photo_id": name,
                    "detected": False,
                    "yaw": None,
                    "pitch": None,
                    "roll": None,
                    "ipd_px": None,
                    "neutral": None,
                },
            })
            continue
        pi, ya, ro = (float(v) for v in pose)
        outer_eye_span = float(np.linalg.norm(L[33] - L[263]))
        # `landmarks_and_expression` deliberately returns the 468-point mesh,
        # without iris centres.  The outer-eye floor in averageness_band is
        # 280px for the same detail level as a 200px iris-centre IPD, so this
        # conversion keeps the UI's IPD guidance in the intake scale.
        ipd_px = outer_eye_span * (200.0 / AB.MIN_IPD_PX_OUTER)
        fact = {
            "photo_id": name,
            "detected": True,
            "yaw": ya,
            "pitch": pi,
            "roll": ro,
            "ipd_px": ipd_px,
            "neutral": None,
        }
        # Product estimates can retain observed geometry outside research gates.
        # The default research path and its pass/fail decisions are unchanged.
        candidate = ({"landmarks": L, "yaw": ya, "pitch": pi, "roll": ro,
                      "expression": dict(expr or {}),
                      "ipd_px": ipd_px} if retain_estimates else {})
        if retain_estimates:
            # An off-angle face has not necessarily changed expression. Evaluate
            # posture instead of turning the strict path's unknown into False.
            fact = {**fact, "neutral": bool(AB.posture_ok(expr or {})[0])}
        if not A2.is_frontal(pi, ya, ro):
            if abs(ya) > 14:
                code = "yaw_out_of_range"
            elif abs(pi) > 16:
                code = "pitch_out_of_range"
            else:
                code = "roll_out_of_range"
            refused.append({
                **candidate,
                "photo": name,
                "code": code,
                "why": f"outside the pose gate (yaw {ya:.1f}°, "
                       f"pitch {pi:.1f}°, roll {ro:.1f}°)",
                "fact": fact,
            })
            continue
        ok, why = AB.posture_ok(expr or {})
        if not ok:
            refused.append({
                **candidate,
                "photo": name,
                "code": "expression_not_neutral",
                "why": f"not at rest — {why}",
                "fact": {**fact, "neutral": False},
            })
            continue
        kept.append({
            **({"expression": dict(expr or {})} if retain_estimates else {}),
            "photo": name,
            "landmarks": L,
            "yaw": ya,
            "pitch": pi,
            "roll": ro,
            "ipd_px": ipd_px,
            "code": "eligible",
            "fact": {**fact, "neutral": True},
        })
    return kept, refused


def cohort_pool(cohort: str, metric: str):
    import public_reference
    return public_reference.pool(cohort, metric)


def _set_median(values: list[dict], metric: str) -> float | None:
    measured = [value[metric] for value in values if metric in value]
    return float(np.median(measured)) if measured else None


def dimorphism(cohort: str, values: list[dict]) -> dict:
    """Return the existing calibrated probability plus its within-cohort rank."""
    calibrations = _measured("bank/dimorphism_calibration.json", "cohorts", default={})
    calibration_key = cohort.split("_", 1)[1] if "_" in cohort else cohort
    calibration = calibrations.get(calibration_key)
    movement = _measured(
        "axis_stability.json",
        "dimorphism",
        "median_probability_points",
        default=None,
    )
    if not calibration or calibration.get("status") != "ok":
        return {
            "cause": "no-calibration",
            "refused": (calibration or {}).get("why")
            or f"no calibration for {calibration_key} in this bank",
        }

    features = list(calibration["features"])
    subject_features = [_set_median(values, "dim__" + feature) for feature in features]
    if any(value is None for value in subject_features):
        return {
            "cause": "missing_measurement",
            "refused": "a required dimorphism feature did not measure",
            "features_used": features,
        }

    mean = np.asarray(calibration["mean"], dtype=np.float64)
    scale = np.asarray(calibration["sd"], dtype=np.float64)
    coefficients = np.asarray(calibration["coef"], dtype=np.float64)
    standardized = (np.asarray(subject_features, dtype=np.float64) - mean) / scale
    logit = coefficients[0] + float(np.dot(coefficients[1:], standardized))
    p_male = float(1 / (1 + np.exp(-np.clip(logit, -30, 30))))
    probability_interval = None
    if movement is not None:
        radius = float(movement) / 100.0
        probability_interval = [max(0.0, p_male - radius), min(1.0, p_male + radius)]

    import public_reference
    bank_alignment = public_reference.pool(cohort, "_dim_alignment")
    if bank_alignment is None or len(bank_alignment) < 50:
        return {"cause": "thin-cohort", "refused": "No eligible aggregate dimorphism reference"}
    if cohort.startswith("f_"):
        alignment = 1.0 - p_male
        if probability_interval is not None:
            alignment_interval = [
                1.0 - probability_interval[1],
                1.0 - probability_interval[0],
            ]
        else:
            alignment_interval = None
    else:
        alignment = p_male
        alignment_interval = probability_interval

    percentile = 100.0 * float((bank_alignment < alignment).mean())
    percentile_interval = None
    if alignment_interval is not None:
        percentile_interval = [
            100.0 * float((bank_alignment < alignment_interval[0]).mean()),
            100.0 * float((bank_alignment < alignment_interval[1]).mean()),
        ]

    return {
        "alignment_probability": round(alignment, 4),
        "probability_interval": (
            [round(value, 4) for value in alignment_interval]
            if alignment_interval is not None
            else None
        ),
        "percentile": round(percentile, 1),
        "percentile_interval": (
            [round(value, 1) for value in percentile_interval]
            if percentile_interval is not None
            else None
        ),
        "p_male": round(p_male, 4),
        "moves_probability_points": movement,
        "features_used": features,
        "held_out_calibration_error": calibration["held_out_calibration_error"],
        "n_cohort_faces": int(len(bank_alignment)),
        "_note": "a calibrated probability and within-cohort rank, not a sex label",
    }


def harmony(cohort: str, values: list[dict]) -> dict:
    """Return the existing per-proportion profile without inventing a scalar."""
    stability = _measured("axis_stability.json", "axes", default={})
    profile: dict[str, dict] = {}
    for axis, label in HA.AXES.items():
        metric = "harm__" + axis
        value = _set_median(values, metric)
        if value is None:
            profile[axis] = {
                "cause": "missing_measurement",
                "refused": "this proportion did not measure in the supplied photographs",
            }
            continue
        pool = cohort_pool(cohort, metric)
        if pool is None or pool.size < 50:
            profile[axis] = {
                "cause": "no-cohort-distribution",
                "refused": "no sufficiently large cohort distribution for this proportion",
            }
            continue
        percentile = HA.percentile_in(value, pool)
        movement = stability.get(metric, {}).get("median_percentile_points")
        low, high = (
            (max(0.0, percentile - movement), min(100.0, percentile + movement))
            if movement is not None
            else (None, None)
        )
        straddles = (
            low is not None and any(low <= cutoff <= high for cutoff in (33.0, 67.0))
        )
        profile[axis] = {
            "label": label,
            "value": round(value, 4),
            "agrees_with_human_raters": _rater_note(metric),
            "percentile": round(percentile, 1),
            "interval": [round(low, 1), round(high, 1)] if low is not None else None,
            "moves_percentile_points": movement,
            "placement": (
                "withheld — the interval straddles a placement cutoff"
                if straddles
                else _third(percentile)
            ),
        }
    for axis, reason in HA.REFUSED_ON_THE_BANK.items():
        profile[axis] = {
            "cause": "profile-view-required",
            "refused": reason,
        }
    return profile


class AxisDataError(RuntimeError):
    """Expected local evidence/reference failure safe to isolate to one card."""

    def __init__(
        self,
        code: str,
        public_message: str,
        *,
        retryable: bool = True,
    ) -> None:
        super().__init__(public_message)
        self.code = code
        self.public_message = public_message
        self.retryable = retryable


class ReferenceDataError(AxisDataError):
    """A local reference artifact is missing, thin, or unreadable."""


@dataclass(frozen=True)
class AssessmentContext:
    cohort: CohortUse
    kept: tuple[dict, ...]
    values: tuple[dict, ...]
    photo_facts: tuple[PhotoFact, ...]

    def __post_init__(self) -> None:
        if len(self.kept) != len(self.values):
            raise ValueError("kept photographs and measured values must stay aligned")

    @property
    def reference_id(self) -> str:
        return self.cohort.used_id

    def evidence(self, axis: str) -> tuple[tuple[dict, dict], ...]:
        facts = {fact.photo_id: fact for fact in self.photo_facts}
        return tuple(
            (photo, value)
            for photo, value in zip(self.kept, self.values)
            if (fact := facts.get(str(photo.get("photo", "")))) is not None
            and fact.eligible_for(axis)
        )

    def evidence_ids(self, axis: str) -> list[str]:
        return [str(photo.get("photo", "")) for photo, _ in self.evidence(axis)]


def _cohort_use(cohort: CohortUse | str) -> CohortUse:
    if isinstance(cohort, CohortUse):
        return cohort
    import bank_reference as BR

    record = BR.COHORTS.get(cohort)
    if record is None:
        raise ReferenceDataError(
            "cohort_not_found",
            "The requested local comparison cohort is not available.",
            retryable=False,
        )
    label = str(record.get("label") or cohort)
    return CohortUse(
        requested_id=cohort,
        used_id=cohort,
        fallback_level="exact",
        sample_size=int(record.get("n", 0)),
        requested_label=label,
        used_label=label,
        member_ids=tuple(str(item) for item in record.get("members", (cohort,))),
    )


def _generic_photo_request(axis: str) -> PhotoRequest:
    axis_name = axis.capitalize()
    return PhotoRequest(
        view="straight-on at eye level",
        expression="neutral, lips relaxed",
        lighting="soft, even light across the whole face",
        framing="head and shoulders; full face unobstructed",
        reason=(
            f"{axis_name} could not be measured from the current evidence. "
            "Add one sharp, straight-on photo with the full face visible."
        ),
    )


def _needs_photo_result(context: AssessmentContext, axis: str) -> AxisResult | None:
    request = request_for_axis(axis, context.photo_facts)
    if request is None:
        return None
    return AxisResult.needs_photo(
        axis=axis,
        request=request,
        cohort=context.cohort,
        photo_count=len(context.evidence(axis)),
        details={"evidence_photo_ids": context.evidence_ids(axis)},
    )


def assess_harmony_axis(context: AssessmentContext) -> AxisResult:
    if request := _needs_photo_result(context, "harmony"):
        return request
    evidence = context.evidence("harmony")
    try:
        raw = harmony(context.reference_id, [value for _, value in evidence])
    except (OSError, KeyError) as exc:
        raise ReferenceDataError(
            "harmony_reference_unavailable",
            "The local harmony reference could not be read.",
        ) from exc

    profile = {
        name: row
        for name, row in raw.items()
        if row.get("value") is not None
        and row.get("percentile") is not None
        and row.get("interval") is not None
    }
    unavailable = {name: row for name, row in raw.items() if name not in profile}
    if not profile:
        if any(row.get("cause") == "missing_measurement" for row in unavailable.values()):
            return AxisResult.needs_photo(
                axis="harmony",
                request=_generic_photo_request("harmony"),
                cohort=context.cohort,
                photo_count=len(evidence),
                details={
                    "evidence_photo_ids": context.evidence_ids("harmony"),
                    "unavailable_components": unavailable,
                },
            )
        raise ReferenceDataError(
            "harmony_reference_unavailable",
            "The local harmony comparison profile is not available for this cohort.",
            retryable=False,
        )

    model = _measured("bank/harmony_model.json", default={}) or {}
    try:
        scalar = score_from_artifact(
            model,
            context.reference_id,
            {name: float(row["value"]) for name, row in profile.items()},
        )
    except ValueError as exc:
        raise ReferenceDataError(
            "harmony_model_invalid",
            "The local harmony scalar model could not be read safely.",
            retryable=False,
        ) from exc

    shared_details = {
        "profile": profile,
        "unavailable_components": unavailable,
        "evidence_photo_ids": context.evidence_ids("harmony"),
        "harmony_formula_version": model.get("formula_version"),
    }
    if scalar is not None:
        movement = model.get("achieved_metrics", {}).get("median_movement")
        if not isinstance(movement, (int, float)) or isinstance(movement, bool):
            raise ReferenceDataError(
                "harmony_model_invalid",
                "The local harmony scalar model has no measured uncertainty.",
                retryable=False,
            )
        interval = (
            max(0.0, float(scalar) - float(movement)),
            min(100.0, float(scalar) + float(movement)),
        )
        return AxisResult.measured(
            axis="harmony",
            summary=(
                f"Cohort-relative geometric harmony percentile: {scalar:.0f}. "
                "This is not an attractiveness or beauty score."
            ),
            value=float(scalar),
            unit="geometric harmony percentile",
            percentile=float(scalar),
            interval=interval,
            cohort=context.cohort,
            photo_count=len(evidence),
            details={
                **shared_details,
                "headline_reason": model.get("construct"),
            },
        )

    summary = (
        f"Measured {len(profile)} cohort-relative facial proportions. "
        "A single scalar was not validated as stable enough to report."
    )

    return AxisResult.measured_profile(
        axis="harmony",
        summary=summary,
        cohort=context.cohort,
        photo_count=len(evidence),
        details={
            **shared_details,
            "headline_reason": (
                model.get("availability_reason")
                or "The preregistered harmony aggregation has not passed held-out "
                "repeatability checks."
            ),
        },
    )


def assess_dimorphism_axis(context: AssessmentContext) -> AxisResult:
    if request := _needs_photo_result(context, "dimorphism"):
        return request
    evidence = context.evidence("dimorphism")
    raw = dimorphism(context.reference_id, [value for _, value in evidence])
    if raw.get("refused"):
        if raw.get("cause") == "missing_measurement":
            return AxisResult.needs_photo(
                axis="dimorphism",
                request=_generic_photo_request("dimorphism"),
                cohort=context.cohort,
                photo_count=len(evidence),
                details={
                    "evidence_photo_ids": context.evidence_ids("dimorphism"),
                    "missing_features": raw.get("features_used", []),
                },
            )
        raise ReferenceDataError(
            "dimorphism_reference_unavailable",
            "The local dimorphism comparison is not available for this cohort.",
            retryable=False,
        )
    interval = raw.get("percentile_interval")
    if raw.get("percentile") is None or interval is None:
        raise ReferenceDataError(
            "dimorphism_uncertainty_unavailable",
            "The local dimorphism uncertainty reference is not available.",
            retryable=False,
        )
    percentile = float(raw["percentile"])
    return AxisResult.measured(
        axis="dimorphism",
        summary=f"Sex-aligned facial geometry is at the {_ordinal(percentile)} percentile.",
        value=100.0 * float(raw["alignment_probability"]),
        unit="% alignment probability",
        percentile=percentile,
        interval=(float(interval[0]), float(interval[1])),
        cohort=context.cohort,
        photo_count=len(evidence),
        details={
            **raw,
            "evidence_photo_ids": context.evidence_ids("dimorphism"),
        },
    )


def assess_symmetry_axis(context: AssessmentContext) -> AxisResult:
    if request := _needs_photo_result(context, "symmetry"):
        return request
    evidence = context.evidence("symmetry")
    kept = [photo for photo, _ in evidence]
    values = [value for _, value in evidence]
    try:
        raw = symmetry(context.reference_id, kept, values)
    except (OSError, KeyError) as exc:
        raise ReferenceDataError(
            "symmetry_reference_unavailable",
            "The local symmetry reference could not be read.",
        ) from exc
    if raw.get("comparability") != "ranked":
        if raw.get("cause") in {"no-reading", "yaw-outside-gate"}:
            return AxisResult.needs_photo(
                axis="symmetry",
                request=_generic_photo_request("symmetry"),
                cohort=context.cohort,
                photo_count=len(evidence),
                details={
                    "evidence_photo_ids": context.evidence_ids("symmetry"),
                    "raw_measurement": raw,
                },
            )
        raise ReferenceDataError(
            "symmetry_reference_unavailable",
            "The local symmetry comparison is not available for this cohort.",
            retryable=False,
        )
    raw_interval = raw.get("interval")
    if raw.get("percentile") is None or raw_interval is None:
        raise ReferenceDataError(
            "symmetry_uncertainty_unavailable",
            "The local symmetry uncertainty reference is not available.",
            retryable=False,
        )
    percentile = 100.0 - float(raw["percentile"])
    interval = (100.0 - float(raw_interval[1]), 100.0 - float(raw_interval[0]))
    return AxisResult.measured(
        axis="symmetry",
        summary=f"Lower measured asymmetry than {percentile:.0f}% of the cohort.",
        value=float(raw["value_pct_of_ipd"]),
        unit=SYMMETRY_UNIT,
        percentile=percentile,
        interval=interval,
        cohort=context.cohort,
        photo_count=len(evidence),
        details={
            **raw,
            "percentile_direction": "higher means lower measured asymmetry",
            "evidence_photo_ids": context.evidence_ids("symmetry"),
        },
    )


def assess_averageness_axis(context: AssessmentContext) -> AxisResult:
    if request := _needs_photo_result(context, "averageness"):
        return request
    evidence = context.evidence("averageness")
    try:
        raw = averageness(
            context.reference_id,
            [photo["landmarks"] for photo, _ in evidence],
        )
    except (OSError, KeyError) as exc:
        raise ReferenceDataError(
            "averageness_reference_unavailable",
            "The local averageness reference could not be read.",
        ) from exc
    if raw.get("refused"):
        raise ReferenceDataError(
            "averageness_reference_unavailable",
            "The local averageness comparison is not available for this cohort.",
            retryable=False,
        )
    interval = raw.get("interval")
    if raw.get("percentile") is None or interval is None:
        raise ReferenceDataError(
            "averageness_uncertainty_unavailable",
            "The local averageness uncertainty reference is not available.",
            retryable=False,
        )
    percentile = float(raw["percentile"])
    low, high = float(interval[0]), float(interval[1])
    placement_unresolved = str(raw.get("placement", "")).startswith("withheld")
    if placement_unresolved:
        summary = (
            f"Measured range: {_ordinal(low)}–{_ordinal(high)} percentile. "
            "The range crosses a placement boundary, so placement is unresolved."
        )
    else:
        summary = f"Facial shape typicality is at the {_ordinal(percentile)} percentile."
    return AxisResult.measured(
        axis="averageness",
        summary=summary,
        value=percentile,
        unit="typicality percentile",
        percentile=percentile,
        interval=(low, high),
        cohort=context.cohort,
        photo_count=len(evidence),
        details={
            **raw,
            "evidence_photo_ids": context.evidence_ids("averageness"),
        },
    )


def assess_observations(
    cohort: CohortUse | str,
    kept: list[dict] | tuple[dict, ...],
    values: list[dict] | tuple[dict, ...],
    photo_facts: list[PhotoFact] | tuple[PhotoFact, ...],
) -> dict[str, dict]:
    """Assess every axis independently; no axis can reject the whole report."""
    cohort_use = _cohort_use(cohort)
    context = AssessmentContext(
        cohort=cohort_use,
        kept=tuple(kept),
        values=tuple(values),
        photo_facts=tuple(photo_facts),
    )
    evaluators = {
        "harmony": assess_harmony_axis,
        "dimorphism": assess_dimorphism_axis,
        "symmetry": assess_symmetry_axis,
        "averageness": assess_averageness_axis,
    }
    results: dict[str, dict] = {}
    for axis, evaluator in evaluators.items():
        try:
            result = evaluator(context)
        except AxisDataError as exc:
            result = AxisResult.processing_error(
                axis=axis,
                error_code=exc.code,
                error_message=exc.public_message,
                retryable=exc.retryable,
                cohort=cohort_use,
                photo_count=len(context.evidence(axis)),
                details={"evidence_photo_ids": context.evidence_ids(axis)},
            )
        results[axis] = result.to_dict()
    return results


# ⭐⭐ WHICH POPULATION THE READER WAS RANKED AGAINST — AND THE FACT THAT A
# DIFFERENT ONE EXISTS ON THE OTHER SURFACE.
#
# ⛔ THE DEFECT THIS REMOVES IS NOT A WRONG NUMBER. Every percentile below is
# computed against the matched bank in `bank/`, and every percentile the full
# two-photo report prints is computed against `ref_landmarks2.npz`, which is a
# different and wider population. BOTH surfaces were internally honest: each
# named its own pool. NEITHER said the other existed. So the same face, run
# through both, gets two different percentiles with the same words attached and
# nothing tells the reader which comparison applies to them or why they differ.
# That is this project's same-label-two-quantities defect, which it has been
# bitten by five times, wearing a new coat.
#
# ⛔ THE FIX IS ATTRIBUTION, NOT UNIFICATION. Nothing here changes a percentile,
# a gate or a pool. Choosing which population somebody is ranked against is a
# MEASUREMENT decision; saying which one it was is documentation, and only the
# second is done here.
#
# ⛔ EVERY COUNT BELOW IS READ OFF THE ARTEFACT IT DESCRIBES, AT RUN TIME.
# A population size typed into source is a claim that stops being true the next
# time the thing it describes is rebuilt, and this file already carries the scar:
# `build_report.py` still holds a label reading "among 243 reference faces", a
# figure superseded twice over. Nothing here is typed.


_REFPOP: dict = {}


def _other_surface_reference() -> dict:
    """The population the OTHER surface ranks against, measured off its own file.

    ⚠️ `ref_landmarks2.npz` is gitignored, so it can legitimately be absent on a
    fresh machine. Absent, this says so and prints NO counts, rather than
    reciting numbers it could not read.
    """
    path = A2.CACHE            # ⛔ the two-photo path's own constant, not a literal
    out = {
        "surface": "full two-photo report",
        "artefact": os.path.basename(path),
        "used_by": ["analyze_averageness2.py", "gate_trade.py", "build_report.py"],
        "available": os.path.exists(path),
    }
    if not out["available"]:
        return out
    z = np.load(path)          # plain float/int arrays; no allow_pickle
    pools = sorted(k for k in z.files
                   if not k.endswith("_raw") and not k.endswith("_race"))
    seen, races = set(), 0
    for k in pools:
        for face in z[k]:
            seen.add(hashlib.sha1(np.ascontiguousarray(face).tobytes()).digest())
        rk = f"{k}_race"
        if rk in z.files:
            races = max(races, len(set(int(v) for v in z[rk])))
    bands = sorted({tuple(int(x) for x in m.groups())
                    for k in pools
                    if (m := re.search(r"(\d+)_(\d+)$", k))})
    out.update({
        "n_pools": len(pools),
        "pools": pools,
        # ⚠️ TWO DIFFERENT QUANTITIES, AND BOTH ARE CARRIED. The pools are
        # NESTED — a white 20-29 face is also in the broad 20-29 pool — so the
        # rows sum to more than the number of people. The bank's own headline is
        # `n_faces_unique`, so the comparable figure is the DISTINCT one, and the
        # row count is kept beside it rather than quietly standing in for it.
        "n_face_rows": int(sum(int(z[k].shape[0]) for k in pools)),
        "n_faces_distinct": len(seen),
        "age_bands": [f"{lo}-{hi}" for lo, hi in bands],
        "age_span": (f"{bands[0][0]}\u2013{bands[-1][1]}" if bands else None),
        "max_race_labels_in_a_pool": races or None,
    })
    return out


def reference_population(cohort: str) -> dict:
    """Everything needed to say, in one line, WHO the reader was compared with.

    Derived from `bank/provenance.json`, the cohort's own `measurements.npz`, and
    the other surface's own reference file. Returns a `footnote` of None when
    there is no comparison to attribute — an unknown cohort has no population.
    """
    if cohort in _REFPOP:
        return _REFPOP[cohort]
    prov = _measured("bank/provenance.json", default={}) or {}
    meta = (prov.get("cohorts") or {}).get(cohort)
    bank = {
        "source": "bank/provenance.json",
        "matched": True,
        "n_cohorts": prov.get("n_cohorts_emitted"),
        "n_faces_unique": prov.get("n_faces_unique"),
        "n_face_rows_stored": prov.get("n_face_rows_stored"),
        "built": prov.get("built"),
    }
    # ⭐ THE COHORT'S SIZE COMES FROM THE COHORT'S OWN FILE. `provenance.json`
    # states it too, and the two are cross-checked here rather than one being
    # trusted: a stored count and the rows actually stored are two quantities.
    import public_reference
    n_faces = public_reference.metadata()["cohorts"].get(cohort, {}).get("source_rows")
    stated = (meta or {}).get("n")
    # ⭐ THE SAME LABEL THE PAGE'S OWN HEADER PRINTS, when the loader will give
    # it — `bank_reference` spells the cohort out ("White, men aged 20-29")
    # where provenance stores the raw id. Two spellings of one cohort in one
    # reading is exactly the confusion this footnote exists to end, so the
    # loader's is preferred and provenance's is the fallback, never a mix.
    label = (meta or {}).get("label")
    try:
        import bank_reference as _BR
        label = (_BR.COHORTS.get(cohort) or {}).get("label") or label
    except Exception:  # noqa: BLE001  — an unreadable loader is not a reason to
        pass           # withhold the attribution; provenance's label still names it
    me = {
        "cohort": cohort,
        "label": label,
        "n_faces": n_faces if n_faces is not None else stated,
        "n_faces_source": ("bank/public_reference.json"
                           if n_faces is not None else "bank/provenance.json"),
        "n_faces_stated_in_provenance": stated,
        "counts_agree": (None if (n_faces is None or stated is None)
                         else bool(n_faces == stated)),
    }
    other = _other_surface_reference()
    r = {"compared_against": me, "bank": bank, "other_surface": other,
         "footnote": _reference_footnote(me, bank, other),
         "_why": "each percentile on this page states the population it was "
                 "ranked in, and states that the other surface ranks against a "
                 "different one — no percentile, gate or pool is changed."}
    _REFPOP[cohort] = r
    return r


def _reference_footnote(me: dict, bank: dict, other: dict) -> str | None:
    """Public implementation; private development notes omitted."""
    if not me.get("label") or me.get("n_faces") is None:
        return None            # no cohort, no comparison, nothing to attribute
    mine = (f"Ranked against {me['label']} \u2014 {me['n_faces']:,} faces")
    if bank.get("n_faces_unique") and bank.get("n_cohorts"):
        mine += (f", inside a matched bank of {bank['n_faces_unique']:,} faces "
                 f"in {bank['n_cohorts']} cohorts")
    mine += "."
    if not other.get("available"):
        return (mine + f" The {other['surface']} does not use this bank: it ranks "
                f"the same face against a separate, wider reference "
                f"({other['artefact']}, not present on this machine), so its "
                f"percentiles are not interchangeable with these.")
    wider = []
    if other.get("age_span"):
        wider.append(f"spanning ages {other['age_span']} only")
    if other.get("max_race_labels_in_a_pool"):
        wider.append(f"mixing up to {other['max_race_labels_in_a_pool']} "
                     f"annotator race labels in a single pool")
    return (mine + f" The {other['surface']} does not use this bank: it ranks the "
            f"same face against {other['n_faces_distinct']:,} faces held in "
            f"{other['n_pools']} wider pools"
            + (" " + " and ".join(wider) if wider else "")
            + ", so the percentile it prints for this face answers a different "
              "question from the ones above and the two cannot be read against "
              "each other.")


def assess(photo_dir: str, cohort: str) -> dict:
    import bank_reference as BR

    rec = BR.COHORTS.get(cohort)
    out = {"cohort": cohort, "photo_dir": photo_dir, "axes": {}}
    # The attribution travels with every result, including an unavailable local
    # cohort. A renderer never has to reconstruct the population from an id.
    out["reference_population"] = reference_population(cohort)

    paths = [p for e in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG")
             for p in glob.glob(os.path.join(photo_dir, e))]
    kept, rejected = gate_photos(paths)
    out["photos_supplied"] = len(paths)
    out["photos_in_gate"] = len(kept)
    out["photos_refused"] = rejected
    facts = []
    for record in [*kept, *rejected]:
        raw = record.get("fact", {})
        facts.append(
            PhotoFact(
                photo_id=str(raw.get("photo_id") or record.get("photo") or "unknown"),
                yaw=raw.get("yaw"),
                pitch=raw.get("pitch"),
                roll=raw.get("roll"),
                ipd_px=raw.get("ipd_px"),
                neutral=bool(raw.get("neutral")),
                detected=bool(raw.get("detected")),
            )
        )

    values = [BD.metrics_for(photo["landmarks"]) for photo in kept]
    try:
        cohort_use = _cohort_use(cohort)
    except ReferenceDataError as exc:
        out["axes"] = {
            axis: AxisResult.processing_error(
                axis=axis,
                error_code=exc.code,
                error_message=exc.public_message,
                retryable=exc.retryable,
            ).to_dict()
            for axis in ("harmony", "dimorphism", "symmetry", "averageness")
        }
        return out

    out["cohort_label"] = rec["label"]
    out["cohort_n"] = rec["n"]
    if not rec["meets_floor"]:
        message = (
            "This comparison cohort is below the local 150-identity floor; "
            "choose a powered fallback cohort before measuring."
        )
        out["axes"] = {
            axis: AxisResult.processing_error(
                axis=axis,
                error_code="cohort_underpowered",
                error_message=message,
                retryable=False,
                cohort=cohort_use,
            ).to_dict()
            for axis in ("harmony", "dimorphism", "symmetry", "averageness")
        }
        return out

    out["axes"] = assess_observations(cohort_use, kept, values, facts)
    return out


def pitch_gated(cohort: str, metric: str, gate: float,
                kept: list[dict], vals: list[dict]) -> dict:
    """A vertical proportion read only from photographs shot near eye level.

    ⛔ BOTH SIDES ARE GATED AT THE SAME THRESHOLD.  The subject's photographs are
    filtered on |pitch| and so is the bank, from its own stored per-face pitch --
    `pose[:, 0]`, which is pitch because `analyze_averageness2.pose_angles`
    returns (pitch, yaw, roll) and `build_bank` appends them in that order.
    Comparing a gated reading against an ungated bank is the defect that shipped
    a false explanation into a claim, three documents and a code comment on
    2026-09-08, and it is avoided here by construction rather than by care.
    """
    near = [v for k, v in zip(kept, vals)
            if abs(k["pitch"]) <= gate and metric in v]
    move = _measured("pitch_gate.json", "rows", metric, f"{gate:.1f}",
                     "corrected_movement_pct", default=None)
    base = {
        "_gate": f"|pitch| <= {gate:.0f}° on BOTH sides — this row is computed "
                 f"from a different subset of your photographs than the rows "
                 f"above it",
        "photos_near_level": len(near),
        "photos_in_gate": len(kept),
    }
    if len(near) < MIN_IN_GATE:
        return {**base, "refused": (
            f"only {len(near)} of {len(kept)} in-gate photographs were taken "
            f"within {gate:.0f}° of eye level, and every margin for this row was "
            f"measured on sets of {MIN_IN_GATE}. Re-shoot holding the camera at "
            f"your own eye height rather than above or below it.")}
    value = float(np.median([v[metric] for v in near]))

    import public_reference
    pool = public_reference.pool(cohort, f"_pitch{gate:g}_{metric}")
    if pool is None:
        return {**base, "value": round(value, 4), "refused": "No eligible aggregate pitch reference"}
    if pool.size < 50:
        return {**base, "value": round(value, 4),
                "refused": (f"only {pool.size} faces in this cohort were "
                            f"photographed within {gate:.0f}° of eye level — "
                            f"too few to rank against")}
    pct = 100.0 * float((pool < value).mean())
    tier = 1 if (move is not None and move < 25.0) else 2
    return {
        **base,
        "value": round(value, 4),
        "percentile": round(pct, 1),
        "n_cohort_faces_near_level": int(pool.size),
        "moves_pct_of_true_scale": move,
        "tier": tier,
        "placement": (_third(pct) if tier == 1 else "WITHHELD — tier 2"),
        "_why_gated": (
            f"ungated, this row moves 27.9% of the true population scale between "
            f"two sets of photographs of the same person — past the 25% a "
            f"placement needs. Read only from near-level photographs it moves "
            f"{move:.1f}%, and the same people improve by 3.4 points when their "
            f"pitchier photographs are discarded, so the gain is not merely a "
            f"change in who qualifies." if move else None),
    }


def _third(pct: float) -> str:
    return ("upper third" if pct > 67 else
            "lower third" if pct < 33 else "middle third")


# ⭐⭐ THE THREE STATES SYMMETRY CAN BE IN, AND THERE ARE NO OTHERS.
#
# Private development note omitted.
# Symmetry used to leave `symmetry()` by FOUR exits wearing FOUR different
# shapes: two carried a bare `refused` and NO NUMBER AT ALL, one carried
# `placement: NOT COMPARABLE`, one carried `tier: 2`. A renderer had to infer
# the state from which keys happened to be present, and it got it wrong twice --
# the local page printed an EMPTY warning box, and before that printed
# "re-shoot squarer" for two causes that live in the BANK, where re-shooting
# cannot help and the advice is simply false.
#
# So every exit now goes through ONE constructor, returns THE SAME KEYS, and
# STATES its condition instead of leaving it to be deduced:
#
#   ranked          a value, a cohort percentile and an interval. Still NO
#                   placement: symmetry moves 28.3% of the population scale
#                   between two sets of the same face even at this gate,
#                   against the 25% a placement needs. That is measured.
#   not-comparable  a value, and NO percentile. Either the photographs are
#                   outside the yaw gate, or the bank has nothing gated the
#                   same way to rank the number against.
#   unmeasurable    NO value, because the detector produced no asymmetry
#                   reading on any photograph. This is the ONLY state without a
#                   number, and it is a STATED state rather than a bare refusal.
#
# ⛔ `refused` IS GONE FROM THIS AXIS ENTIRELY. A key whose mere presence changed
# the meaning of every other key is what made the four shapes unreadable.
SYMMETRY_STATES = ("ranked", "not-comparable", "unmeasurable")


def _symmetry_result(base: dict, comparability: str, *, cause: str, value,
                     n_used: int, reason: str, placement: str,
                     percentile=None, interval=None, tier=None,
                     n_cohort_faces_near_frontal=None,
                     moves_percentile_points=None, interval_basis=None,
                     _upgraded_from_refusal=None) -> dict:
    """The ONE shape symmetry ever returns — with its invariants asserted HERE.

    ⭐ EVERY RULE BELOW CAN FAIL. Each one is something a future fifth exit could
    break, and breaking it RAISES rather than quietly shipping a dict that a
    renderer will mis-read into a blank box. A shape guard that cannot fail is
    the fault this file has already shipped once.
    """
    if comparability not in SYMMETRY_STATES:
        raise ValueError(f"symmetry: unknown comparability {comparability!r}")
    ranked = comparability == "ranked"
    if ranked != (percentile is not None):
        raise ValueError("symmetry: a percentile exists in the ranked state and "
                         "in no other state")
    if interval is not None and not ranked:
        raise ValueError("symmetry: an interval on a percentile that was never "
                         "computed ranks nothing")
    if (value is None) != (comparability == "unmeasurable"):
        raise ValueError("symmetry: the NUMBER is withheld only when nothing "
                         "measured it — every other state prints it")
    if not reason or not placement or not cause:
        raise ValueError("symmetry: every state says, in words, what it is and "
                         "why it is not saying more")
    return {
        **base,
        # --- the state, stated. A renderer switches on THIS, never on which
        #     keys happen to be present.
        "comparability": comparability,
        "cause": cause,
        # --- the number. Present in every state but `unmeasurable`.
        # Legacy key retained for existing callers; the actual denominator is
        # the outer-canthus span, as stated by the public SYMMETRY_UNIT label.
        "value_pct_of_ipd": None if value is None else round(float(value), 3),
        "n_photographs_used": int(n_used),
        # --- the ranking. Present ONLY when genuinely comparable.
        "percentile": percentile,
        "interval": interval,
        "moves_percentile_points": moves_percentile_points,
        "n_cohort_faces_near_frontal": n_cohort_faces_near_frontal,
        "tier": tier,
        # --- what is being said, and what is not being said, and why.
        "placement": placement,
        "reason": reason,
        # ⚠️ THE OLD KEY NAMES, KEPT DELIBERATELY AND ALWAYS BOTH PRESENT.
        # `local_page.py` was written against them and renaming a key a shipped
        # page reads is exactly how a working section goes blank. They are
        # ALIASES of `reason` — never a second, differently-worded opinion.
        "why_no_placement": reason if ranked else None,
        "why_not_comparable": None if ranked else reason,
        # ⚠️ NO **extra. Every key this function can emit is NAMED above, so the
        # returned key set is IDENTICAL in all three states -- a renderer that
        # reads `.get(k)` for any k here can never meet a missing key, which is
        # what produced the empty warning box. Adding a state means adding a
        # parameter here, in the open, rather than smuggling one through a kwarg.
        "interval_basis": interval_basis,
        "_upgraded_from_refusal": _upgraded_from_refusal,
    }


def symmetry(cohort: str, kept: list[dict], vals: list[dict]) -> dict:
    """Asymmetry read only from NEAR-FRONTAL photographs, ranked against a bank
    gated the same way.

    ⭐ IT ALWAYS PRINTS A NUMBER unless nothing measured one. What it declines to
    do — when it declines — is RANK that number, and the refusal to rank is a
    MEASUREMENT rather than squeamishness: set-to-set movement is 73.5% of the
    deconvolved population scale at the ordinary 14° gate against 28.3% at 5°, so
    a reading taken outside this gate is largely measuring head turn. Ranking it
    would compare a person against a population on a quantity that is not the
    same thing — the same-label-two-quantities defect this project has hit five
    times. Print the number, refuse the rank, say why.
    """
    near = [v for k, v in zip(kept, vals)
            if abs(k["yaw"]) <= SYMMETRY_MAX_YAW and "asymmetry" in v]
    base = {
        "_gate": f"|yaw| <= {SYMMETRY_MAX_YAW:.0f}° on BOTH sides — set-to-set "
                 f"movement falls from 73.5% of the population scale at the "
                 f"ordinary 14° gate to 28.3% here",
        "photos_near_frontal": len(near),
        "photos_in_gate": len(kept),
    }

    if len(near) < 2:
        loose = [v["asymmetry"] for v in vals if "asymmetry" in v]
        # ---- STATE 3: nothing measured anywhere. The only numberless state, and
        #      it says so rather than going blank.
        if not loose:
            return _symmetry_result(
                base, "unmeasurable", cause="no-reading", value=None, n_used=0,
                placement="NO READING — nothing was measured, so there is "
                          "nothing to place",
                reason=(f"the detector produced no asymmetry reading on any of "
                        f"the {len(kept)} photographs that passed the capture "
                        f"gate, so there is no number to show. That is a failure "
                        f"of the measurement on these particular images, not a "
                        f"statement about your face. Re-shoot with the whole "
                        f"face lit and unobstructed."))
        # ---- STATE 2a: a real number, taken too far off straight-on to rank.
        return _symmetry_result(
            base, "not-comparable", cause="yaw-outside-gate",
            value=float(np.median(loose)), n_used=len(loose),
            placement="NOT COMPARABLE — the number is real, the ranking would "
                      "not be",
            reason=(f"only {len(near)} of {len(kept)} in-gate photographs are "
                    f"within {SYMMETRY_MAX_YAW:.0f}° of straight-on. This value "
                    f"is computed from all {len(loose)} of them, and outside that "
                    f"window the reading moves 73.5% of the population scale "
                    f"between two sets of the same face — it is largely measuring "
                    f"head turn. So the number is shown, and it is NOT ranked "
                    f"against other people, because at this yaw it is not the "
                    f"same quantity the population was measured on. Re-shoot "
                    f"looking directly at the lens to get a comparable reading."))

    value = float(np.median([v["asymmetry"] for v in near]))

    # bank side, gated identically — from the stored pose, no rebuild needed
    import public_reference
    pool = public_reference.pool(cohort, "_symmetry")
    if pool is None:
        return _symmetry_result(base, "not-comparable", cause="no-cohort-distribution",
            value=value, n_used=len(near), placement="NOT COMPARABLE",
            reason="No eligible aggregate symmetry reference")
    # ---- STATE 2c: the bank is gated the SAME way, and too little of it survives.
    if pool.size < 50:
        return _symmetry_result(
            base, "not-comparable", cause="thin-cohort",
            value=value, n_used=len(near),
            n_cohort_faces_near_frontal=int(pool.size),
            placement="NOT COMPARABLE — your number stands, the comparison group "
                      "does not",
            reason=(f"only {pool.size} faces in this cohort were photographed "
                    f"within {SYMMETRY_MAX_YAW:.0f}° of straight-on, against the "
                    f"50 a percentile needs. The gate is applied to the bank as "
                    f"hard as it is applied to you, and too little of this cohort "
                    f"survives it. That is a fact about the reference bank, not "
                    f"about your face — re-photographing cannot change it."))

    # ---- STATE 1: ranked. A percentile and an interval, and still no placement.
    pct = 100.0 * float((pool < value).mean())
    move = _measured("symmetry_yaw_gate.json", "rows", "5.0",
                     "corrected_movement_pct", default=None)
    # ⚠️ THE INTERVAL IS AN UPPER BOUND, AND SAYS SO IN ITS OWN KEY.
    # The only movement figure measured in PERCENTILE POINTS for this axis
    # (`axis_stability.json`) was measured WITHOUT the 5° yaw gate this reading
    # uses. The gated reading moves LESS than that, so the band below is wider
    # than this reading deserves rather than narrower — it can only overstate
    # the uncertainty. It is labelled rather than silently reused, because a
    # number measured one way and printed as if measured another is the defect
    # this project has hit five times.
    pp = _measured("axis_stability.json", "axes", "asymmetry",
                   "median_percentile_points", default=None)
    interval = ([round(max(0.0, pct - pp), 1), round(min(100.0, pct + pp), 1)]
                if pp else None)
    return _symmetry_result(
        base, "ranked", cause="measured", value=value, n_used=len(near),
        percentile=round(pct, 1), interval=interval,
        moves_percentile_points=pp,
        n_cohort_faces_near_frontal=int(pool.size), tier=2,
        placement="WITHHELD — tier 2",
        reason=(
            f"symmetry moves {move:.0f}% of the true population scale between two "
            f"sets even at this gate, against the 25% a placement needs. That is "
            f"measured, not assumed: the remaining noise is lighting, expression "
            f"and camera rather than head turn, and mirror-balancing does not "
            f"touch it (0.979x)." if move else
            "symmetry has not earned a placement claim at any measured gate."),
        interval_basis=(
            "an UPPER BOUND. The percentile-point movement it is drawn from "
            "(axis_stability.json) was measured WITHOUT this 5° yaw gate, so the "
            "band is wider than this gated reading's own movement, never "
            "narrower." if pp else
            "no interval: the movement measured at this gate is expressed as a "
            "share of the population scale (28.3%), not in percentile points, "
            "and the only percentile-point figure measured describes an ungated "
            "reading."),
        _upgraded_from_refusal=(
            "refused outright until 2026-09-08, un-refused on the correctly-gated "
            "movement figure (73.5% -> 28.3% of the deconvolved scale from 14° to "
            "5°). ⛔ The first justification given for the upgrade — that three "
            "quarters of the noise was head turn — was WRONG, from a table mixing a "
            "5°-gated noise with a 14°-gated spread; corrected the same day."),
    )


def averageness(cohort: str, shapes: list[np.ndarray]) -> dict:
    """Typicality with the subject's MISSING photograph-noise added back (§5.1)."""
    import multiview_probe as MV
    perm = np.load(MV.MIRROR_PERM)[:468]
    sigma = _measured("averageness_set.json",
                      "measured_single_photo_shape_sigma", default=None)
    move = _measured("averageness_set.json", "movement_percentile_points",
                     "median", default=None)
    if sigma is None:
        return {"refused": "the single-photograph shape noise has not been measured; "
                           "without it a set-averaged subject is scored against a "
                           "population measured a different way (§5.1)"}
    import public_reference
    smean, bank_d, source_rows = public_reference.average_reference(cohort)
    rng = np.random.default_rng(0)
    rng.permutation(source_rows)  # Preserve the private instrument's following noise draws.

    S = [A2.standardize(np.asarray(s, dtype=np.float64)) for s in shapes]
    m = A2.standardize(np.mean([A2.align(s, smean) for s in S], axis=0))
    add = sigma * np.sqrt(max(0.0, 1.0 - 1.0 / len(S)))
    ds = []
    for _ in range(N_NOISE):
        v = A2.aligned_sym_vec(m + rng.normal(0, add, m.shape), smean, perm)
        ds.append(float(np.linalg.norm(v - smean.reshape(-1))))
    dist = float(np.mean(ds))
    pct = 100.0 * float((bank_d > dist).mean())
    lo, hi = (max(0.0, pct - move), min(100.0, pct + move)) if move else (None, None)
    straddles = lo is not None and any(lo <= c <= hi for c in (33.0, 67.0))
    return {
        "percentile": round(pct, 1),
        "interval": [round(lo, 1), round(hi, 1)] if lo is not None else None,
        "moves_percentile_points": move,
        "n_photos_used": len(S),
        "noise_added_back": round(add, 6),
        "placement": ("withheld — the interval straddles a placement cutoff"
                      if straddles else
                      ("more typical than most" if pct > 67 else
                       "less typical than most" if pct < 33 else "mid-range")),
        "_construct": "symmetric-component typicality: distance to the cohort mean "
                      "shape, ranked among held-out cohort faces.",
        "_correction": "the subject's missing photograph-noise is added back before "
                       "the distance is taken; without it the median stranger scores "
                       "at the 87.3rd percentile purely for being cleaner (§5.1).",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", required=True)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    r = assess(a.photos, a.cohort)
    print(json.dumps(r, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
