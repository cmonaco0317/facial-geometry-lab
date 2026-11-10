"""Best-effort product readings; strict research gates remain in assess_set."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from .contracts import AxisResult, SYMMETRY_UNIT
from .regional_evidence import REGIONAL_VERSION, metric_evidence

ESTIMATE_VERSION = "photo-estimate-v2"


def _median(values: list[dict], key: str) -> float | None:
    numbers = [
        float(row[key])
        for row in values
        if isinstance(row.get(key), (int, float, np.number))
        and not isinstance(row[key], bool)
        and np.isfinite(row[key])
    ]
    return float(np.median(numbers)) if numbers else None


def _percentile(value: float, pool: Any, *, reverse: bool = False) -> float | None:
    if pool is None:
        return None
    finite = np.asarray(pool, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return None
    # Match the strict evaluator's definition so gaining photos cannot change it.
    rank = 100 * float(np.mean(finite < value))
    return round(100 - rank if reverse else rank, 1)


def _pool(assessment: Any, cohort: str, metric: str) -> Any:
    try:
        return assessment.cohort_pool(cohort, metric)
    except (OSError, KeyError, ValueError):
        return None


def _symmetry_pool(assessment: Any, cohort: str) -> Any:
    """Use the same near-frontal comparison population at every photo count."""
    import public_reference
    return public_reference.pool(cohort, "_symmetry")


def estimate_axis(axis: str, context: Any, photo_ids: tuple[str, ...]) -> AxisResult:
    """Calculate from real detected landmarks, never from substituted cohort means."""
    import assess_set as assessment

    pairs = [
        (p, v) for p, v in zip(context.kept, context.values) if p["photo"] in photo_ids
    ]
    if not pairs:
        return AxisResult.needs_photo(
            axis=axis,
            cohort=context.cohort,
            request=assessment._generic_photo_request(axis),
            photo_count=0,
        )
    facts = [f for f in context.photo_facts if f.photo_id in photo_ids]
    issues = dict(Counter(issue for f in facts for issue in f.failures_for(axis)))
    if len(pairs) < 4:
        issues["limited_photo_count"] = len(pairs)
    details = {
        "method": ESTIMATE_VERSION,
        "capture_issues": issues,
        "evidence_photo_ids": list(photo_ids),
        "estimate_note": (
            "Best-effort reading from detected geometry. Pose, expression and image "
            "detail can affect it. A calibrated uncertainty range is not available "
            "for these capture conditions. When only detail fails, compatible "
            "repeat observations are combined; this does not make them measured."
        ),
    }
    values = [v for _, v in pairs]
    common = dict(
        axis=axis,
        status="estimated",
        cohort=context.cohort,
        photo_count=len(pairs),
        details=details,
    )
    if axis == "harmony":
        profile = {}
        for name, label in assessment.HA.AXES.items():
            value = _median(values, "harm__" + name)
            if value is None:
                continue
            profile[name] = dict(
                label=label,
                value=round(value, 4),
                interval=None,
                percentile=_percentile(
                    value, _pool(assessment, context.reference_id, "harm__" + name)
                ),
            )
        if profile:
            return AxisResult(
                **{**common, "details": {**details, "profile": profile}},
                headline_available=False,
                summary=f"Estimated {len(profile)} proportions from the available photos.",
            )
    elif axis == "symmetry":
        value = _median(values, "asymmetry")
        if value is not None:
            percentile = _percentile(
                value, _symmetry_pool(assessment, context.reference_id), reverse=True
            )
            return AxisResult(
                **common,
                value=value,
                unit=SYMMETRY_UNIT,
                percentile=percentile,
                summary="Estimated symmetry from the available facial geometry.",
            )
    elif axis in {"dimorphism", "averageness"}:
        try:
            raw = (
                assessment.dimorphism(context.reference_id, values)
                if axis == "dimorphism"
                else assessment.averageness(
                    context.reference_id, [p["landmarks"] for p, _ in pairs]
                )
            )
        except (OSError, KeyError, ValueError) as exc:
            raise assessment.ReferenceDataError(
                f"{axis}_reference_unavailable",
                f"The local {axis} reference could not be read.",
            ) from exc
        if not raw.get("refused") and raw.get("percentile") is not None:
            value = (
                100 * float(raw["alignment_probability"])
                if axis == "dimorphism"
                else float(raw["percentile"])
            )
            return AxisResult(
                **common,
                value=value,
                percentile=float(raw["percentile"]),
                unit="% alignment probability"
                if axis == "dimorphism"
                else "typicality percentile",
                summary=f"Estimated {axis} from the available photos.",
            )
        if raw.get("cause") == "missing_measurement":
            return AxisResult.needs_photo(
                axis=axis,
                cohort=context.cohort,
                request=assessment._generic_photo_request(axis),
                photo_count=len(pairs),
                details={"missing_features": raw.get("features_used", [])},
            )
        raise assessment.ReferenceDataError(
            f"{axis}_reference_unavailable",
            f"The local {axis} comparison is unavailable for this reference group.",
            retryable=False,
        )
    return AxisResult.needs_photo(
        axis=axis,
        cohort=context.cohort,
        request=assessment._generic_photo_request(axis),
        photo_count=len(pairs),
    )


def regional_measurements(
    context: Any, photo_ids: tuple[str, ...], *,
    forehead_clear_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Extract the existing instrument's ratios, without importing legacy verdicts."""
    import assess_set as assessment
    import facial_catalogue

    rows = []
    for name, label, _formula, unit, *_ in facial_catalogue.CATALOGUE:
        pairs = metric_evidence(context, photo_ids, name)
        if forehead_clear_ids is not None:
            from .automatic_forehead import FOREHEAD_METRICS

            if name in FOREHEAD_METRICS:
                pairs = tuple((p, v) for p, v in pairs if p["photo"] in forehead_clear_ids)
        values = [v for _, v in pairs]
        value = _median(values, name)
        contributing_ids = [
            p["photo"] for p, v in pairs
            if isinstance(v.get(name), (int, float, np.number))
            and not isinstance(v[name], bool) and np.isfinite(v[name])
        ]
        if name == "canthal_tilt":
            angles = []
            contributing_ids = []
            for photo, _ in pairs:
                points = photo["landmarks"]
                # A collapsed eye cannot supply an angle; atan2(0, 0) is not evidence.
                if any(abs(float(points[a][0] - points[b][0])) < 1e-8
                       for a, b in ((33, 133), (263, 362))):
                    continue
                angle = facial_catalogue.base(points)["canthal_tilt_deg"]
                if np.isfinite(angle):
                    angles.append(angle)
                    contributing_ids.append(photo["photo"])
            value = float(np.median(angles)) if angles else None
        if unit not in {"ratio", "deg"} or value is None:
            continue
        # The detector's top point is a forehead proxy, not a detected hairline.
        proxy = name in {"facial_index", "upper_third", "middle_third", "lower_third"}
        rows.append(
            dict(
                id=name,
                label=label,
                value=round(value, 4),
                unit=unit,
                method=REGIONAL_VERSION if name in {"eye_spacing", "eye_almondness", "canthal_tilt"}
                else "automatic-regional-v1",
                evidence_photo_count=len(contributing_ids),
                evidence_photo_ids=contributing_ids,
                percentile=_percentile(
                    value, _pool(assessment, context.reference_id, name)
                ),
                note=(
                    "Uses the mesh forehead boundary; not a measured hairline."
                    if proxy
                    else "Image-derived geometry; camera angle and expression can affect this ratio."
                ) + (
                    " Estimated eye-region geometry combines pose-compatible observations; "
                    "mouth-posture flags remain excluded from mouth-sensitive readings. "
                    "Regional uncertainty is not calibrated."
                    if {p["photo"] for p, _ in pairs} != set(photo_ids) else ""
                ),
            )
        )
    return rows
