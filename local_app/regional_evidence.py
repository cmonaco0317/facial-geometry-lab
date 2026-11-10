"""Conservative eye-only estimate selection; no inferred precision weights."""

from __future__ import annotations

from math import isfinite
from numbers import Real
from typing import Any

EYE_METRICS = frozenset({"eye_spacing", "eye_almondness", "canthal_tilt"})
REGIONAL_VERSION = "automatic-regional-v2"


def _pucker_only(expression: Any) -> bool:
    """Use existing QC cutoffs, requiring every relevant model output to exist.

    These cutoffs are QC heuristics, not calibrated regional uncertainty.
    A smile, open jaw, blink or squint does not qualify for this narrow exception.
    """
    import averageness_band
    from capture_check import EXPRESSION

    keys = {
        "mouthSmileLeft", "mouthSmileRight", "jawOpen",
        "eyeBlinkLeft", "eyeBlinkRight",
        "eyeSquintLeft", "eyeSquintRight",
    }
    limits = {key: limit for key, limit, _ in EXPRESSION if key in keys}
    if not isinstance(expression, dict) or len(limits) != len(keys):
        return False
    required = keys | {"mouthPucker"}
    if any(
        isinstance(expression.get(key), bool)
        or not isinstance(expression.get(key), Real)
        or not isfinite(expression[key])
        or not 0 <= expression[key] <= 1
        for key in required
    ):
        return False
    return expression["mouthPucker"] > averageness_band.MOUTH_PUCKER_GATE and all(
        expression[key] <= limit for key, limit in limits.items()
    )


def metric_evidence(
    context: Any, primary_ids: tuple[str, ...], metric: str
) -> tuple[tuple[dict, dict], ...]:
    """Keep strict evidence first; expand only observed eye-region estimates.

    Mouth/outline measurements and all four axis evaluators keep their original
    evidence. No out-of-pose photo is added to an eye estimate. Correlated images
    are not treated as independent samples or used to fabricate an interval.
    """
    pairs = tuple(zip(context.kept, context.values))
    primary = tuple(pair for pair in pairs if pair[0]["photo"] in primary_ids)
    if not primary or metric not in EYE_METRICS:
        return primary
    facts = {fact.photo_id: fact for fact in getattr(context, "photo_facts", ())}
    if any(
        (fact := facts.get(photo["photo"])) is not None
        and fact.eligible_for("harmony")
        for photo, _ in primary
    ):
        return primary
    candidates = []
    for photo, value in pairs:
        fact = facts.get(photo["photo"])
        if fact is None:
            continue
        failures = set(fact.failures_for("harmony"))
        if failures <= {"expression", "too_small"} and (
            "expression" not in failures or _pucker_only(photo.get("expression"))
        ):
            candidates.append((photo, value))
    return tuple(candidates) or primary
