"""Turn structured photo evidence gaps into one useful next-photo request."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import isfinite

from .contracts import AXES, PhotoRequest

MIN_ELIGIBLE_PHOTOS = 4
MIN_IPD_PX = 120.0
GENERAL_MAX_YAW = 14.0
SYMMETRY_MAX_YAW = 5.0
MAX_PITCH = 16.0
MAX_ROLL = 25.0


@dataclass(frozen=True)
class PhotoFact:
    photo_id: str
    yaw: float | None
    pitch: float | None
    roll: float | None
    ipd_px: float | None
    neutral: bool
    detected: bool = True
    sharp: bool = True
    evenly_lit: bool = True
    unoccluded: bool = True

    def failures_for(self, axis: str) -> tuple[str, ...]:
        if axis not in AXES:
            raise ValueError(f"unknown analysis axis: {axis}")
        failures: list[str] = []
        if not self.detected:
            return ("face_not_detected",)
        if self.ipd_px is None or not isfinite(self.ipd_px) or self.ipd_px < MIN_IPD_PX:
            failures.append("too_small")
        if not self.sharp:
            failures.append("blurred")
        if not self.evenly_lit:
            failures.append("uneven_light")
        if not self.unoccluded:
            failures.append("occluded")
        if not self.neutral:
            failures.append("expression")
        if self.yaw is None or not isfinite(self.yaw):
            failures.append("pose_unknown")
        else:
            yaw_limit = SYMMETRY_MAX_YAW if axis == "symmetry" else GENERAL_MAX_YAW
            if abs(self.yaw) > yaw_limit:
                failures.append("yaw")
        if self.pitch is None or not isfinite(self.pitch) or abs(self.pitch) > MAX_PITCH:
            failures.append("pitch")
        if self.roll is None or not isfinite(self.roll) or abs(self.roll) > MAX_ROLL:
            failures.append("roll")
        return tuple(failures)

    def eligible_for(self, axis: str) -> bool:
        return not self.failures_for(axis)


def request_for_axis(
    axis: str,
    facts: list[PhotoFact] | tuple[PhotoFact, ...],
    *,
    minimum: int = MIN_ELIGIBLE_PHOTOS,
) -> PhotoRequest | None:
    if axis not in AXES:
        raise ValueError(f"unknown analysis axis: {axis}")
    if minimum <= 0:
        raise ValueError("minimum eligible photo count must be positive")

    eligible = sum(photo.eligible_for(axis) for photo in facts)
    if eligible >= minimum:
        return None

    failures = Counter(
        failure
        for photo in facts
        for failure in photo.failures_for(axis)
    )
    priority = (
        "face_not_detected",
        "too_small",
        "yaw",
        "pitch",
        "roll",
        "expression",
        "uneven_light",
        "blurred",
        "occluded",
        "pose_unknown",
    )
    reason_code = max(priority, key=lambda code: (failures[code], -priority.index(code)))
    if not failures[reason_code]:
        reason_code = "face_not_detected"

    needed = minimum - eligible
    shortfall = f"{needed} more eligible photo" + ("" if needed == 1 else "s")
    axis_name = axis.capitalize()
    base = {
        "view": "straight-on",
        "expression": "neutral, lips relaxed",
        "lighting": "even light across the whole face",
        "framing": "head and shoulders",
    }

    explanations = {
        "face_not_detected": (
            f"{axis_name} needs {shortfall} with the face clearly visible."
        ),
        "too_small": (
            f"{axis_name} needs {shortfall} with the eyes at least 120 pixels apart."
        ),
        "yaw": (
            f"{axis_name} needs {shortfall} shot straight-on so both sides are comparable."
        ),
        "pitch": (
            f"{axis_name} needs {shortfall} with the camera held at eye level."
        ),
        "roll": (
            f"{axis_name} needs {shortfall} with both eyes level."
        ),
        "expression": (
            f"{axis_name} needs {shortfall} with a neutral, relaxed expression."
        ),
        "uneven_light": (
            f"{axis_name} needs {shortfall} in even light without a strong side shadow."
        ),
        "blurred": (
            f"{axis_name} needs {shortfall} that are sharp rather than blurred."
        ),
        "occluded": (
            f"{axis_name} needs {shortfall} with the full face unobstructed."
        ),
        "pose_unknown": (
            f"{axis_name} needs {shortfall} where the face angle can be measured."
        ),
    }

    if reason_code == "too_small":
        base["framing"] = "head and shoulders; move closer"
    elif reason_code == "pitch":
        base["view"] = "straight-on at eye level"
    elif reason_code == "roll":
        base["view"] = "straight-on with both eyes level"
    elif reason_code == "uneven_light":
        base["lighting"] = "soft, even light; no strong side shadow"
    elif reason_code == "blurred":
        base["framing"] = "head and shoulders; hold the camera steady"
    elif reason_code == "occluded":
        base["framing"] = "full face visible; remove obstructions"

    return PhotoRequest(reason=explanations[reason_code], **base)
