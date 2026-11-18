"""Automatic turned-view mesh proxies for profile geometry, without norms."""

from __future__ import annotations

import hashlib
import json
import math
import re
from numbers import Real
from typing import Any

from .observations import Observation, Point, SourceEvidence
from .profile_v2 import ROWS, contract as manual_profile_contract, measure

MESH_PROFILE_METHOD = "mesh-profile-proxy-v1"
MESH_PROFILE_FAMILY = "mesh_profile_proxy"

ANCHOR_INDICES = {
    "nasion": 168,
    "tip": 4,
    "subnasale": 2,
    "upper_lip": 0,
    "lower_lip": 17,
    "pogonion": 200,
    "menton": 152,
}

ANCHOR_NOTES = {
    "nasion": "p168 mesh bridge proxy",
    "tip": "p4 mesh nose-tip proxy",
    "subnasale": "p2 mesh nose-base proxy",
    "upper_lip": "p0 mesh upper-lip proxy",
    "lower_lip": "p17 mesh lower-lip proxy",
    "pogonion": "p200 fixed mesh chin proxy, not true pogonion",
    "menton": "p152 mesh lower-chin proxy",
}

PROFILE_PROXY_FORMULAS = {
    "convexity": "degrees(acos(dot(N-Sn,Pog_proxy-Sn) / (D(N,Sn)*D(Pog_proxy,Sn))))",
    "total_convexity": "degrees(acos(dot(N-Prn,Pog_proxy-Prn) / (D(N,Prn)*D(Pog_proxy,Prn))))",
    "upper_eline": "100 * offset(upper_lip,Prn,Pog_proxy) / D(N,Me)",
    "lower_eline": "100 * offset(lower_lip,Prn,Pog_proxy) / D(N,Me)",
    "nasal_projection_ratio": "abs(perp(Prn,N,Sn)) / D(N,Me)",
    "nasal_length_ratio": "D(N,Prn) / D(N,Me)",
    "lower_face_ratio": "D(Sn,Me) / D(N,Me)",
    "lip_offset_difference": "100 * (offset(upper_lip,Prn,Pog_proxy)-offset(lower_lip,Prn,Pog_proxy)) / D(N,Me)",
}

FORMULA_PREFIX = (
    "mapping: N=p168, Prn=p4, Sn=p2, upper_lip=p0, lower_lip=p17, "
    "Pog_proxy=p200 fixed mesh chin proxy (not true pogonion), Me=p152; "
)
REFERENCE_POLICY = (
    "No reference distribution, norms, percentile, ranking, score eligibility or "
    "anatomical accuracy claim. This is a report-only three-quarter/turned-view "
    "mesh proxy that does not replace independent human placement."
)
SCORE_EXCLUSION_REASON = (
    "Mesh profile proxies are not admitted to any score, rank, norm or percentile "
    "instrument and do not claim anatomical equivalence. Model suggestions may shift when an image is mirrored; no calibrated accuracy is available."
)
HEURISTICS = {
    "orientation_normalization": "Left-turned inputs use face-right inference; all proposed coordinates and pose map back to the original image.",
    "canonical_left_trigger_yaw_degrees": -20.0,
    "accepted_landmark_lengths": [468, 478],
    "yaw_abs_degrees_min": 35.0,
    "yaw_abs_degrees_max": 85.0,
    "pitch_abs_degrees_max": 20.0,
    "roll_abs_degrees_max": 20.0,
    "min_finite_model_bbox_height_px": 80.0,
    "facing_rule": "from p4 nose-tip x relative to p168 bridge x after mapping back to the original image; no mirroring of exported coordinates",
}
_HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")


class _DetectionUnavailable(Exception):
    """A whole-frame routing failure, not a row-level missing anchor."""


def _json_sha256(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def _validate_model_hash(model_sha256: str | None) -> str | None:
    if model_sha256 is None:
        return None
    if not isinstance(model_sha256, str) or not _HEX64.fullmatch(model_sha256):
        raise ValueError("model_sha256 must be a 64-character SHA-256 hex digest.")
    return model_sha256.lower()


def _proxy_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for row in ROWS:
        source_id = row["id"]
        proxy_id = f"mesh_profile_proxy_{source_id}"
        requires = tuple(row["requires"])
        input_names = [f"p{ANCHOR_INDICES[name]}" for name in requires]
        definition = (
            f"Three-quarter/turned-view face-mesh proxy for reviewed profile "
            f"{source_id}. Uses {', '.join(input_names)} from the original image; "
            "p200 is a fixed mesh chin proxy, not true pogonion. No anatomical "
            "equivalence, independent human placement, evaluated uncertainty, "
            "norm, percentile or calibration is claimed."
        )
        specs.append(
            {
                "requires": requires,
                "definition": {
                    "key": f"{MESH_PROFILE_FAMILY}:{proxy_id}",
                    "family": MESH_PROFILE_FAMILY,
                    "id": proxy_id,
                    "label": f"Mesh-profile proxy: {row['label']}",
                    "definition": definition,
                    "formula": FORMULA_PREFIX + PROFILE_PROXY_FORMULAS[source_id],
                    "unit": row["unit"],
                    "inputs": [ANCHOR_INDICES[name] for name in requires],
                    "method": MESH_PROFILE_METHOD,
                    "display_decimals": 4,
                    "proxy_for": f"reviewed_profile:{source_id}",
                    "reference_policy": REFERENCE_POLICY,
                },
            }
        )
    return specs


def profile_proxy_definitions() -> list[dict]:
    """Return fresh metadata rows without importing the reading registry."""

    return [dict(spec["definition"]) for spec in _proxy_specs()]


def profile_proxy_contract(model_sha256: str | None) -> str:
    """Hash the immutable proxy instrument identity supplied by the caller."""

    model_hash = _validate_model_hash(model_sha256)
    return _json_sha256(
        {
            "method": MESH_PROFILE_METHOD,
            "family": MESH_PROFILE_FAMILY,
            "model_sha256": model_hash,
            "anchor_indices": ANCHOR_INDICES,
            "anchor_notes": ANCHOR_NOTES,
            "definitions": profile_proxy_definitions(),
            "heuristics": HEURISTICS,
            "manual_numeric_engine_contract": manual_profile_contract(),
        }
    )


def _validate_metadata(
    *,
    width: int,
    height: int,
    photo_id: str,
    photo_number: int,
    image_revision: int,
) -> None:
    if (
        type(width) is not int
        or type(height) is not int
        or not 1 <= width <= 100000
        or not 1 <= height <= 100000
    ):
        raise ValueError("Profile proposal dimensions must be valid source pixels.")
    if not isinstance(photo_id, str) or not re.fullmatch(
        r"[A-Za-z0-9_-]{1,96}", photo_id
    ):
        raise ValueError("Profile proposal photo_id must be a stable safe token.")
    if type(photo_number) is not int or not 1 <= photo_number <= 10000:
        raise ValueError("Profile proposal photo_number must be a report ordinal.")
    if type(image_revision) is not int or image_revision < 1:
        raise ValueError("Profile proposal image_revision must be a positive integer.")


def _finite_pose(value: float | None) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    try:
        value = float(value)
    except (OverflowError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _length(value: Any) -> int | None:
    try:
        return len(value)
    except TypeError:
        return None


def _point_pair(raw: Any) -> tuple[float, float]:
    if isinstance(raw, (str, bytes)) or _length(raw) != 2:
        raise _DetectionUnavailable(
            "Malformed geometry: each landmark must contain x and y coordinates."
        )
    try:
        x, y = raw[0], raw[1]
    except (KeyError, IndexError, TypeError) as exc:
        raise _DetectionUnavailable(
            "Malformed geometry: expected an indexed numeric x/y pair."
        ) from exc
    if (
        isinstance(x, bool)
        or isinstance(y, bool)
        or not isinstance(x, Real)
        or not isinstance(y, Real)
    ):
        raise _DetectionUnavailable(
            "Malformed geometry: landmark coordinates must be numeric."
        )
    try:
        return float(x), float(y)
    except (OverflowError, ValueError) as exc:
        raise _DetectionUnavailable(
            "Malformed geometry: coordinate conversion failed."
        ) from exc


def _landmark_sequence(landmarks: Any) -> Any:
    if landmarks is None:
        raise _DetectionUnavailable("No compatible face detections were supplied.")
    if isinstance(landmarks, (str, bytes)) or _length(landmarks) not in (468, 478):
        raise _DetectionUnavailable(
            "No compatible face detections: expected 468x2 or 478x2 landmark array."
        )
    return landmarks


def _extract_points(
    landmarks: Any, *, width: int, height: int
) -> tuple[dict[str, list[float] | None], dict[str, list[float] | None]]:
    sequence = _landmark_sequence(landmarks)
    finite_inside: list[tuple[float, float]] = []
    parsed: list[tuple[float, float]] = []
    for raw in sequence:
        x, y = _point_pair(raw)
        parsed.append((x, y))
        if (
            math.isfinite(x)
            and math.isfinite(y)
            and 0 <= x <= width
            and 0 <= y <= height
        ):
            finite_inside.append((x, y))
    if not finite_inside:
        raise _DetectionUnavailable("No compatible face detections had finite points.")
    bbox_height = max(y for _x, y in finite_inside) - min(y for _x, y in finite_inside)
    if bbox_height < HEURISTICS["min_finite_model_bbox_height_px"]:
        raise _DetectionUnavailable(
            "The detected mesh has limited image detail: finite model bbox height is below 80px."
        )

    pixels: dict[str, list[float] | None] = {}
    normalized: dict[str, list[float] | None] = {}
    for name, index in ANCHOR_INDICES.items():
        x, y = parsed[index]
        if (
            math.isfinite(x)
            and math.isfinite(y)
            and 0 <= x <= width
            and 0 <= y <= height
        ):
            pixels[name] = [x, y]
            normalized[name] = [x / width, y / height]
        else:
            pixels[name] = None
            normalized[name] = None
    return pixels, normalized


def _empty_points() -> dict[str, None]:
    return {name: None for name in ANCHOR_INDICES}


def _facing(pixels: dict[str, list[float] | None]) -> str | None:
    nasion, tip = pixels.get("nasion"), pixels.get("tip")
    if nasion is None or tip is None:
        return None
    dx = tip[0] - nasion[0]
    if abs(dx) < 1e-8:
        return None
    return "right" if dx > 0 else "left"


def _source_points(
    requires: tuple[str, ...], points: dict[str, list[float] | None]
) -> tuple[Point, ...]:
    return tuple(
        Point(name=f"p{ANCHOR_INDICES[name]}", x=points[name][0], y=points[name][1])
        for name in requires
        if points[name] is not None
    )


def _source(
    *,
    photo_id: str,
    photo_number: int,
    width: int,
    height: int,
    image_revision: int,
    facing: str,
    requires: tuple[str, ...],
    points: dict[str, list[float] | None],
) -> SourceEvidence:
    return SourceEvidence(
        photo_id=photo_id,
        photo_number=photo_number,
        view=f"profile_{facing}",
        width=width,
        height=height,
        revision=image_revision,
        review_revision=0,
        points=_source_points(requires, points),
        visibility="unassessed",
        capture_eligible=False,
    )


def _missing_reason(
    requires: tuple[str, ...], points: dict[str, list[float] | None]
) -> str:
    missing = [name for name in requires if points[name] is None]
    details = [ANCHOR_NOTES[name] for name in missing]
    return "Missing required mesh anchors: " + ", ".join(details) + "."


def _observations(
    *,
    contract: str,
    values: dict[str, float],
    unavailable: dict[str, str],
    global_reason: str | None,
    source_context: dict[str, Any] | None,
    points: dict[str, list[float] | None],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for spec in _proxy_specs():
        definition = spec["definition"]
        requires = spec["requires"]
        target_id = definition["proxy_for"].split(":", 1)[1]
        value = values.get(target_id)
        source = (
            _source(requires=requires, points=points, **source_context)
            if source_context is not None
            else None
        )
        if value is not None:
            record = Observation(
                definition_key=definition["key"],
                definition_version=contract,
                label=definition["label"],
                definition=definition["definition"],
                formula=definition["formula"],
                method=MESH_PROFILE_METHOD,
                unit=definition["unit"],
                status="proxy",
                value=value,
                revision=1,
                sources=(source,),
                display_decimals=definition["display_decimals"],
                proxy_for=definition["proxy_for"],
                score_eligible=False,
                score_exclusion_reason=SCORE_EXCLUSION_REASON,
            )
        else:
            if global_reason is not None:
                reason = global_reason
            elif unavailable.get(target_id) == "missing_anchors":
                reason = _missing_reason(requires, points)
            else:
                reason = (
                    "Mesh proxy points do not define finite arithmetic for this formula; "
                    "no default geometry was substituted."
                )
            record = Observation(
                definition_key=definition["key"],
                definition_version=contract,
                label=definition["label"],
                definition=definition["definition"],
                formula=definition["formula"],
                method=MESH_PROFILE_METHOD,
                unit=definition["unit"],
                status="unavailable",
                value=None,
                revision=1,
                sources=(source,) if source is not None else (),
                display_decimals=definition["display_decimals"],
                withheld_reason=reason,
                score_eligible=False,
                score_exclusion_reason=SCORE_EXCLUSION_REASON,
            )
        result.append(record.model_dump(mode="json"))
    return result


def _proposal_id(
    *,
    contract: str,
    photo_id: str,
    photo_number: int,
    width: int,
    height: int,
    image_revision: int,
    model_sha256: str | None,
    points: dict[str, list[float] | None],
    pose: dict[str, float | None],
) -> str:
    return _json_sha256(
        {
            "contract": contract,
            "photo_id": photo_id,
            "photo_number": photo_number,
            "width": width,
            "height": height,
            "image_revision": image_revision,
            "model_sha256": model_sha256,
            "points": points,
            "pose": pose,
        }
    )


def propose_profile(
    landmarks: Any,
    *,
    width: int,
    height: int,
    photo_id: str,
    photo_number: int,
    yaw: float | None,
    pitch: float | None,
    roll: float | None,
    model_sha256: str | None,
    image_revision: int = 1,
) -> dict:
    """Build JSON-safe mesh profile proxy proposals from detector coordinates."""

    _validate_metadata(
        width=width,
        height=height,
        photo_id=photo_id,
        photo_number=photo_number,
        image_revision=image_revision,
    )
    model_hash = _validate_model_hash(model_sha256)
    contract = profile_proxy_contract(model_hash)
    pose = {
        "yaw": _finite_pose(yaw),
        "pitch": _finite_pose(pitch),
        "roll": _finite_pose(roll),
    }
    points: dict[str, list[float] | None] = _empty_points()
    pixels: dict[str, list[float] | None] = _empty_points()
    facing: str | None = None
    global_reason: str | None = None
    values: dict[str, float] = {}
    unavailable: dict[str, str] = {}
    source_context: dict[str, Any] | None = None

    try:
        pixels, points = _extract_points(landmarks, width=width, height=height)
        facing = _facing(pixels)
    except _DetectionUnavailable as exc:
        global_reason = str(exc)

    if global_reason is None:
        if model_hash is None:
            global_reason = "Missing model identity: caller must supply the actual detector model SHA-256."
        elif any(value is None for value in pose.values()):
            global_reason = (
                "Missing finite yaw, pitch and roll pose values for routing."
            )
        elif not (
            HEURISTICS["yaw_abs_degrees_min"]
            <= abs(pose["yaw"])
            <= HEURISTICS["yaw_abs_degrees_max"]
        ):
            global_reason = (
                "Unsupported view for this routing heuristic: expected a turned profile "
                "with 35 <= abs(yaw) <= 85 degrees."
            )
        elif (
            abs(pose["pitch"]) > HEURISTICS["pitch_abs_degrees_max"]
            or abs(pose["roll"]) > HEURISTICS["roll_abs_degrees_max"]
        ):
            global_reason = (
                "Unsupported head pose for this routing heuristic: pitch and roll must "
                "be within 20 degrees."
            )
        elif facing is None:
            global_reason = (
                "Cannot establish profile facing from p4 nose tip relative to p168 bridge; "
                "no yaw-sign fallback was used."
            )

    if global_reason is None:
        payload = measure(points, width, height, facing=facing)
        values = {row["id"]: row["value"] for row in payload["measurements"]}
        unavailable = dict(payload["unavailable"])
        source_context = {
            "photo_id": photo_id,
            "photo_number": photo_number,
            "width": width,
            "height": height,
            "image_revision": image_revision,
            "facing": facing,
        }
        if not values:
            global_reason = (
                "Mesh proxy geometry did not define any finite profile measurements; "
                "no default geometry was substituted."
            )
            source_context = None

    observations = _observations(
        contract=contract,
        values=values,
        unavailable=unavailable,
        global_reason=global_reason,
        source_context=source_context,
        points=points,
    )
    status = "candidate" if global_reason is None else "unavailable"
    reason = global_reason or "eligible_profile_mesh_proxy"
    return {
        "method": MESH_PROFILE_METHOD,
        "contract": contract,
        "proposal_id": _proposal_id(
            contract=contract,
            photo_id=photo_id,
            photo_number=photo_number,
            width=width,
            height=height,
            image_revision=image_revision,
            model_sha256=model_hash,
            points=points,
            pose=pose,
        ),
        "photo_id": photo_id,
        "photo_number": photo_number,
        "width": width,
        "height": height,
        "image_revision": image_revision,
        "model_sha256": model_hash,
        "facing": facing,
        "status": status,
        "reason": reason,
        "points": points,
        "mesh_indices": dict(ANCHOR_INDICES),
        "pose": pose,
        "observations": observations,
    }
