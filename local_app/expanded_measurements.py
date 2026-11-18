"""Checkpoint 5 expanded proxy definitions and pure observation helpers.

These helpers create report-only mesh/reviewed 2D constructions. They do not
claim anatomical accuracy, physical scale, norms, ranks or score eligibility.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from copy import deepcopy
from numbers import Real
from pathlib import Path
from statistics import median
from typing import Any

from .observations import Observation, Point, SourceEvidence

EXPANDED_FRONTAL_MESH_METHOD = "expanded-frontal-mesh-v1"
EXPANDED_PROFILE_MESH_METHOD = "expanded-profile-mesh-v1"
EXPANDED_PROFILE_REVIEWED_METHOD = "expanded-profile-reviewed-v1"

FRONTAL_FAMILY = "expanded_frontal_mesh"
PROFILE_MESH_FAMILY = "expanded_profile_mesh"
PROFILE_REVIEWED_FAMILY = "expanded_profile_reviewed"

_HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")
_TOKEN = re.compile(r"[A-Za-z0-9_-]{1,96}")

_REFERENCE_POLICY = (
    "Unranked report-only geometry. No reference distribution, percentile, "
    "ranking, score eligibility, clinical interpretation, physical-scale unit "
    "or human-verification claim is attached."
)
_SCORE_EXCLUSION = (
    "Expanded checkpoint 5 paths are unranked report-only 2D constructions and "
    "are not admitted to any score, norm, percentile or ranking instrument."
)


class _FrameUnavailable(Exception):
    """A candidate frame failed routing or geometry validation."""


def _module_bytes() -> bytes:
    return Path(__file__).read_bytes()


def _validate_model_hash(model_sha256: str | None) -> str | None:
    if model_sha256 is None:
        return None
    if not isinstance(model_sha256, str) or not _HEX64.fullmatch(model_sha256):
        raise ValueError("model_sha256 must be a 64-character SHA-256 hex digest.")
    return model_sha256.lower()


def _hash_payload(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def expanded_contract(model_sha256: str | None = None) -> str:
    """Hash this real module file together with the optional detector identity."""

    model_hash = _validate_model_hash(model_sha256)
    suffix = b"\nmodel_sha256=" + (model_hash or "none").encode()
    return _hash_payload(_module_bytes() + suffix)


_FRONTAL_SPECS: tuple[dict[str, Any], ...] = (
    {
        "id": "eye_width_height_a",
        "label": "Eye width/height a Mesh proxy",
        "unit": "ratio",
        "definition": "Image-side a mesh proxy: horizontal p33-p133 span divided by vertical p159-p145 span; a and b are model/image sides, not anatomical left/right.",
        "formula": "H(p33,p133) / V(p159,p145)",
        "inputs": (33, 133, 159, 145),
    },
    {
        "id": "eye_width_height_b",
        "label": "Eye width/height b Mesh proxy",
        "unit": "ratio",
        "definition": "Image-side b mesh proxy: horizontal p263-p362 span divided by vertical p386-p374 span; not a reciprocal of pooled legacy eye almondness.",
        "formula": "H(p263,p362) / V(p386,p374)",
        "inputs": (263, 362, 386, 374),
    },
    {
        "id": "canthal_tilt_a",
        "label": "Canthal tilt a Mesh proxy",
        "unit": "deg",
        "definition": "Image-side a mesh proxy angle from outer p33 to inner p133 in original image coordinates.",
        "formula": "degrees(atan2(y(p133)-y(p33), abs(x(p133)-x(p33))))",
        "inputs": (133, 33),
    },
    {
        "id": "canthal_tilt_b",
        "label": "Canthal tilt b Mesh proxy",
        "unit": "deg",
        "definition": "Image-side b mesh proxy angle from outer p263 to inner p362 in original image coordinates.",
        "formula": "degrees(atan2(y(p362)-y(p263), abs(x(p362)-x(p263))))",
        "inputs": (362, 263),
    },
    {
        "id": "brow_tilt_a",
        "label": "Brow tilt a Mesh proxy",
        "unit": "deg",
        "definition": "Image-side a mesh proxy tilt from brow arch p105 to inner brow p107.",
        "formula": "degrees(atan2(y(p107)-y(p105), abs(x(p107)-x(p105))))",
        "inputs": (107, 105),
    },
    {
        "id": "brow_tilt_b",
        "label": "Brow tilt b Mesh proxy",
        "unit": "deg",
        "definition": "Image-side b mesh proxy tilt from brow arch p334 to inner brow p336.",
        "formula": "degrees(atan2(y(p336)-y(p334), abs(x(p336)-x(p334))))",
        "inputs": (336, 334),
    },
    {
        "id": "brow_span_face_a",
        "label": "Brow span/face a Mesh proxy",
        "unit": "ratio",
        "definition": "Image-side a mesh proxy: p70-p107 horizontal brow span over p234-p454 mesh face width.",
        "formula": "H(p70,p107) / H(p234,p454)",
        "inputs": (70, 107, 234, 454),
    },
    {
        "id": "brow_span_face_b",
        "label": "Brow span/face b Mesh proxy",
        "unit": "ratio",
        "definition": "Image-side b mesh proxy: p300-p336 horizontal brow span over p234-p454 mesh face width.",
        "formula": "H(p300,p336) / H(p234,p454)",
        "inputs": (300, 336, 234, 454),
    },
    {
        "id": "brow_gap_eye_height_a",
        "label": "Brow gap/eye height a Mesh proxy",
        "unit": "ratio",
        "definition": "Image-side a mesh proxy: mean p159/p145 y minus p107 y, divided by p159-p145 vertical eye height.",
        "formula": "(mean(y(p159),y(p145))-y(p107)) / V(p159,p145)",
        "inputs": (159, 145, 107),
    },
    {
        "id": "brow_gap_eye_height_b",
        "label": "Brow gap/eye height b Mesh proxy",
        "unit": "ratio",
        "definition": "Image-side b mesh proxy: mean p386/p374 y minus p336 y, divided by p386-p374 vertical eye height.",
        "formula": "(mean(y(p386),y(p374))-y(p336)) / V(p386,p374)",
        "inputs": (386, 374, 336),
    },
    {
        "id": "lower_upper_lip_ratio",
        "label": "Lower/upper lip Mesh proxy",
        "unit": "ratio",
        "definition": "Mesh proxy using separate inner seams p13/p14: lower p14-p17 vertical span over upper p0-p13 vertical span.",
        "formula": "V(p14,p17) / V(p0,p13)",
        "inputs": (14, 17, 0, 13),
    },
    {
        "id": "lip_thickness_lower_face",
        "label": "Lip thickness/lower face Mesh proxy",
        "unit": "ratio",
        "definition": "Mesh proxy sum of upper and lower seam spans divided by p2-p152 lower-face vertical span.",
        "formula": "(V(p0,p13)+V(p14,p17)) / V(p2,p152)",
        "inputs": (0, 13, 14, 17, 2, 152),
    },
    {
        "id": "chin_philtrum_ratio",
        "label": "Chin/philtrum Mesh proxy",
        "unit": "ratio",
        "definition": "Mesh proxy p17-p152 vertical span divided by p2-p0 vertical span.",
        "formula": "V(p17,p152) / V(p2,p0)",
        "inputs": (17, 152, 2, 0),
    },
    {
        "id": "mouth_nose_ratio",
        "label": "Mouth/nose Mesh proxy",
        "unit": "ratio",
        "definition": "Mesh proxy mouth p61-p291 horizontal span over nose p129-p358 horizontal span; reciprocal of a legacy construction, with separate reference status.",
        "formula": "H(p61,p291) / H(p129,p358)",
        "inputs": (61, 291, 129, 358),
    },
    {
        "id": "cupid_depth_mouth_ratio",
        "label": "Cupid depth/mouth Mesh proxy",
        "unit": "ratio",
        "definition": "Signed-down mesh contour proxy using p0 relative to p37/p267, divided by p61-p291 mouth width; fixed mesh contour, not observed tissue depth.",
        "formula": "(y(p0)-mean(y(p37),y(p267))) / H(p61,p291)",
        "inputs": (0, 37, 267, 61, 291),
    },
    {
        "id": "mouth_corner_seam_ratio",
        "label": "Mouth corner/seam Mesh proxy",
        "unit": "ratio",
        "definition": "Signed-down mesh proxy: mean mouth-corner y minus mean seam y, divided by p61-p291 mouth width.",
        "formula": "(mean(y(p61),y(p291))-mean(y(p13),y(p14))) / H(p61,p291)",
        "inputs": (61, 291, 13, 14),
    },
    {
        "id": "jaw_v_angle",
        "label": "Jaw V angle Mesh proxy",
        "unit": "deg",
        "definition": "Mesh proxy angle p172-p152-p397; this is not the competitor jaw-frontal construction.",
        "formula": "angle(p172,p152,p397)",
        "inputs": (172, 152, 397),
    },
    {
        "id": "jaw_side_angle_a",
        "label": "Jaw side angle a Mesh proxy",
        "unit": "deg",
        "definition": "Image-side a mesh proxy angle p234-p172-p152; not an anatomical gonial angle.",
        "formula": "angle(p234,p172,p152)",
        "inputs": (234, 172, 152),
    },
    {
        "id": "jaw_side_angle_b",
        "label": "Jaw side angle b Mesh proxy",
        "unit": "deg",
        "definition": "Image-side b mesh proxy angle p454-p397-p152; not an anatomical gonial angle.",
        "formula": "angle(p454,p397,p152)",
        "inputs": (454, 397, 152),
    },
)

_PROFILE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "id": "nasofacial",
        "unit": "deg",
        "definition": "2D constructed profile angle Prn-N-Pog in image coordinates, not a clinical or physical-scale measurement.",
        "formula": "angle(Prn,N,Pog)",
        "inputs": ("tip", "nasion", "pogonion"),
    },
    {
        "id": "nasomental",
        "unit": "deg",
        "definition": "2D constructed profile angle N-Prn-Pog in image coordinates, with no clinical scoring or posture-plane assumption.",
        "formula": "angle(N,Prn,Pog)",
        "inputs": ("nasion", "tip", "pogonion"),
    },
    {
        "id": "upper_sline",
        "unit": "% N-Me",
        "definition": "Signed upper-lip perpendicular offset to midpoint(Prn,Sn)-Pog, normalized by D(N,Me). Positive is away from the N side of the line.",
        "formula": "100 * signed_perp(UL,midpoint(Prn,Sn),Pog,N) / D(N,Me)",
        "inputs": ("upper_lip", "tip", "subnasale", "pogonion", "nasion", "menton"),
    },
    {
        "id": "lower_sline",
        "unit": "% N-Me",
        "definition": "Signed lower-lip perpendicular offset to midpoint(Prn,Sn)-Pog, normalized by D(N,Me). Positive is away from the N side of the line.",
        "formula": "100 * signed_perp(LL,midpoint(Prn,Sn),Pog,N) / D(N,Me)",
        "inputs": ("lower_lip", "tip", "subnasale", "pogonion", "nasion", "menton"),
    },
    {
        "id": "upper_burstone",
        "unit": "% N-Me",
        "definition": "Signed upper-lip perpendicular offset to Sn-Pog, normalized by D(N,Me). Positive is away from the N side of the line.",
        "formula": "100 * signed_perp(UL,Sn,Pog,N) / D(N,Me)",
        "inputs": ("upper_lip", "subnasale", "pogonion", "nasion", "menton"),
    },
    {
        "id": "lower_burstone",
        "unit": "% N-Me",
        "definition": "Signed lower-lip perpendicular offset to Sn-Pog, normalized by D(N,Me). Positive is away from the N side of the line.",
        "formula": "100 * signed_perp(LL,Sn,Pog,N) / D(N,Me)",
        "inputs": ("lower_lip", "subnasale", "pogonion", "nasion", "menton"),
    },
    {
        "id": "lower_holdaway",
        "unit": "% N-Me",
        "definition": "Signed lower-lip perpendicular offset to UL-Pog, normalized by D(N,Me). Positive is away from the N side of the line.",
        "formula": "100 * signed_perp(LL,UL,Pog,N) / D(N,Me)",
        "inputs": ("lower_lip", "upper_lip", "pogonion", "nasion", "menton"),
    },
)

_MESH_TO_PROFILE = {
    "p168": "nasion",
    "p4": "tip",
    "p2": "subnasale",
    "p0": "upper_lip",
    "p17": "lower_lip",
    "p200": "pogonion",
    "p152": "menton",
}
_PROFILE_TO_MESH = {value: key for key, value in _MESH_TO_PROFILE.items()}


def expanded_definitions() -> list[dict]:
    """Return fresh metadata rows for all 33 expanded paths."""

    rows: list[dict[str, Any]] = []
    for spec in _FRONTAL_SPECS:
        rows.append(
            {
                "key": f"{FRONTAL_FAMILY}:{spec['id']}",
                "family": FRONTAL_FAMILY,
                "id": spec["id"],
                "label": spec["label"],
                "region": "frontal",
                "unit": spec["unit"],
                "definition": spec["definition"],
                "formula": spec["formula"],
                "method": EXPANDED_FRONTAL_MESH_METHOD,
                "inputs": list(spec["inputs"]),
                "reference_policy": _REFERENCE_POLICY,
            }
        )
    for family, method, label_suffix in (
        (PROFILE_MESH_FAMILY, EXPANDED_PROFILE_MESH_METHOD, "Mesh proxy"),
        (PROFILE_REVIEWED_FAMILY, EXPANDED_PROFILE_REVIEWED_METHOD, "Reviewed 2D"),
    ):
        for spec in _PROFILE_SPECS:
            rows.append(
                {
                    "key": f"{family}:{spec['id']}",
                    "family": family,
                    "id": spec["id"],
                    "label": f"{spec['id'].replace('_', ' ').title()} {label_suffix}",
                    "region": "profile",
                    "unit": spec["unit"],
                    "definition": f"{label_suffix} construction. {spec['definition']}",
                    "formula": spec["formula"],
                    "method": method,
                    "inputs": list(spec["inputs"]),
                    "reference_policy": _REFERENCE_POLICY,
                }
            )
    return deepcopy(rows)


def _attr(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    try:
        result = float(value)
    except (OverflowError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _length(value: Any) -> int | None:
    try:
        return len(value)
    except TypeError:
        return None


def _raw_pair(raw: Any) -> tuple[float, float]:
    if isinstance(raw, (str, bytes)) or _length(raw) != 2:
        raise _FrameUnavailable("Malformed landmark vector.")
    try:
        x, y = raw[0], raw[1]
    except (KeyError, IndexError, TypeError) as exc:
        raise _FrameUnavailable("Malformed landmark vector.") from exc
    if (
        isinstance(x, bool)
        or isinstance(y, bool)
        or not isinstance(x, Real)
        or not isinstance(y, Real)
    ):
        raise _FrameUnavailable("Landmark coordinates must be numeric.")
    try:
        return float(x), float(y)
    except (OverflowError, ValueError) as exc:
        raise _FrameUnavailable("Landmark coordinates must fit finite floats.") from exc


def _inside(point: tuple[float, float] | None, width: int, height: int) -> bool:
    return (
        point is not None
        and math.isfinite(point[0])
        and math.isfinite(point[1])
        and 0 <= point[0] <= width
        and 0 <= point[1] <= height
    )


def _photo_id(photo: Any) -> str | None:
    entry = _attr(photo, "entry")
    candidate = (
        _attr(entry, "photo_id") or _attr(photo, "photo_id") or _attr(photo, "id")
    )
    return (
        candidate
        if isinstance(candidate, str) and _TOKEN.fullmatch(candidate)
        else None
    )


def _photo_meta(photos: Any) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for ordinal, photo in enumerate(photos or (), start=1):
        photo_id = _photo_id(photo)
        entry = _attr(photo, "entry")
        width = _attr(entry, "width", _attr(photo, "width"))
        height = _attr(entry, "height", _attr(photo, "height"))
        revision = _attr(photo, "image_revision", _attr(photo, "revision", 1))
        if (
            photo_id
            and type(width) is int
            and type(height) is int
            and type(revision) is int
            and 1 <= width <= 100000
            and 1 <= height <= 100000
            and revision >= 1
        ):
            result[photo_id] = {
                "photo_number": ordinal,
                "width": width,
                "height": height,
                "revision": revision,
            }
    return result


def _fact_map(context: Any) -> dict[str, Any]:
    raw = _attr(context, "photo_facts", ()) or ()
    if isinstance(raw, dict):
        iterable = raw.values()
    else:
        iterable = raw
    result: dict[str, Any] = {}
    for fact in iterable:
        photo_id = _attr(fact, "photo_id") or _attr(_attr(fact, "photo"), "id")
        if isinstance(photo_id, str):
            result[photo_id] = fact
    return result


def _kept_photo_id(record: Any) -> str | None:
    photo = _attr(record, "photo")
    if isinstance(photo, str):
        return photo
    return _attr(photo, "id") or _attr(photo, "photo_id") or _attr(record, "photo_id")


def _pose_ok(fact: Any) -> bool:
    yaw = _finite(_attr(fact, "yaw"))
    pitch = _finite(_attr(fact, "pitch"))
    roll = _finite(_attr(fact, "roll"))
    return (
        yaw is not None
        and pitch is not None
        and roll is not None
        and abs(yaw) <= 20
        and abs(pitch) <= 15
        and abs(roll) <= 15
    )


def _frontal_points(
    landmarks: Any, *, width: int, height: int
) -> list[tuple[float, float] | None]:
    if (
        landmarks is None
        or isinstance(landmarks, (str, bytes))
        or _length(landmarks) not in (468, 478)
    ):
        raise _FrameUnavailable("Expected a 468/478 landmark pixel array.")
    parsed = []
    for raw in landmarks:
        try:
            parsed.append(_raw_pair(raw))
        except _FrameUnavailable:
            parsed.append(None)
    p10, p152 = parsed[10], parsed[152]
    if not _inside(p10, width, height) or not _inside(p152, width, height):
        raise _FrameUnavailable("Missing bounded p10-p152 reference height.")
    if abs(p152[1] - p10[1]) < 80:
        raise _FrameUnavailable("Detected p10-p152 reference height is below 80px.")
    return parsed


def _require_points(
    parsed: list[tuple[float, float] | None],
    inputs: tuple[int, ...],
    *,
    width: int,
    height: int,
) -> dict[int, tuple[float, float]] | None:
    points: dict[int, tuple[float, float]] = {}
    for index in inputs:
        point = parsed[index]
        if not _inside(point, width, height):
            return None
        points[index] = point
    return points


def _h(points: dict[int, tuple[float, float]], a: int, b: int) -> float:
    return abs(points[a][0] - points[b][0])


def _v(points: dict[int, tuple[float, float]], a: int, b: int) -> float:
    return abs(points[a][1] - points[b][1])


def _angle(
    a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]
) -> float | None:
    ux, uy = a[0] - b[0], a[1] - b[1]
    vx, vy = c[0] - b[0], c[1] - b[1]
    den = math.hypot(ux, uy) * math.hypot(vx, vy)
    if den <= 1e-8:
        return None
    cosine = max(-1.0, min(1.0, (ux * vx + uy * vy) / den))
    return math.degrees(math.acos(cosine))


def _ratio(numerator: float, denominator: float) -> float | None:
    if abs(denominator) <= 1e-8:
        return None
    value = numerator / denominator
    return value if math.isfinite(value) else None


def _frontal_value(metric_id: str, p: dict[int, tuple[float, float]]) -> float | None:
    tilt_spans = {
        "canthal_tilt_a": (133, 33),
        "canthal_tilt_b": (362, 263),
        "brow_tilt_a": (107, 105),
        "brow_tilt_b": (336, 334),
    }
    if metric_id in tilt_spans and _h(p, *tilt_spans[metric_id]) <= 1e-8:
        return None
    if metric_id == "eye_width_height_a":
        return _ratio(_h(p, 33, 133), _v(p, 159, 145))
    if metric_id == "eye_width_height_b":
        return _ratio(_h(p, 263, 362), _v(p, 386, 374))
    if metric_id == "canthal_tilt_a":
        return math.degrees(math.atan2(p[133][1] - p[33][1], _h(p, 133, 33)))
    if metric_id == "canthal_tilt_b":
        return math.degrees(math.atan2(p[362][1] - p[263][1], _h(p, 362, 263)))
    if metric_id == "brow_tilt_a":
        return math.degrees(math.atan2(p[107][1] - p[105][1], _h(p, 107, 105)))
    if metric_id == "brow_tilt_b":
        return math.degrees(math.atan2(p[336][1] - p[334][1], _h(p, 336, 334)))
    if metric_id == "brow_span_face_a":
        return _ratio(_h(p, 70, 107), _h(p, 234, 454))
    if metric_id == "brow_span_face_b":
        return _ratio(_h(p, 300, 336), _h(p, 234, 454))
    if metric_id == "brow_gap_eye_height_a":
        return _ratio(((p[159][1] + p[145][1]) / 2) - p[107][1], _v(p, 159, 145))
    if metric_id == "brow_gap_eye_height_b":
        return _ratio(((p[386][1] + p[374][1]) / 2) - p[336][1], _v(p, 386, 374))
    if metric_id == "lower_upper_lip_ratio":
        return _ratio(_v(p, 14, 17), _v(p, 0, 13))
    if metric_id == "lip_thickness_lower_face":
        return _ratio(_v(p, 0, 13) + _v(p, 14, 17), _v(p, 2, 152))
    if metric_id == "chin_philtrum_ratio":
        return _ratio(_v(p, 17, 152), _v(p, 2, 0))
    if metric_id == "mouth_nose_ratio":
        return _ratio(_h(p, 61, 291), _h(p, 129, 358))
    if metric_id == "cupid_depth_mouth_ratio":
        return _ratio(p[0][1] - ((p[37][1] + p[267][1]) / 2), _h(p, 61, 291))
    if metric_id == "mouth_corner_seam_ratio":
        return _ratio(
            ((p[61][1] + p[291][1]) / 2) - ((p[13][1] + p[14][1]) / 2), _h(p, 61, 291)
        )
    if metric_id == "jaw_v_angle":
        return _angle(p[172], p[152], p[397])
    if metric_id == "jaw_side_angle_a":
        return _angle(p[234], p[172], p[152])
    if metric_id == "jaw_side_angle_b":
        return _angle(p[454], p[397], p[152])
    raise KeyError(metric_id)


def _source_points_from_indices(
    points: dict[int, tuple[float, float]],
    inputs: tuple[int, ...],
    *,
    width: int,
    height: int,
) -> tuple[Point, ...]:
    return tuple(
        Point(name=f"p{index}", x=points[index][0] / width, y=points[index][1] / height)
        for index in inputs
    )


def _frontal_source(
    *,
    photo_id: str,
    meta: dict[str, int],
    points: dict[int, tuple[float, float]],
    inputs: tuple[int, ...],
) -> SourceEvidence:
    return SourceEvidence(
        photo_id=photo_id,
        photo_number=meta["photo_number"],
        view="unknown",
        width=meta["width"],
        height=meta["height"],
        revision=meta["revision"],
        review_revision=0,
        points=_source_points_from_indices(
            points, inputs, width=meta["width"], height=meta["height"]
        ),
        visibility="unassessed",
        capture_eligible=False,
    )


def _frontal_definitions() -> dict[str, dict[str, Any]]:
    return {
        row["id"]: row
        for row in expanded_definitions()
        if row["family"] == FRONTAL_FAMILY
    }


def expanded_frontal_observations(
    context: Any, photos: Any, *, model_sha256: str | None = None
) -> tuple[Observation, ...]:
    """Build one expanded frontal mesh card per definition from eligible frames."""

    contract = expanded_contract(model_sha256)
    facts = _fact_map(context)
    meta_by_photo = _photo_meta(photos)
    candidates: list[tuple[str, dict[str, int], list[tuple[float, float]]]] = []
    for record in _attr(context, "kept", ()) or ():
        photo_id = _kept_photo_id(record)
        if not isinstance(photo_id, str) or photo_id not in meta_by_photo:
            continue
        fact = _attr(record, "fact", None) or facts.get(photo_id)
        if not _pose_ok(fact):
            continue
        meta = meta_by_photo[photo_id]
        try:
            parsed = _frontal_points(
                _attr(record, "landmarks"), width=meta["width"], height=meta["height"]
            )
        except _FrameUnavailable:
            continue
        candidates.append((photo_id, meta, parsed))

    definitions = _frontal_definitions()
    result: list[Observation] = []
    for spec in _FRONTAL_SPECS:
        definition = definitions[spec["id"]]
        values: list[float] = []
        sources: list[SourceEvidence] = []
        for photo_id, meta, parsed in candidates:
            points = _require_points(
                parsed,
                spec["inputs"],
                width=meta["width"],
                height=meta["height"],
            )
            if points is None:
                continue
            value = _frontal_value(spec["id"], points)
            if value is None or not math.isfinite(value):
                continue
            values.append(value)
            sources.append(
                _frontal_source(
                    photo_id=photo_id, meta=meta, points=points, inputs=spec["inputs"]
                )
            )
        if values:
            result.append(
                Observation(
                    definition_key=definition["key"],
                    definition_version=contract,
                    label=definition["label"],
                    definition=definition["definition"],
                    formula=definition["formula"],
                    method=EXPANDED_FRONTAL_MESH_METHOD,
                    unit=definition["unit"],
                    status="proxy",
                    value=round(float(median(values)), 6),
                    revision=1,
                    sources=tuple(sources),
                    display_decimals=6,
                    proxy_for=f"frontal_mesh_proxy:{spec['id']}",
                    score_eligible=False,
                    score_exclusion_reason=_SCORE_EXCLUSION,
                )
            )
        else:
            result.append(
                Observation(
                    definition_key=definition["key"],
                    definition_version=contract,
                    label=definition["label"],
                    definition=definition["definition"],
                    formula=definition["formula"],
                    method=EXPANDED_FRONTAL_MESH_METHOD,
                    unit=definition["unit"],
                    status="unavailable",
                    value=None,
                    revision=1,
                    sources=(),
                    display_decimals=6,
                    withheld_reason=(
                        "No eligible frontal mesh frame supplied all required finite, "
                        "bounded points under the declared pose and p10-p152 routing "
                        "heuristics; no substitutions or default zeros were used."
                    ),
                    score_eligible=False,
                    score_exclusion_reason=_SCORE_EXCLUSION,
                )
            )
    return tuple(result)


def exclude_reviewed_frontal_sources(
    records: tuple[Observation, ...], reviewed_ids: set[str]
) -> tuple[Observation, ...]:
    """Reaggregate saved mesh geometry after reviews, without detector/axis reruns."""
    result = []
    specs = {f"{FRONTAL_FAMILY}:{s['id']}": s for s in _FRONTAL_SPECS}
    for record in records:
        if record.definition_key not in specs:
            raise ValueError("Unknown expanded frontal construction.")
        if not any(s.photo_id in reviewed_ids for s in record.sources):
            result.append(record)
            continue
        sources, values = [], []
        for source in record.sources:
            if source.photo_id in reviewed_ids:
                continue
            points = {
                int(p.name[1:]): (p.x * source.width, p.y * source.height)
                for p in source.points
            }
            value = _frontal_value(specs[record.definition_key]["id"], points)
            if value is not None and math.isfinite(value):
                sources.append(source)
                values.append(value)
        available = bool(values)
        result.append(
            Observation.model_validate(
                {
                    **record.model_dump(),
                    "sources": tuple(sources),
                    "value": round(median(values), 6) if available else None,
                    "status": "proxy" if available else "unavailable",
                    "proxy_for": record.proxy_for if available else None,
                    "withheld_reason": (
                        None
                        if available
                        else "No unreviewed compatible source remains for these separate mesh anchors. Manual frontal edits do not review this anchor set."
                    ),
                }
            )
        )
    return tuple(result)


def _profile_definitions(family: str) -> dict[str, dict[str, Any]]:
    return {row["id"]: row for row in expanded_definitions() if row["family"] == family}


def _family(record: Observation) -> str | None:
    if record.definition_key.startswith("mesh_profile_proxy:"):
        return PROFILE_MESH_FAMILY
    if record.definition_key.startswith("reviewed_profile:"):
        return PROFILE_REVIEWED_FAMILY
    return None


def _group_key(source: SourceEvidence, family: str) -> tuple[Any, ...]:
    return (
        family,
        source.photo_id,
        source.photo_number,
        source.width,
        source.height,
        source.view,
        source.revision,
        source.review_revision,
    )


def _ancestry_text(definitions: list[str]) -> str:
    for text in definitions:
        marker = "Initial positions came from"
        if marker in text:
            return " " + text[text.index(marker) :].strip()
    return ""


def _profile_version(family: str, ancestry: list[tuple[str, str, str]]) -> str:
    payload = repr(
        {
            "expanded_contract": expanded_contract(None),
            "family": family,
            "base": sorted(ancestry),
        }
    ).encode()
    return _hash_payload(payload)


def _collect_profile_groups(
    records: tuple[Observation, ...],
) -> dict[tuple[Any, ...], dict[str, Any]]:
    groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    for raw in records:
        record = Observation.model_validate(raw)
        if not record.geometry_retained:
            raise ValueError(
                "Archived geometry_retained=False observations cannot supply live expanded profile geometry."
            )
        family = _family(record)
        if family is None:
            continue
        expected_method = (
            "mesh-profile-proxy-v1"
            if family == PROFILE_MESH_FAMILY
            else "manual-profile-v2"
        )
        if record.method != expected_method:
            raise ValueError(
                "Unsupported base profile instrument for expanded constructions."
            )
        for source in record.sources:
            if (
                family == PROFILE_REVIEWED_FAMILY
                and source.points
                and source.review_revision <= 0
            ):
                raise ValueError("Reviewed profile sources need review_revision > 0.")
            key = (
                *_group_key(source, family),
                record.method,
                record.definition_version,
            )
            group = groups.setdefault(
                key,
                {
                    "family": family,
                    "source": source,
                    "points": {},
                    "definitions": [],
                    "ancestry": [],
                },
            )
            group["definitions"].append(record.definition)
            group["ancestry"].append(
                (record.definition_key, record.definition_version, record.method)
            )
            for point in source.points:
                name = point.name
                if family == PROFILE_MESH_FAMILY:
                    if name not in _MESH_TO_PROFILE:
                        continue
                    name = _MESH_TO_PROFILE[name]
                elif name not in _PROFILE_TO_MESH:
                    continue
                pixel = (point.x * source.width, point.y * source.height)
                previous = group["points"].get(name)
                if previous is not None and (
                    abs(previous[0] - pixel[0]) > 1e-6
                    or abs(previous[1] - pixel[1]) > 1e-6
                ):
                    raise ValueError(
                        "Profile source points conflict within the same photo/revision group."
                    )
                group["points"][name] = pixel
    return groups


def _perpendicular(
    q: tuple[float, float], a: tuple[float, float], b: tuple[float, float]
) -> float | None:
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy)
    if length <= 1e-8:
        return None
    return (dy * (q[0] - a[0]) - dx * (q[1] - a[1])) / length


def _signed_offset(
    q: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
    n: tuple[float, float],
    me: tuple[float, float],
) -> float | None:
    height = math.dist(n, me)
    side_n = _perpendicular(n, a, b)
    side_q = _perpendicular(q, a, b)
    if height <= 1e-8 or side_n is None or side_q is None or abs(side_n) <= 1e-8:
        return None
    return side_q * (-1 if side_n > 0 else 1) * 100 / height


def _profile_required(metric_id: str) -> tuple[str, ...]:
    return next(spec["inputs"] for spec in _PROFILE_SPECS if spec["id"] == metric_id)


def _profile_value(metric_id: str, p: dict[str, tuple[float, float]]) -> float | None:
    if any(name not in p for name in _profile_required(metric_id)):
        return None
    n, prn, sn, ul, ll, pog, me = (
        p.get(name)
        for name in (
            "nasion",
            "tip",
            "subnasale",
            "upper_lip",
            "lower_lip",
            "pogonion",
            "menton",
        )
    )
    if metric_id == "nasofacial":
        return _angle(prn, n, pog)
    if metric_id == "nasomental":
        return _angle(n, prn, pog)
    if metric_id == "upper_sline":
        mid = ((prn[0] + sn[0]) / 2, (prn[1] + sn[1]) / 2)
        return _signed_offset(ul, mid, pog, n, me)
    if metric_id == "lower_sline":
        mid = ((prn[0] + sn[0]) / 2, (prn[1] + sn[1]) / 2)
        return _signed_offset(ll, mid, pog, n, me)
    if metric_id == "upper_burstone":
        return _signed_offset(ul, sn, pog, n, me)
    if metric_id == "lower_burstone":
        return _signed_offset(ll, sn, pog, n, me)
    if metric_id == "lower_holdaway":
        return _signed_offset(ll, ul, pog, n, me)
    raise KeyError(metric_id)


def _profile_source_points(
    family: str,
    source: SourceEvidence,
    points: dict[str, tuple[float, float]],
    required: tuple[str, ...],
) -> tuple[Point, ...]:
    result = []
    for name in required:
        if name not in points:
            continue
        point_name = _PROFILE_TO_MESH[name] if family == PROFILE_MESH_FAMILY else name
        x, y = points[name]
        result.append(Point(name=point_name, x=x / source.width, y=y / source.height))
    return tuple(result)


def _profile_source(
    family: str,
    source: SourceEvidence,
    points: dict[str, tuple[float, float]],
    required: tuple[str, ...],
) -> SourceEvidence | None:
    kept_points = _profile_source_points(family, source, points, required)
    if not kept_points:
        return None
    return SourceEvidence(
        photo_id=source.photo_id,
        photo_number=source.photo_number,
        view=source.view,
        width=source.width,
        height=source.height,
        revision=source.revision,
        review_revision=source.review_revision,
        points=kept_points,
        visibility=source.visibility,
        forehead_visibility=source.forehead_visibility,
        automatic_forehead_status=source.automatic_forehead_status,
        capture_eligible=False,
    )


def _profile_observation(
    *,
    family: str,
    definition: dict[str, Any],
    version: str,
    value: float | None,
    source: SourceEvidence | None,
    ancestry_note: str = "",
) -> Observation:
    method = (
        EXPANDED_PROFILE_MESH_METHOD
        if family == PROFILE_MESH_FAMILY
        else EXPANDED_PROFILE_REVIEWED_METHOD
    )
    available = value is not None and source is not None
    return Observation(
        definition_key=definition["key"],
        definition_version=version,
        label=definition["label"],
        definition=definition["definition"] + ancestry_note,
        formula=definition["formula"],
        method=method,
        unit=definition["unit"],
        status=(
            "proxy"
            if available and family == PROFILE_MESH_FAMILY
            else "user_reviewed" if available else "unavailable"
        ),
        value=value,
        revision=1,
        sources=(source,) if source is not None else (),
        display_decimals=6,
        proxy_for=(
            f"constructed_profile:{definition['id']}"
            if available and family == PROFILE_MESH_FAMILY
            else None
        ),
        withheld_reason=(
            None
            if available
            else "No same-photo, same-revision source group supplied all finite required profile points; no points were borrowed across photos, revisions or families."
        ),
        score_eligible=False,
        score_exclusion_reason=_SCORE_EXCLUSION,
    )


def expanded_profile_observations(
    records: tuple[Observation, ...],
) -> tuple[Observation, ...]:
    """Construct expanded profile observations from current base profile rows only."""

    groups = _collect_profile_groups(records)
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group in groups.values():
        by_family[group["family"]].append(group)

    result: list[Observation] = []
    for family in (PROFILE_MESH_FAMILY, PROFILE_REVIEWED_FAMILY):
        definitions = _profile_definitions(family)
        family_groups = by_family.get(family) or [
            {
                "family": family,
                "source": None,
                "points": {},
                "definitions": [],
                "ancestry": [],
            }
        ]
        for group in family_groups:
            ancestry = list(group["ancestry"])
            version = _profile_version(family, ancestry)
            ancestry_note = (
                _ancestry_text(group["definitions"])
                if family == PROFILE_REVIEWED_FAMILY
                else ""
            )
            for spec in _PROFILE_SPECS:
                value = _profile_value(spec["id"], group["points"])
                rounded = (
                    round(value, 6)
                    if value is not None and math.isfinite(value)
                    else None
                )
                source = (
                    _profile_source(
                        family,
                        group["source"],
                        group["points"],
                        _profile_required(spec["id"]),
                    )
                    if group["source"] is not None
                    else None
                )
                result.append(
                    _profile_observation(
                        family=family,
                        definition=definitions[spec["id"]],
                        version=version,
                        value=rounded,
                        source=source,
                        ancestry_note=ancestry_note,
                    )
                )
    return tuple(result)
