"""Partial, revision-bound 2D frontal review. Automatic geometry stays immutable."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import html
import json
import math
from typing import Any, Annotated

from pydantic import ConfigDict, Field, RootModel

from .measurement_registry import METRICS, REGISTRY_VERSION, measurement_contract

# A/B refer to the two labelled sides in the original image; display mirroring
# never changes stored coordinates or silently enforces bilateral symmetry.
ANCHORS = {
    "mesh_top": ("Mesh forehead top (not hairline)", 10, "forehead"),
    "hairline": (
        "Visible central hairline — leave unavailable if hidden",
        None,
        "forehead",
    ),
    "forehead_a": ("Forehead outline A", 54, "forehead"),
    "forehead_b": ("Forehead outline B", 284, "forehead"),
    "brow_head_a": ("Medial brow A", 55, "eyes"),
    "brow_head_b": ("Medial brow B", 285, "eyes"),
    "brow_peak_a": ("Brow peak A", None, "eyes"),
    "brow_peak_b": ("Brow peak B", None, "eyes"),
    "eye_outer_a": ("Eye A outer corner", 33, "eyes"),
    "eye_inner_a": ("Eye A inner corner", 133, "eyes"),
    "eye_inner_b": ("Eye B inner corner", 362, "eyes"),
    "eye_outer_b": ("Eye B outer corner", 263, "eyes"),
    "eye_top_a": ("Eye A opening top", 159, "eyes"),
    "eye_bottom_a": ("Eye A opening bottom", 145, "eyes"),
    "eye_top_b": ("Eye B opening top", 386, "eyes"),
    "eye_bottom_b": ("Eye B opening bottom", 374, "eyes"),
    "eye_centre_a": ("Eye opening centre A", None, "eyes"),
    "eye_centre_b": ("Eye opening centre B", None, "eyes"),
    "cheek_a": ("Cheek outline A", 234, "jaw"),
    "cheek_b": ("Cheek outline B", 454, "jaw"),
    "jaw_a": ("Lower jaw outline A", 172, "jaw"),
    "jaw_b": ("Lower jaw outline B", 397, "jaw"),
    "chin_a": ("Chin width A", 32, "jaw"),
    "chin_b": ("Chin width B", 262, "jaw"),
    "chin": ("Lowest chin", 152, "jaw"),
    "nose_a": ("Nose wing A", 129, "nose"),
    "nose_b": ("Nose wing B", 358, "nose"),
    "nose_base": ("Nose base", 2, "nose"),
    "mouth_a": ("Mouth corner A", 61, "lips"),
    "mouth_b": ("Mouth corner B", 291, "lips"),
    "upper_lip": ("Upper lip top", 0, "lips"),
    "lower_lip": ("Lower lip bottom", 17, "lips"),
}
REVIEW_NOTE = "Reviewed 2D photo measurements. No matching manual-reference percentiles or calibrated uncertainty. Head pose and placement affect readings. Automatic pillar cards and composite use their original instrument; these readings do not overwrite inferred depth."
FOREHEAD_VISIBILITY = {
    "unassessed": "Use automatic visibility assessment",
    "visible": "Forehead and central hairline are clearly visible",
    "hairline_hidden": "Hair hides the central hairline; forehead outline is visible",
    "forehead_hidden": "Hair hides the forehead outline and hairline",
    "uncertain": "I cannot tell where the hairline is",
}


def forehead_points(
    raw: Any, visibility: str, automatic_status: str | None = None
) -> dict[str, list[float] | None]:
    """Hair coverage takes precedence over retained model/user coordinates."""
    if not isinstance(visibility, str) or visibility not in FOREHEAD_VISIBILITY:
        raise ValueError("Choose a supported forehead visibility option.")
    if automatic_status is not None:
        from .automatic_forehead import EXPLANATIONS

        if not isinstance(automatic_status, str) or automatic_status not in EXPLANATIONS:
            raise ValueError("Unsupported automatic forehead finding.")
    points = validate_points(raw)
    if visibility != "visible":
        points["hairline"] = None
    if visibility == "forehead_hidden" or (
        visibility == "unassessed" and automatic_status is not None
        and automatic_status != "forehead_clear"
    ):
        for key in ("mesh_top", "forehead_a", "forehead_b"):
            points[key] = None
    return points


def forehead_note(visibility: str, automatic_status: str | None = None) -> str:
    if visibility == "unassessed" and automatic_status is not None:
        from .automatic_forehead import EXPLANATIONS

        return "Automatic visibility assessment: " + EXPLANATIONS[automatic_status]
    label = FOREHEAD_VISIBILITY.get(visibility, FOREHEAD_VISIBILITY["unassessed"])
    return (
        f"User-reviewed visibility: {label}. "
        + ("Hairline readings use the visible, manually placed central hairline."
           if visibility == "visible" else
           "Hidden or uncertain hairline readings are unavailable. For hairline-defined "
           "proportions, add a clear frontal photo with hair lifted away from the skin. "
           "Do not place the hairline on the edge of a fringe or guess behind hair.")
    )


def render_forehead_warning(frontal: list[dict] | None) -> str:
    if not any(r.get("forehead_visibility") == "forehead_hidden" for r in frontal or []):
        return ""
    return ('<p class="review-confirm"><strong>Hair covers the forehead in a reviewed photo.</strong> '
            'Automatic forehead-dependent regional readings using that photo are omitted. '
            'The pillar cards and geometry score still use their original mesh proxies; '
            'this is not a hairline-corrected score. Add a clear frontal photo with hair '
            'lifted away to review the hairline and forehead.</p>')

Coordinate = Annotated[float, Field(strict=True, allow_inf_nan=False, ge=0, le=1)]
Point = Annotated[list[Coordinate], Field(min_length=2, max_length=2)]


class PointMap(RootModel[dict[str, Point | None]]):
    model_config = ConfigDict(strict=True)


class ReviewConflict(ValueError):
    """The page no longer refers to the current photo/review revision."""


def validate_points(raw: Any) -> dict[str, list[float] | None]:
    if not isinstance(raw, dict) or set(raw) != set(ANCHORS):
        raise ValueError("The frontal landmark fields are incomplete or unsupported.")
    raw = PointMap.model_validate(raw).root
    output = {}
    for key, pair in raw.items():
        if pair is None:
            output[key] = None
        elif (
            not isinstance(pair, list)
            or len(pair) != 2
            or any(
                isinstance(v, bool)
                or not isinstance(v, (float, int))
                or not math.isfinite(v)
                or not 0 <= v <= 1
                for v in pair
            )
        ):
            raise ValueError(
                "Use finite coordinates from 0 to 100 percent, or mark the point unavailable."
            )
        else:
            output[key] = [float(v) for v in pair]
    return output


def save_review(
    current: dict,
    raw: Any,
    *,
    expected_revision: int,
    binding: str,
    acknowledged: bool = False,
    forehead_visibility: str = "unassessed",
) -> dict:
    if (
        isinstance(expected_revision, bool)
        or not isinstance(expected_revision, int)
        or expected_revision != current["revision"]
        or binding != current["binding"]
    ):
        raise ReviewConflict(
            "This review changed in another tab or the photograph changed. Reload before saving."
        )
    points = validate_points(raw)
    if points["hairline"] is not None and forehead_visibility == "unassessed":
        raise ValueError("Confirm that the central hairline is visible before measuring it.")
    points = forehead_points(points, forehead_visibility, current.get("automatic_forehead_status"))
    warnings = geometry_warnings(points)
    if warnings and not acknowledged:
        raise ValueError(
            "Check landmark order using Calculate draft readings, then correct points or acknowledge the placement warnings."
        )
    return {
        **deepcopy(current),
        "points": points,
        "warnings": warnings,
        "reviewed": True,
        "forehead_visibility": forehead_visibility,
        "visibility_pending": False,
        "revision": expected_revision + 1,
    }


def _values(p: dict[str, list[float]]) -> dict[str, float]:
    def horizontal(a, b):
        return abs(p[a][0] - p[b][0])

    def vertical(a, b):
        return abs(p[a][1] - p[b][1])

    def brow():
        return (p["brow_head_a"][1] + p["brow_head_b"][1]) / 2

    def segment(name):
        if name == "upper":
            return abs(brow() - p["mesh_top"][1])
        if name == "middle":
            return abs(p["nose_base"][1] - brow())
        return vertical("nose_base", "chin")

    def total():
        return sum(segment(n) for n in ("upper", "middle", "lower"))

    def htotal():
        return abs(brow() - p["hairline"][1]) + segment("middle") + segment("lower")

    functions = {
        "reviewed_facial_index": lambda: (
            vertical("mesh_top", "chin") / horizontal("cheek_a", "cheek_b")
        ),
        "reviewed_fwhr": lambda: (
            horizontal("cheek_a", "cheek_b") / abs(brow() - p["upper_lip"][1])
        ),
        "reviewed_upper_third": lambda: segment("upper") / total(),
        "reviewed_middle_third": lambda: segment("middle") / total(),
        "reviewed_lower_third": lambda: segment("lower") / total(),
        "reviewed_middle_lower_ratio": lambda: segment("middle") / segment("lower"),
        "reviewed_forehead_ratio": lambda: (
            horizontal("forehead_a", "forehead_b") / horizontal("cheek_a", "cheek_b")
        ),
        "reviewed_jaw_ratio": lambda: (
            horizontal("jaw_a", "jaw_b") / horizontal("cheek_a", "cheek_b")
        ),
        "reviewed_chin_jaw_ratio": lambda: (
            horizontal("chin_a", "chin_b") / horizontal("jaw_a", "jaw_b")
        ),
        "reviewed_eye_spacing": lambda: (
            horizontal("eye_inner_a", "eye_inner_b")
            / horizontal("eye_outer_a", "eye_outer_b")
        ),
        "reviewed_eye_almondness": lambda: (
            (
                vertical("eye_top_a", "eye_bottom_a")
                + vertical("eye_top_b", "eye_bottom_b")
            )
            / (
                horizontal("eye_outer_a", "eye_inner_a")
                + horizontal("eye_outer_b", "eye_inner_b")
            )
        ),
        "reviewed_brow_position": lambda: (
            sum(p["eye_centre_" + s][1] - p["brow_peak_" + s][1] for s in ("a", "b"))
            / 2
            / vertical("mesh_top", "chin")
        ),
        "reviewed_alar_ratio": lambda: (
            horizontal("nose_a", "nose_b") / horizontal("cheek_a", "cheek_b")
        ),
        "reviewed_naso_oral": lambda: (
            horizontal("nose_a", "nose_b") / horizontal("mouth_a", "mouth_b")
        ),
        "reviewed_mouth_ratio": lambda: (
            horizontal("mouth_a", "mouth_b") / horizontal("cheek_a", "cheek_b")
        ),
        "reviewed_lip_lower_face": lambda: (
            vertical("upper_lip", "lower_lip") / segment("lower")
        ),
        "reviewed_philtrum_ratio": lambda: (
            vertical("nose_base", "upper_lip") / segment("lower")
        ),
        "hairline_facial_index": lambda: (
            vertical("hairline", "chin") / horizontal("cheek_a", "cheek_b")
        ),
        "hairline_upper_share": lambda: abs(brow() - p["hairline"][1]) / htotal(),
        "hairline_middle_share": lambda: segment("middle") / htotal(),
        "hairline_lower_share": lambda: segment("lower") / htotal(),
    }
    result = {}
    for key, func in functions.items():
        try:
            value = func()
            if math.isfinite(value):
                result[key] = value
        except (KeyError, ZeroDivisionError):
            continue  # Partial review: only dependent measurements are unavailable.
    if all(
        k in p for k in ("eye_outer_a", "eye_inner_a", "eye_outer_b", "eye_inner_b")
    ):
        widths = [horizontal("eye_outer_" + s, "eye_inner_" + s) for s in ("a", "b")]
        if min(widths) > 1e-8:
            result["reviewed_canthal_tilt"] = (
                sum(
                    math.degrees(
                        math.atan2(p["eye_inner_" + s][1] - p["eye_outer_" + s][1], w)
                    )
                    for s, w in zip(("a", "b"), widths)
                )
                / 2
            )
    return result


def calculate(raw: Any, width: int, height: int) -> list[dict]:
    if any(
        isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in (width, height)
    ):
        raise ValueError("Invalid original photograph dimensions.")
    normalized = validate_points(raw)
    points = {
        k: [v[0] * width, v[1] * height] for k, v in normalized.items() if v is not None
    }
    values = _values(points)
    return [
        dict(
            id=m.id,
            label=m.label,
            region=m.region,
            value=round(values[m.id], 4),
            unit=m.unit,
            definition=m.definition,
            percentile=None,
            method=REGISTRY_VERSION,
        )
        for m in METRICS
        if m.id in values and all(k in points for k in m.requires)
    ]


def detected_points(path: Any) -> dict:
    import fa_common as common
    from facial_catalogue import L_BROW, R_BROW

    points, _image, (width, height) = common.detect_landmarks(str(path))
    output = {key: None for key in ANCHORS}
    if points is None:
        return output
    for key, (_label, index, _region) in ANCHORS.items():
        if index is not None:
            output[key] = [
                float(points[index][0] / width),
                float(points[index][1] / height),
            ]
    for side, ring, brow in (
        ("a", common.LEFT_EYE_RING, L_BROW),
        ("b", common.RIGHT_EYE_RING, R_BROW),
    ):
        centre = points[ring].mean(axis=0)
        peak = min((points[i] for i in brow), key=lambda p: p[1])
        output["eye_centre_" + side] = [
            float(centre[0] / width),
            float(centre[1] / height),
        ]
        output["brow_peak_" + side] = [float(peak[0] / width), float(peak[1] / height)]
    return {
        k: v if v is None or all(math.isfinite(x) and 0 <= x <= 1 for x in v) else None
        for k, v in output.items()
    }


def observation_binding(entry: Any, instrument: str) -> str:
    value = [
        entry.sha256,
        entry.width,
        entry.height,
        instrument,
        measurement_contract(),
    ]
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def public_review(record: dict, width: int, height: int) -> dict | None:
    visibility = record.get("forehead_visibility", "unassessed")
    visibility_only = not record.get("reviewed") and record.get("visibility_pending") is True
    if not record.get("reviewed") and not visibility_only:
        return None
    if visibility_only and visibility not in {"hairline_hidden", "forehead_hidden", "uncertain"}:
        return None
    automatic_status = record.get("automatic_forehead_status")
    points = forehead_points(
        {key: None for key in ANCHORS} if visibility_only else record["points"], visibility,
        automatic_status,
    )
    return dict(
        **({"automatic_forehead_status": automatic_status} if automatic_status is not None else {}),
        **({"visibility_only": True} if visibility_only else {}),
        method=REGISTRY_VERSION,
        contract=measurement_contract(),
        revision=record["revision"],
        forehead_visibility=visibility,
        measurements=calculate(points, width, height),
        warnings=geometry_warnings(points),
        diagram=dict(
            points=points, width=width, height=height
        ),
        note=REVIEW_NOTE,
    )


def render_reviewed(payload: dict | None) -> str:
    if not payload:
        return ""
    photo = payload.get("photo_number")
    heading = "Reviewed frontal measurements" + (
        f" · Photo {photo}"
        if isinstance(photo, int) and not isinstance(photo, bool) and photo > 0
        else ""
    )
    if payload.get("visibility_only"):
        return (f'<section class="feature-chapter"><h2>{html.escape(heading.replace("Reviewed frontal measurements", "Saved forehead coverage"))}</h2>'
                '<p>Saved hair coverage is retained as an exclusion. Recheck visibility '
                'and confirm landmarks before using restored measurements.</p>'
                f'<p>{html.escape(forehead_note(payload["forehead_visibility"]))}</p></section>')
    warnings = "".join(
        f"<li>{html.escape(w)}</li>" for w in payload.get("warnings", [])
    )
    rows = "".join(
        f"<tr><th>{html.escape(r['label'])}</th><td>{r['value']:.4f} {html.escape(r['unit'])}</td><td>{html.escape(r['definition'])}</td></tr>"
        for r in payload["measurements"]
    )
    visibility_note = forehead_note(payload.get("forehead_visibility", "unassessed"),
                                    payload.get("automatic_forehead_status"))
    return f'<section class="feature-chapter"><h2>{html.escape(heading)}</h2><ul>{warnings}</ul>{render_review_diagram(payload.get("diagram"))}<p>{html.escape(visibility_note)}</p><p>{html.escape(REVIEW_NOTE)}</p><div class="table-wrap"><table><thead><tr><th>Measurement</th><th>Reading</th><th>Definition</th></tr></thead><tbody>{rows}</tbody></table></div></section>'


def safe_review(raw: Any) -> dict | None:
    """Export only canonical definitions and bounded numerical review geometry."""
    if (
        not isinstance(raw, dict)
        or raw.get("method") != REGISTRY_VERSION
        or raw.get("contract") != measurement_contract()
    ):
        return None
    revision = raw.get("revision")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
        return None
    diagram = raw.get("diagram")
    if not isinstance(diagram, dict) or set(diagram) != {"width", "height", "points"}:
        return None
    try:
        width, height = diagram["width"], diagram["height"]
        if any(
            isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= 100_000
            for v in (width, height)
        ):
            return None
        visibility = raw.get("forehead_visibility", "unassessed")
        visibility_only = raw.get("visibility_only", False)
        if type(visibility_only) is not bool or (visibility_only and visibility not in {
            "hairline_hidden", "forehead_hidden", "uncertain"
        }):
            return None
        automatic_status = raw.get("automatic_forehead_status")
        points = forehead_points(diagram["points"], visibility, automatic_status)
        if visibility_only:
            points = {key: None for key in ANCHORS}
        rows = calculate(points, width, height)
    except ValueError:
        return None
    photo = raw.get("photo_number")
    if photo is not None and (
        isinstance(photo, bool)
        or not isinstance(photo, int)
        or not 1 <= photo <= 1_000_000
    ):
        return None
    return dict(
        **({"automatic_forehead_status": automatic_status} if automatic_status is not None else {}),
        **({"visibility_only": True} if visibility_only else {}),
        **({"photo_number": photo} if photo is not None else {}),
        method=REGISTRY_VERSION,
        contract=measurement_contract(),
        revision=revision,
        forehead_visibility=visibility,
        measurements=rows,
        warnings=geometry_warnings(points),
        diagram=dict(points=points, width=width, height=height),
        note=REVIEW_NOTE,
    )


def render_review_diagram(raw: Any) -> str:
    if not isinstance(raw, dict):
        return ""
    try:
        points = validate_points(raw["points"])
        width, height = raw["width"], raw["height"]
        calculate(points, width, height)
    except (KeyError, TypeError, ValueError):
        return ""
    scale = max(width / 300, height / 380)
    marks = ""
    for i, key in enumerate(ANCHORS, 1):
        pair = points[key]
        if pair is None:
            continue
        x, y = pair[0] * width / scale + 10, pair[1] * height / scale + 10
        marks += f'<circle cx="{x:.3f}" cy="{y:.3f}" r="3"/><text x="{x + 5:.3f}" y="{y - 5:.3f}">{i}</text>'
    return f'<figure><svg class="reviewed-diagram" viewBox="0 0 {width / scale + 20:.3f} {height / scale + 20:.3f}" role="img" aria-label="Reviewed landmark positions; aspect ratio preserved" xmlns="http://www.w3.org/2000/svg"><g fill="#164b42" font-size="8">{marks}</g></svg><figcaption>Saved reviewed points on one photograph; original aspect ratio preserved. No source image is included.</figcaption></figure>'


def geometry_warnings(raw: Any) -> list[str]:
    points = validate_points(raw)
    warnings = []
    if all(points[k] is not None for k in ("eye_inner_a", "eye_inner_b")):
        centre = (points["eye_inner_a"][0] + points["eye_inner_b"][0]) / 2
        for side in ("a", "b"):
            inner, outer = points["eye_inner_" + side], points["eye_outer_" + side]
            if (
                inner is not None
                and outer is not None
                and abs(outer[0] - centre) <= abs(inner[0] - centre)
            ):
                warnings.append(
                    f"Eye {side.upper()}: the outer corner is not farther from the central eye gap than the inner corner. Check for swapped points."
                )
    if all(
        points[k] is not None
        for k in ("brow_head_a", "brow_head_b", "nose_base", "chin")
    ):
        brow = (points["brow_head_a"][1] + points["brow_head_b"][1]) / 2
        if not brow < points["nose_base"][1] < points["chin"][1]:
            warnings.append(
                "The brow, nose base and chin are not vertically ordered on this photograph. Check head orientation and landmark labels."
            )
        if points["hairline"] is not None and points["hairline"][1] >= brow:
            warnings.append(
                "The central hairline is not above the medial brows. Check that the visible hairline point is correctly placed."
            )
    return warnings
