"""Photo-bound appearance panels; pixel descriptors never imply tissue diagnosis."""

from __future__ import annotations

import base64
from copy import deepcopy
import hashlib
import html
from math import isfinite
from numbers import Real
from typing import Any

import cv2
import numpy as np

VERSION = "appearance-image-regions-v1"
FAMILIES = {
    "smile": "Smile and visible mouth opening",
    "hair": "Hair and framing",
    "eyes": "Eyes and surrounding frame",
    "skin": "Visible surface appearance",
    "ears": "Ears",
    "cheeks": "Cheeks",
    "jaw": "Jaw outline",
    "neck": "Neck",
}
NOTES = {
    "smile": "A smile candidate uses model expression outputs. Mouth opening is an image-space mesh ratio, not dental alignment, tooth health or smile quality.",
    "hair": "Select a visible hair region to describe its image appearance. No hairline, density, recession or hair-health inference is enabled.",
    "eyes": "The crop and eyelid-opening ratio describe this expression and camera view. Glasses and reflections can obscure the lids; no eyesight or fatigue inference is enabled.",
    "skin": "Brightness and contrast describe pixels in a proposed cheek patch, including light, shadows, makeup and facial hair. They do not measure pigmentation, skin tone, texture quality or a skin condition.",
    "ears": "Ear visibility requires your inspection and a corrected crop. No automatic ear detection, prominence or anatomical rating is enabled.",
    "cheeks": "The proposed cheek crop shows visible surface appearance. It does not establish bone position, volume or body fat.",
    "jaw": "The proposed lower-face crop shows the surface outline, including facial hair and shadows. It does not locate the underlying bone or establish bite.",
    "neck": "Neck visibility requires your inspection and a corrected crop. No neck-angle, posture, thyroid or health inference is enabled.",
}
ROLES = ("unknown", "neutral", "smile", "profile")
VISIBILITY = ("unassessed", "visible", "obscured", "not_visible", "uncertain")
LIGHTING = ("unknown", "even", "uneven")
COVERINGS = ("unknown", "present", "absent")
LANDMARKS = {
    "smile": (13, 14, 61, 291),
    "eyes": (159, 145, 33, 133, 386, 374, 362, 263),
}
GEOMETRY_DEFINITIONS = {
    "mouth_opening_ratio": "Euclidean p13–p14 distance / p61–p291 distance in original image pixels.",
    "mean_eyelid_opening_ratio": "Mean of the Euclidean p159–p145 / p33–p133 and p386–p374 / p362–p263 distance ratios in original image pixels.",
}


def _box(value: Any) -> list[float]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 4
        or any(
            isinstance(v, bool)
            or not isinstance(v, Real)
            or not isfinite(v)
            or not 0 <= v <= 1
            for v in value
        )
        or value[2] <= value[0]
        or value[3] <= value[1]
    ):
        raise ValueError(
            "Use a nonempty region within the photograph (0–1 coordinates)."
        )
    return [float(v) for v in value]


def _jpeg(image: np.ndarray, maximum: int = 480) -> str:
    scale = min(1.0, maximum / max(image.shape[:2]))
    if scale < 1:
        height, width = image.shape[:2]
        image = cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 86])
    if not ok:
        raise ValueError("The appearance photograph could not be encoded.")
    return base64.b64encode(encoded).decode("ascii")


def _region(
    points: np.ndarray,
    ids: tuple[int, ...],
    width: int,
    height: int,
    padding: float = 0.03,
) -> list[float] | None:
    selected = points[list(ids)] / [width, height]
    if not np.isfinite(selected).all() or (selected < 0).any() or (selected > 1).any():
        return None
    lo, hi = selected.min(axis=0) - padding, selected.max(axis=0) + padding
    try:
        return _box(
            [max(0.0, lo[0]), max(0.0, lo[1]), min(1.0, hi[0]), min(1.0, hi[1])]
        )
    except ValueError:
        return None


def validate_points(family: str, value: Any) -> dict:
    expected = {str(i) for i in LANDMARKS.get(family, ())}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(
            "Use the specified appearance endpoints; leave hidden points unavailable."
        )
    result = {}
    for key, pair in value.items():
        if pair is not None and (
            not isinstance(pair, (list, tuple))
            or len(pair) != 2
            or any(
                isinstance(v, bool)
                or not isinstance(v, Real)
                or not isfinite(v)
                or not 0 <= v <= 1
                for v in pair
            )
        ):
            raise ValueError(
                "Appearance endpoints must be finite photo coordinates from 0 to 1."
            )
        result[key] = [float(v) for v in pair] if pair is not None else None
    return result


def _shape(family: str, normalized: dict, width: int, height: int) -> dict[str, float]:
    points = {
        int(i): np.asarray(p) * [width, height]
        for i, p in normalized.items()
        if p is not None
    }

    def ratio(a: int, b: int, c: int, d: int) -> float | None:
        if not {a, b, c, d} <= set(points):
            return None
        denominator = float(np.linalg.norm(points[c] - points[d]))
        return (
            float(np.linalg.norm(points[a] - points[b])) / denominator
            if denominator > 0
            else None
        )

    if family == "smile":
        smile = ratio(13, 14, 61, 291)
        return {"mouth_opening_ratio": round(smile, 6)} if smile is not None else {}
    if family != "eyes":
        return {}
    eyes = [ratio(159, 145, 33, 133), ratio(386, 374, 362, 263)]
    return (
        {"mean_eyelid_opening_ratio": round(sum(eyes) / 2, 6)}
        if all(v is not None for v in eyes)
        else {}
    )


def observe_appearance(photos: tuple, records: tuple) -> list[dict]:
    """Keep one bounded thumbnail per original and explicit proposed regions.

    All input points are original pixel coordinates. No claim that a proposed
    rectangle contains only the named anatomy, or that a detector saw occlusion.
    """
    by_id = {r["photo"]: r for r in records}
    result = []
    for number, photo in enumerate(photos, 1):
        image = cv2.imread(str(photo.path))
        if image is None:
            continue
        height, width = image.shape[:2]
        raw = by_id.get(photo.entry.photo_id, {})
        points = np.asarray(raw.get("landmarks", []), dtype=float)
        usable = points.shape == (468, 2)
        regions = {key: None for key in FAMILIES}
        if usable:
            regions.update(
                {
                    "smile": _region(points, (61, 291, 13, 14, 0, 17), width, height),
                    "eyes": _region(
                        points, (33, 133, 362, 263, 70, 300, 145, 374), width, height
                    ),
                    "skin": _region(points, (50, 101, 118, 119), width, height, 0.012),
                    "cheeks": _region(points, (50, 101, 118, 119, 234), width, height),
                    "jaw": _region(points, (172, 397, 152), width, height),
                }
            )
        expression = raw.get("expression") or {}
        smile = all(
            type(expression.get(k)) in (int, float)
            and isfinite(expression[k])
            and 0.45 <= expression[k] <= 1
            for k in ("mouthSmileLeft", "mouthSmileRight")
        )
        endpoints = {}
        for family, ids in LANDMARKS.items():
            endpoints[family] = {
                str(i): (
                    (points[i] / [width, height]).tolist()
                    if usable
                    and np.isfinite(points[i]).all()
                    and (points[i] >= 0).all()
                    and (points[i] <= [width, height]).all()
                    else None
                )
                for i in ids
            }
        result.append(
            dict(
                photo_id=photo.entry.photo_id,
                photo_number=number,
                source_sha256=hashlib.sha256(photo.path.read_bytes()).hexdigest(),
                width=width,
                height=height,
                jpeg_b64=_jpeg(image),
                role="smile" if regions["smile"] is not None and smile else "unknown",
                role_method=(
                    "detector-smile-candidate-v1"
                    if regions["smile"] is not None and smile
                    else "unassessed"
                ),
                regions=regions,
                points=endpoints,
                pose={
                    k: (
                        float(raw[k])
                        if type(raw.get(k)) in (int, float) and isfinite(raw[k])
                        else None
                    )
                    for k in ("yaw", "pitch", "roll")
                },
            )
        )
    return result


def review_appearance(
    raw: list[dict],
    current: dict,
    *,
    photo_number: int,
    family: str,
    revision: int,
    action: str,
    role: str = "unknown",
    visibility: str = "unassessed",
    box: Any = None,
    lighting: str = "unknown",
    glasses: str = "unknown",
    makeup: str = "unknown",
    facial_hair: str = "unknown",
    note: str = "",
    points: dict | None = None,
) -> dict:
    """Append a source-bound revision; reset removes only this region override."""
    photo = next((p for p in raw if p["photo_number"] == photo_number), None)
    if photo is None or family not in FAMILIES or action not in {"save", "reset"}:
        raise ValueError("Choose a current photograph and appearance region.")
    if type(revision) is not int or revision != current.get("revision", 0):
        raise ValueError("This appearance form is stale. Reload the report.")
    if (
        role not in ROLES
        or visibility not in VISIBILITY
        or lighting not in LIGHTING
        or any(v not in COVERINGS for v in (glasses, makeup, facial_hair))
        or not isinstance(note, str)
        or len(note) > 240
    ):
        raise ValueError(
            "Unsupported appearance selection or note longer than 240 characters."
        )
    updated = (
        deepcopy(current)
        if current
        else {"revision": 0, "overrides": {}, "history": []}
    )
    key = f'{photo["photo_id"]}:{family}'
    if action == "reset":
        updated["overrides"].pop(key, None)
    else:
        updated.setdefault("photo_settings", {})[photo["photo_id"]] = dict(
            source_sha256=photo["source_sha256"], role=role
        )
        updated["overrides"][key] = dict(
            source_sha256=photo["source_sha256"],
            box=_box(box),
            role=role,
            visibility=visibility,
            lighting=lighting,
            glasses=glasses,
            makeup=makeup,
            facial_hair=facial_hair,
            note=note.strip(),
            points=validate_points(family, points) if points is not None else None,
        )
    if len(updated["history"]) >= 500:
        raise ValueError(
            "The appearance revision limit was reached. Export this report before starting a new session."
        )
    updated["revision"] += 1
    updated["history"].append(
        dict(
            revision=updated["revision"],
            photo_number=photo_number,
            family=family,
            action=action,
            source_sha256=photo["source_sha256"],
            previous=deepcopy(current.get("overrides", {}).get(key)),
            current=deepcopy(updated["overrides"].get(key)),
            previous_photo=deepcopy(
                current.get("photo_settings", {}).get(photo["photo_id"])
            ),
            current_photo=deepcopy(
                updated.get("photo_settings", {}).get(photo["photo_id"])
            ),
        )
    )
    return updated


def _pixels(photo: dict, box: list[float]) -> tuple[str, dict]:
    image = cv2.imdecode(
        np.frombuffer(
            base64.b64decode(photo["jpeg_b64"], validate=True), dtype=np.uint8
        ),
        cv2.IMREAD_COLOR,
    )
    h, w = image.shape[:2]
    x0, y0, x1, y1 = box
    crop = image[
        min(h - 1, int(y0 * h)) : max(int(y0 * h) + 1, min(h, int(np.ceil(y1 * h)))),
        min(w - 1, int(x0 * w)) : max(int(x0 * w) + 1, min(w, int(np.ceil(x1 * w)))),
    ]
    encoded = _jpeg(crop, 240)
    exported = cv2.imdecode(
        np.frombuffer(base64.b64decode(encoded), dtype=np.uint8), cv2.IMREAD_COLOR
    )
    gray = cv2.cvtColor(exported, cv2.COLOR_BGR2GRAY).astype(float)
    return encoded, dict(
        mean_brightness=round(float(gray.mean()), 2),
        brightness_spread=round(float(gray.std()), 2),
        dark_or_clipped_fraction=round(float(np.mean((gray < 20) | (gray > 235))), 4),
    )


def appearance_payload(raw: list[dict] | None, review: dict | None = None) -> dict:
    review = review or {}
    families = []
    for family, title in FAMILIES.items():
        findings = []
        for photo in raw or []:
            override = review.get("overrides", {}).get(
                f'{photo["photo_id"]}:{family}', {}
            )
            if override.get("source_sha256") != photo["source_sha256"]:
                override = {}
            settings = review.get("photo_settings", {}).get(photo["photo_id"], {})
            if settings.get("source_sha256") != photo["source_sha256"]:
                settings = {}
            role = settings.get("role", photo["role"])
            box = override.get("box", photo["regions"][family])
            visible = override.get("visibility", "unassessed")
            if (
                box is None
                or visible in {"obscured", "not_visible", "uncertain"}
                or (family == "smile" and role != "smile")
                or (family in {"hair", "ears", "neck"} and visible != "visible")
            ):
                continue
            crop, pixels = _pixels(photo, _box(box))
            intensity = (
                "low-intensity-crop"
                if pixels["mean_brightness"] < 60
                else (
                    "high-intensity-crop"
                    if pixels["mean_brightness"] > 200
                    else "midrange-intensity-crop"
                )
            )
            points = override.get("points") or photo["points"].get(family, {})
            lighting = override.get("lighting", "unknown")
            confounds = ", ".join(
                f'{k.replace("_", " ")}: {override.get(k, "unknown")}'
                for k in ("glasses", "makeup", "facial_hair")
            )
            findings.append(
                dict(
                    photo_number=photo["photo_number"],
                    source_sha256=photo["source_sha256"],
                    source_dimensions=[photo["width"], photo["height"]],
                    box=box,
                    jpeg_b64=crop,
                    role=role,
                    method=(
                        "user-reviewed-image-region-v1"
                        if override and visible == "visible"
                        else (
                            "user-adjusted-region-proposal-v1"
                            if override
                            else (
                                photo["role_method"]
                                if family == "smile"
                                else "mesh-region-proposal-v1"
                            )
                        )
                    ),
                    visibility=visible,
                    measurements=_shape(
                        family, points, photo["width"], photo["height"]
                    ),
                    points={
                        f"p{i}": (
                            [p[0] * photo["width"], p[1] * photo["height"]]
                            if p is not None
                            else None
                        )
                        for i, p in points.items()
                    },
                    geometry_method=(
                        "user-adjusted-image-points-v1"
                        if override.get("points") is not None
                        else "detector-image-points-v1"
                    ),
                    pixel_descriptors=pixels,
                    pixel_method="saved-jpeg-gray-v1/" + intensity,
                    pose=photo["pose"],
                    note=override.get("note", ""),
                    uncertainty=f"Uncalibrated image appearance. Pixel method: {intensity} (heuristic image intensity, not skin tone or exposure accuracy). Lighting: {lighting}; {confounds}. Camera/white balance are uncontrolled. "
                    "Pixel descriptors use the saved downsampled JPEG crop (0–255 intensity). Region visibility and identity are unverified unless you inspect them; a reviewed note is user supplied, not a qualified assessment.",
                )
            )
        families.append(
            dict(
                id=family,
                title=title,
                findings=findings,
                note=NOTES[family],
                missing_reason="No eligible visible region. Inspect/correct a source crop"
                + (
                    " and select a smile view."
                    if family == "smile"
                    else "; add a clearer view if needed."
                ),
            )
        )
    return dict(
        version=VERSION,
        revision=review.get("revision", 0),
        geometry_definitions=GEOMETRY_DEFINITIONS.copy(),
        pixel_definition="Decode the retained JPEG to BGR, convert with OpenCV COLOR_BGR2GRAY (0–255). Mean and population standard deviation round to 2 decimals; fraction of pixels <20 or >235 rounds to 4 decimals. No skin-tone, diagnostic, percentile or score interpretation.",
        families=families,
        scope="Photo appearance and mesh proposals; no anatomical validation, health assessment, percentile or score weight.",
    )


def render_appearance(
    payload: dict, *, action: str | None = None, csrf: str | None = None
) -> str:
    esc = html.escape
    sections = []
    for family in payload["families"]:
        findings = []
        for row in family["findings"]:
            numbers = ", ".join(
                f'{k.replace("_", " ")}: {v:.6g}'
                for k, v in row["measurements"].items()
            )
            pixels = row["pixel_descriptors"]
            findings.append(
                f'<article><figure><img width="240" class="appearance-image" alt="Photo {row["photo_number"]} proposed {esc(family["id"])} crop" src="data:image/jpeg;base64,{row["jpeg_b64"]}"><figcaption>Photo {row["photo_number"]} · {esc(row["method"])} · {esc(row["role"])} · visibility: {esc(row["visibility"])}</figcaption></figure>'
                f'<p>{esc(numbers)}</p><p>Image brightness {pixels["mean_brightness"]:.2f}/255; spread {pixels["brightness_spread"]:.2f}; dark/clipped pixels {pixels["dark_or_clipped_fraction"]:.1%}.</p>'
                f'<p>{esc(row["uncertainty"])}</p><p>{esc(row["note"])}</p>'
                f'<details><summary>Source and definition</summary><p>Original normalized photo SHA-256: {row["source_sha256"]}. Crop x0,y0,x1,y1: {esc(str(row["box"]))}. '
                f'Points in original photo pixels: {esc(str(row["points"]))}. Geometry method: {esc(row["geometry_method"])}. '
                "Mouth opening = distance p13–p14 / p61–p291. Eye opening = mean of p159–p145 / p33–p133 and p386–p374 / p362–p263. Ratios use original image pixels; crop edits change pixel descriptors. Appearance endpoint edits change these ratios and remain separate from the frontal geometry editor.</p></details></article>"
            )
        sections.append(
            f'<section class="feature-chapter" id="appearance-{esc(family["id"],quote=True)}"><h3>{esc(family["title"])}</h3><p>{esc(family["note"])}</p>'
            + ("".join(findings) or f'<p>{esc(family["missing_reason"])}</p>')
            + "</section>"
        )
    link = (
        f'<p><a href="{esc(action, quote=True)}">Inspect or correct appearance crops, view labels and conditions</a></p>'
        if action
        else ""
    )
    return f'<section id="appearance"><h2>Appearance in your photographs</h2><p>{esc(payload["scope"])}</p>{link}{"".join(sections)}</section>'
