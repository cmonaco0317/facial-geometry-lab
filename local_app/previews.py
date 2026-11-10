"""Original, local SVG aids: no generated anatomy and no analysis-compatible object."""

from __future__ import annotations
import base64
from dataclasses import dataclass
import hashlib
import html
import json
import math
from .photos import PhotoCatalog, MAX_PIXELS

VERSION = "illustrative-overlay-v1"


@dataclass(frozen=True)
class Preview:
    content: bytes
    source_sha256: str
    kind: str = "illustrative-preview"


def make_preview(
    source: bytes,
    *,
    style="eyewear",
    x=0.5,
    y=0.4,
    scale=0.4,
    review=None,
    original=False,
):
    import cv2
    import numpy as np

    if not isinstance(source, bytes) or len(source) > 10_000_000:
        raise ValueError("Select a bounded source photograph.")
    if style not in {"eyewear", "hair", "reviewed"}:
        raise ValueError("Unsupported local overlay.")
    for value, lo, hi in ((x, 0, 1), (y, 0, 1), (scale, 0.1, 0.8)):
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or not lo <= value <= hi
        ):
            raise ValueError("Overlay placement is out of bounds.")
    format = PhotoCatalog._sniff_format(source)
    dimensions = PhotoCatalog._declared_dimensions(source, format)
    if not dimensions or dimensions[0] * dimensions[1] > MAX_PIXELS:
        raise ValueError("Unsupported source photograph.")
    image = cv2.imdecode(np.frombuffer(source, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Source photograph could not be decoded.")
    ratio = min(1.0, 1200 / max(image.shape[:2]))
    if ratio < 1:
        image = cv2.resize(
            image, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA
        )
    h, w = image.shape[:2]
    ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise ValueError("Source photograph could not be rendered.")
    cx, cy, size = x * w, y * h, scale * w
    # Original geometric assets authored for this project; user-adjusted, not a fit prediction.
    if style == "eyewear":
        overlay = f'<g fill="none" stroke="#1b3631" stroke-width="{max(1, size * 0.018):.3f}"><rect x="{cx - size * 0.5:.3f}" y="{cy - size * 0.15:.3f}" width="{size * 0.43:.3f}" height="{size * 0.3:.3f}" rx="{size * 0.08:.3f}"/><rect x="{cx + size * 0.07:.3f}" y="{cy - size * 0.15:.3f}" width="{size * 0.43:.3f}" height="{size * 0.3:.3f}" rx="{size * 0.08:.3f}"/><path d="M {cx - size * 0.07:.3f},{cy:.3f} Q {cx:.3f},{cy - size * 0.1:.3f} {cx + size * 0.07:.3f},{cy:.3f}"/></g>'
    elif style == "hair":
        overlay = f'<path d="M {cx - size * 0.55:.3f},{cy + size * 0.25:.3f} Q {cx - size * 0.65:.3f},{cy - size * 0.7:.3f} {cx:.3f},{cy - size * 0.55:.3f} Q {cx + size * 0.65:.3f},{cy - size * 0.7:.3f} {cx + size * 0.55:.3f},{cy + size * 0.25:.3f} Q {cx + size * 0.32:.3f},{cy - size * 0.45:.3f} {cx:.3f},{cy - size * 0.25:.3f} Q {cx - size * 0.32:.3f},{cy - size * 0.45:.3f} {cx - size * 0.55:.3f},{cy + size * 0.25:.3f}" fill="#173b35" fill-opacity=".65"/>'
    else:
        if not isinstance(review, dict) or not review.get("reviewed"):
            raise ValueError("Save a reviewed frontal annotation first.")
        from .landmark_review import validate_points

        overlay = ""
        for key, color in (("automatic", "#e3aa46"), ("points", "#39d0af")):
            points = validate_points(review.get(key))
            overlay += "".join(
                f'<circle cx="{p[0] * w:.3f}" cy="{p[1] * h:.3f}" r="{max(1, w * 0.005):.3f}" fill="{color}"/>'
                for p in points.values()
                if p
            )
    if original:
        overlay = ""
    source_hash = hashlib.sha256(source).hexdigest()
    metadata = dict(
        kind="illustrative-preview",
        version=VERSION,
        source="Explicitly selected local source photograph",
        source_revision=review.get("revision") if style == "reviewed" else None,
        asset=style,
        asset_license="Original project geometry; LICENSE",
        parameters=dict(x=x, y=y, scale=scale, original=original),
        label="Illustrative only; not a measurement, prediction or fit prescription.",
    )
    label = "Illustrative only · " + (
        "Original gold / reviewed green points"
        if style == "reviewed"
        else style + " placement experiment"
    )
    # A fixed 60-unit footer keeps provenance visible even when the overlay is hidden.
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h + 60}" viewBox="0 0 {w} {h + 60}"><metadata>{html.escape(json.dumps(metadata, sort_keys=True))}</metadata><image width="{w}" height="{h}" href="data:image/jpeg;base64,{base64.b64encode(jpeg).decode()}"/><g id="illustrative-overlay">{overlay}</g><rect y="{h}" width="{w}" height="60" fill="#f5f2e9"/><text x="{w * 0.025:.3f}" y="{h + 22}" font-size="{min(16, max(1, w * 0.026)):.3f}" fill="#163e36">{html.escape(label)}</text><text x="{w * 0.025:.3f}" y="{h + 44}" font-size="{min(12, max(1, w * 0.019)):.3f}" fill="#163e36">Local illustration · Not for remeasurement · {VERSION}</text></svg>'
    return Preview(svg.encode(), source_hash)
