"""Bounded, invertible local point warps for explicitly illustrative experiments."""

from __future__ import annotations

import base64
from decimal import Decimal
import html
import json
import math
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from .landmark_review import calculate, forehead_points, validate_points
from .measurement_registry import METRICS, REGISTRY_VERSION, measurement_contract
from .photo_experiments import VERSION, bind_svg, public_experiment, seal_entry
from .previews import make_preview

METHOD = "geometry-point-warp-v1"


def validate_geometry(entry: dict, metadata: dict) -> None:
    """Reconstruct retained numbers before showing them as a saved simulation."""
    try:
        data = entry["geometry"]
        if (
            data["method"] != METHOD
            or data["contract"] != measurement_contract()
            or data["calculation_method"] != REGISTRY_VERSION
        ):
            raise ValueError("Unsupported geometry experiment instrument.")
        width, height = data["dimensions"]
        if (
            any(type(v) is not int or not 1 <= v <= 100000 for v in (width, height))
            or width * height > 80000000
        ):
            raise ValueError("Invalid geometry experiment dimensions.")
        before = forehead_points(
            data["original_points"],
            data["forehead_visibility"],
            data.get("automatic_forehead_status"),
        )
        after = validate_points(data["simulated_points"])
        params = entry["parameters"]
        anchor, x, y, radius = (params[k] for k in ("anchor", "x", "y", "radius"))
        if (
            anchor not in before
            or before[anchor] is None
            or any(
                type(v) not in (int, float) or not math.isfinite(v) or not lo <= v <= hi
                for v, lo, hi in ((x, 0, 1), (y, 0, 1), (radius, 0.08, 0.4))
            )
        ):
            raise ValueError("Invalid saved geometry controls.")
        center = (before[anchor][0] * width, before[anchor][1] * height)
        delta = (x * width - center[0], y * height - center[1])
        radius_px = radius * min(width, height)
        if math.hypot(*delta) > 0.2 * radius_px:
            raise ValueError(
                "Saved geometry displacement exceeds the invertible bound."
            )
        expected_after = {}
        for key, p in before.items():
            weight = (
                float(_weight(p[0] * width, p[1] * height, center, radius_px))
                if p is not None
                else 0
            )
            expected_after[key] = (
                [p[0] + delta[0] * weight / width, p[1] + delta[1] * weight / height]
                if p is not None
                else None
            )
        if after != expected_after:
            raise ValueError("Saved simulated points do not follow the declared warp.")
        if before != data["original_points"] or entry["measurements"] != _readings(
            before, after, width, height
        ):
            raise ValueError(
                "Saved simulation readings do not reconstruct from their points."
            )
        if any(
            type(row[field]) is bool
            for row in entry["measurements"]
            for field in ("before", "after", "delta")
        ):
            raise ValueError("A simulation reading is not a number.")
        simulated = metadata["simulation_svg"]
        if (
            simulated.get("original_points") != before
            or simulated.get("simulated_points") != after
            or simulated.get("parameters") != entry["parameters"]
            or simulated.get("source_dimensions") != [width, height]
        ):
            raise ValueError("The illustration and retained geometry disagree.")
    except (KeyError, TypeError) as error:
        raise ValueError("Incomplete saved geometry experiment.") from error


def _weight(x, y, center, radius):
    return (
        np.maximum(0, 1 - ((x - center[0]) ** 2 + (y - center[1]) ** 2) / radius**2)
        ** 2
    )


def _raster(
    image: np.ndarray, dimensions: list[int], center: tuple, delta: tuple, radius: float
) -> tuple[bytes, float]:
    h, w = image.shape[:2]
    ow, oh = dimensions
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    original_x, original_y = gx * (ow / w), gy * (oh / h)
    u, v = original_x.copy(), original_y.copy()
    # Displacement <=0.2R bounds the field derivative below one. Ten fixed-point
    # iterations invert the forward point map; residual is numerical, not anatomical.
    for _ in range(10):
        weight = _weight(u, v, center, radius)
        u, v = original_x - delta[0] * weight, original_y - delta[1] * weight
    weight = _weight(u, v, center, radius)
    residual = float(
        np.sqrt(
            ((u + delta[0] * weight - original_x) * (w / ow)) ** 2
            + ((v + delta[1] * weight - original_y) * (h / oh)) ** 2
        ).max()
    )
    if not math.isfinite(residual) or residual > 0.1:
        raise ValueError(
            "This edit could not be inverted accurately enough to display."
        )
    warped = cv2.remap(
        image,
        (u * (w / ow)).astype(np.float32),
        (v * (h / oh)).astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(245, 245, 245),
    )
    ok, encoded = cv2.imencode(".jpg", warped, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise ValueError("The illustrative image could not be rendered.")
    return encoded.tobytes(), residual


def _readings(before: dict, after: dict, width: int, height: int) -> list[dict]:
    original = {r["id"]: r["value"] for r in calculate(before, width, height)}
    simulated = {r["id"]: r["value"] for r in calculate(after, width, height)}
    rows = []
    for metric in METRICS:
        a, b = original.get(metric.id), simulated.get(metric.id)
        missing = [k for k in metric.requires if after[k] is None]
        rows.append(
            dict(
                id=metric.id,
                label=metric.label.removeprefix("Reviewed "),
                unit=metric.unit,
                definition=metric.definition,
                inputs=list(metric.requires),
                before=a,
                after=b,
                delta=(
                    float(Decimal(str(b)) - Decimal(str(a)))
                    if a is not None and b is not None
                    else None
                ),
                method=METHOD,
                definition_version=measurement_contract(),
                percentile=None,
                withheld_reason=(
                    "Missing inputs: " + ", ".join(missing)
                    if missing
                    else "Collapsed or unavailable construction." if b is None else None
                ),
            )
        )
    return rows


def geometry_experiment(
    source: bytes,
    baseline: dict,
    *,
    photo_number: int,
    basis: str,
    anchor: str,
    x: float,
    y: float,
    radius: float = 0.2,
) -> dict:
    for value, lo, hi in ((x, 0, 1), (y, 0, 1), (radius, 0.08, 0.4)):
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or not lo <= value <= hi
        ):
            raise ValueError("Choose finite, bounded geometry controls.")
    if (
        type(baseline.get("revision")) is not int
        or baseline["revision"] < 1
        or type(baseline.get("reviewed", False)) is not bool
    ):
        raise ValueError("The source annotation has no compatible revision.")
    before = forehead_points(
        baseline["points"],
        baseline.get("forehead_visibility", "unassessed"),
        baseline.get("automatic_forehead_status"),
    )
    if not isinstance(anchor, str) or anchor not in before or before[anchor] is None:
        raise ValueError(
            "This source point is unavailable; choose an existing visible input."
        )
    original = make_preview(source, original=True)
    root = ET.fromstring(original.content)
    meta = json.loads(root.find("{*}metadata").text)
    width, height = meta["source_dimensions"]
    center = (before[anchor][0] * width, before[anchor][1] * height)
    delta = (x * width - center[0], y * height - center[1])
    radius_px = radius * min(width, height)
    if math.hypot(*delta) > 0.2 * radius_px:
        raise ValueError(
            "Move this point a smaller distance, or increase the influence radius. Maximum displacement is 20 percent of that radius."
        )
    after = {}
    for key, p in before.items():
        if p is None:
            after[key] = None
        else:
            weight = float(_weight(p[0] * width, p[1] * height, center, radius_px))
            after[key] = [
                p[0] + delta[0] * weight / width,
                p[1] + delta[1] * weight / height,
            ]
    after = validate_points(after)
    href = root.find("{*}image").attrib["href"]
    original_jpeg = base64.b64decode(href.split(",", 1)[1])
    image = cv2.imdecode(
        np.frombuffer(original_jpeg, np.uint8),
        cv2.IMREAD_COLOR,
    )
    if image is None:
        raise ValueError("The original display image could not be opened.")
    jpeg, residual = (
        (original_jpeg, 0.0)
        if delta == (0.0, 0.0)
        else _raster(image, [width, height], center, delta, radius_px)
    )
    h, w = image.shape[:2]
    canvas_width = max(480, w)
    offset = (canvas_width - w) / 2
    metadata = dict(
        kind="illustrative-preview",
        version=METHOD,
        source_sha256=original.source_sha256,
        source_dimensions=[width, height],
        image_display_dimensions=[w, h],
        viewport_padding=[offset, 0],
        asset="geometry",
        parameters=dict(anchor=anchor, x=x, y=y, radius=radius),
        forward_map="T(p)=p+d*max(0,1-||p-c||^2/R^2)^2",
        inverse_residual_pixels=residual,
        original_points=before,
        simulated_points=after,
        label="Illustrative only; not a predicted outcome or remeasurement.",
    )
    cx, cy = before[anchor][0] * w + offset, before[anchor][1] * h
    tx, ty = after[anchor][0] * w + offset, after[anchor][1] * h
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_width}" height="{h+60}" viewBox="0 0 {canvas_width} {h+60}"><metadata>{html.escape(json.dumps(metadata,sort_keys=True))}</metadata><image x="{offset}" width="{w}" height="{h}" href="data:image/jpeg;base64,{base64.b64encode(jpeg).decode()}"/><line x1="{cx}" y1="{cy}" x2="{tx}" y2="{ty}" stroke="#163e36" stroke-width="1"/><circle cx="{cx}" cy="{cy}" r="3" fill="#d89b24"/><circle cx="{tx}" cy="{ty}" r="3" fill="#16866b"/><rect y="{h}" width="{canvas_width}" height="60" fill="#f5f2e9"/><text x="12" y="{h+22}" font-size="12" fill="#163e36">Illustrative only · Geometry simulation · gold original / green simulated</text><text x="12" y="{h+44}" font-size="11" fill="#163e36">Not for remeasurement · {METHOD}</text></svg>'
    geometry = dict(
        method=METHOD,
        calculation_method=REGISTRY_VERSION,
        contract=measurement_contract(),
        dimensions=[width, height],
        original_points=before,
        simulated_points=after,
        source_review_revision=baseline["revision"],
        source_status=(
            "saved_user_reviewed_points"
            if baseline.get("reviewed")
            else "detector_proposed_points"
        ),
        forehead_visibility=baseline.get("forehead_visibility", "unassessed"),
        automatic_forehead_status=baseline.get("automatic_forehead_status"),
    )
    entry = seal_entry(
        dict(
            version=VERSION,
            kind="geometry",
            photo_number=photo_number,
            source_sha256=original.source_sha256,
            source_dimensions=[width, height],
            base_report_revision=basis,
            parameters=metadata["parameters"],
            geometry=geometry,
            original_svg=bind_svg(
                original.content, basis=basis, photo_number=photo_number
            ),
            simulation_svg=bind_svg(
                svg.encode(), basis=basis, photo_number=photo_number
            ),
            measurements=_readings(before, after, width, height),
            label="Illustrative geometry simulation only, not measured anatomy or a predicted outcome. All supported frontal point constructions are retained, including unavailable ones. Original detector/appearance readings, percentiles and scores are unchanged; the warped image is not reanalyzed.",
        )
    )
    return public_experiment({"entry": entry}, basis)["entry"]
