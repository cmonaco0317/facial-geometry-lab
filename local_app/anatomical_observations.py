"""Unranked iris observations and faithful names for the older mesh proxies."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .regional_evidence import metric_evidence

IRIS_METHOD = "automatic-iris-v1"
IRIS_LABELS = {
    "iris_spacing_ratio": "Detected iris-centre distance / outer-eye span",
    "iris_brow_elevation": "Detected iris-centre to brow elevation / outer-eye span",
}
PROXY_LABELS = {
    "facial_index": "Mesh forehead-to-chin height / cheek span",
    "upper_third": "Mesh upper segment share (not hairline-defined)",
    "middle_third": "Mesh middle segment share",
    "lower_third": "Mesh lower segment share",
    "forehead_ratio": "Mesh forehead width / cheek span",
    "jaw_ratio": "Lower-jaw landmark span / cheek span",
    "brow_position": "Eyelid-centre to brow elevation / mesh face height",
}
PROXY_NOTE = (
    "Fixed mesh landmarks approximate visible soft-tissue boundaries; they do not "
    "establish skeletal landmarks. Hair can hide the forehead or hairline while "
    "the model still predicts points there. Review visibility before interpreting "
    "forehead readings; hair covering skin is not the anatomical hairline."
)


def iris_metrics(raw: Any) -> dict[str, float]:
    """Use model iris centres directly; absent irises never become eyelid means."""
    points = np.asarray(raw, dtype=float)
    if points.shape != (478, 2):
        return {}
    needed = [33, 133, 362, 263, 159, 145, 386, 374, 468, 473]
    if not np.isfinite(points[needed]).all():
        return {}
    span = float(np.linalg.norm(points[33] - points[263]))
    if span <= 1e-8:
        return {}
    for centre, outer, inner, top, bottom in (
        (468, 33, 133, 159, 145), (473, 263, 362, 386, 374)
    ):
        xlow, xhigh = sorted(points[[outer, inner], 0])
        ylow, yhigh = sorted(points[[top, bottom], 1])
        x, y = points[centre]
        if xhigh - xlow <= 1e-8 or yhigh - ylow <= 1e-8:
            return {}
        if not (xlow <= x <= xhigh and ylow <= y <= yhigh):
            return {}
    result = {"iris_spacing_ratio": float(np.linalg.norm(points[468] - points[473]) / span)}
    brows = ([70, 63, 105, 66, 107, 55, 65, 52, 46],
             [300, 293, 334, 296, 336, 285, 295, 282, 276])
    if all(np.isfinite(points[ids]).all() for ids in brows):
        elevation = sum(points[i, 1] - min(points[ids, 1])
                        for i, ids in zip((468, 473), brows)) / 2
        result["iris_brow_elevation"] = float(elevation / span)
    return {key: value for key, value in result.items() if np.isfinite(value)}


def observed_iris_rows(
    context: Any, photo_ids: tuple[str, ...], paths: Mapping[str, Path | str], *,
    forehead_clear_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Additional direct readings, with the existing eye-region capture policy."""
    from fa_common import detect_landmarks

    values: dict[str, list[tuple[str, float]]] = {key: [] for key in IRIS_LABELS}
    for photo, _ in metric_evidence(context, photo_ids, "eye_spacing"):
        path = paths.get(photo["photo"])
        if path is None:
            continue
        try:
            points, _image, _dimensions = detect_landmarks(str(path))
            measured = iris_metrics(points) if points is not None else {}
        except (OSError, RuntimeError, ValueError):
            continue  # Optional iris evidence unavailable; never substitute the proxy.
        for key, value in measured.items():
            if (key == "iris_brow_elevation" and forehead_clear_ids is not None
                    and photo["photo"] not in forehead_clear_ids):
                continue
            values[key].append((photo["photo"], value))
    return [dict(
        id=key, label=IRIS_LABELS[key], value=round(float(np.median([v for _, v in readings])), 4),
        unit="ratio", method=IRIS_METHOD, percentile=None,
        evidence_photo_count=len(readings),
        evidence_photo_ids=[photo for photo, _ in readings],
        note=("Uses directly model-detected iris centres, not eyelid-ring averages. "
              "A separate detection of the same selected photographs uses the pinned "
              "model. These are image-space iris observations, not physical pupil "
              "measurements. No matched reference percentile or calibrated uncertainty "
              "is available; these readings do not change the composite."),
    ) for key, readings in values.items() if readings]
