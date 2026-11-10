"""Local hair/skin overlap checks; a fringe boundary is never an anatomical hairline."""

from __future__ import annotations

import hashlib
import html
from math import isfinite
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

METHOD = "automatic-forehead-v1"
MODEL_NAME = "forehead_context.bin"
LABELS = ("background", "hair", "body-skin", "face-skin", "clothes", "others")
FOREHEAD_METRICS = frozenset({"facial_index", "upper_third", "middle_third",
                              "lower_third", "forehead_ratio", "brow_position",
                              "iris_brow_elevation"})
PIXEL_CONFIDENCE = .65
HAIR_OVERLAP_FRACTION = .08
MAX_CLEAR_HAIR_FRACTION = .02
MIN_CLEAR_SKIN_FRACTION = .80
MIN_OUTER_EYE_PIXELS = 40.
MAX_INPUT_SIDE = 512
EXPLANATIONS = {
    "hair_overlap": "Hair overlaps the assessed forehead band. Dependent forehead readings need a clearer photo; the fringe edge is not the anatomical hairline.",
    "forehead_clear": "Skin is visible in the assessed forehead band. This does not establish the anatomical hairline or validate every landmark.",
    "uncertain": "Forehead visibility is uncertain. The image does not support treating hidden forehead points as observed skin boundaries.",
    "unavailable": "Automatic hair segmentation is unavailable. Other face measurements can continue; forehead visibility has not been established.",
    "no_face": "No face was detected in this photo. Add a clear photo of one person.",
}
REASONS = {
    "hair_overlap": "Detected hair in the forehead band",
    "skin_visible": "Skin visible in the forehead band",
    "uncertain_segmentation": "Mixed or uncertain segmentation",
    "invalid_segmentation": "Segmentation output could not be interpreted",
    "insufficient_detail": "Not enough original image detail",
    "geometry_unavailable": "Forehead geometry is unavailable or cropped",
    "pose": "Use a more frontal, level photo for the forehead",
    "model_unavailable": "The local segmentation model is unavailable",
    "image_unavailable": "This photo could not be segmented",
    "no_face": "No detected face",
}


def _unknown(reason: str) -> dict[str, Any]:
    return dict(status="uncertain", reason=reason, method=METHOD)


def forehead_polygon(points: Any) -> np.ndarray | None:
    """A head-oriented band above the brows, excluding the brow-hair boundary."""
    p = np.asarray(points, dtype=float)
    ids = [10, 54, 284, 70, 300, 33, 263]
    if p.shape != (468, 2) or not np.isfinite(p[ids]).all():
        return None
    if np.linalg.norm(p[33] - p[263]) < MIN_OUTER_EYE_PIXELS:
        return None
    top = p[10]
    return np.array([top, .75 * p[284] + .25 * top,
                     .85 * p[300] + .15 * top,
                     .85 * p[70] + .15 * top,
                     .75 * p[54] + .25 * top])


def classify_forehead(confidences: Any, points: Any) -> dict[str, Any]:
    """Thresholds are transparent engineering heuristics, not calibrated accuracy."""
    values = np.asarray(confidences, dtype=float)
    if values.ndim == 4 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.ndim != 3 or values.shape[0] != len(LABELS):
        return _unknown("invalid_segmentation")
    polygon = forehead_polygon(points)
    if polygon is None:
        return _unknown("insufficient_detail")
    height, width = values.shape[1:]
    if (np.any(polygon < 0) or np.any(polygon[:, 0] > width - 1)
            or np.any(polygon[:, 1] > height - 1)):
        return _unknown("geometry_unavailable")
    selected = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(selected, cv2.convexHull(np.rint(polygon).astype(np.int32)), 1)
    area = selected.astype(bool)
    if int(area.sum()) < 64:
        return _unknown("insufficient_detail")
    pixels = values[:, area]
    if (not np.isfinite(pixels).all() or np.any(pixels < 0)
            or np.any(pixels > 1) or not np.allclose(pixels.sum(axis=0), 1, atol=.01)):
        return _unknown("invalid_segmentation")
    hair = float(np.mean(pixels[1] >= PIXEL_CONFIDENCE))
    skin = float(np.mean(pixels[2] + pixels[3] >= PIXEL_CONFIDENCE))
    status, reason = "uncertain", "uncertain_segmentation"
    if hair >= HAIR_OVERLAP_FRACTION:
        status, reason = "hair_overlap", "hair_overlap"
    elif hair <= MAX_CLEAR_HAIR_FRACTION and skin >= MIN_CLEAR_SKIN_FRACTION:
        status, reason = "forehead_clear", "skin_visible"
    return dict(status=status, reason=reason, method=METHOD,
                hair_fraction=round(hair, 4), skin_fraction=round(skin, 4))


class ForeheadSegmenter:
    """Own one pinned local model per analysis, with bounded image/mask size."""

    def __enter__(self) -> ForeheadSegmenter:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        from macos_app.config import MODEL_ASSETS

        model = Path(__file__).resolve().parent.parent / MODEL_NAME
        if model.is_symlink() or not model.is_file():
            raise ValueError("The segmentation model is missing.")
        if hashlib.sha256(model.read_bytes()).hexdigest() != MODEL_ASSETS[MODEL_NAME].sha256:
            raise ValueError("The segmentation model checksum differs.")
        self._mp = mp
        self._model = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
            base_options=python.BaseOptions(model_asset_path=str(model)),
            running_mode=vision.RunningMode.IMAGE,
            output_category_mask=False, output_confidence_masks=True,
        ))
        if tuple(self._model.labels) != LABELS:
            self._model.close()
            raise ValueError("The segmentation labels differ.")
        return self

    def __exit__(self, *_args: Any) -> None:
        self._model.close()

    def observe(self, path: Path, points: np.ndarray) -> dict[str, Any]:
        image = cv2.imread(str(path))
        if image is None:
            return _unknown("image_unavailable")
        p = np.asarray(points, dtype=float)
        if p.shape != (468, 2) or not np.isfinite(p).all():
            return _unknown("geometry_unavailable")
        # Do not turn an enlarged tiny face into apparently sufficient detail.
        if np.linalg.norm(p[33] - p[263]) < MIN_OUTER_EYE_PIXELS:
            return _unknown("insufficient_detail")
        height, width = image.shape[:2]
        low, high = p.min(axis=0), p.max(axis=0)
        span = high - low
        x0, y0 = np.floor(np.maximum(0, low - span * (.25, .45))).astype(int)
        x1, y1 = np.ceil(np.minimum((width, height), high + span * (.25, .15))).astype(int)
        if x1 <= x0 or y1 <= y0:
            return _unknown("geometry_unavailable")
        crop = image[y0:y1, x0:x1]
        scale = min(1., MAX_INPUT_SIDE / max(crop.shape[:2]))
        if scale < 1:
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        transformed = (p - (x0, y0)) * scale
        result = self._model.segment(self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
        ))
        masks = np.stack([mask.numpy_view() for mask in result.confidence_masks])
        return classify_forehead(masks, transformed)


def observe_photos(photos: Sequence[Any], records: Sequence[dict]) -> list[dict]:
    by_id = {record["photo"]: record for record in records}
    observations = []
    detected = [photo for photo in photos if photo.entry.photo_id in by_id]
    results: dict[str, dict] = {}
    if detected:
        try:
            with ForeheadSegmenter() as segmenter:
                for photo in detected:
                    record = by_id[photo.entry.photo_id]
                    if any(not isfinite(float(record.get(k, float("nan"))))
                           or abs(float(record[k])) > 25 for k in ("yaw", "pitch", "roll")):
                        results[photo.entry.photo_id] = _unknown("pose")
                        continue
                    try:
                        results[photo.entry.photo_id] = segmenter.observe(photo.path, record["landmarks"])
                    except (OSError, RuntimeError, ValueError):
                        results[photo.entry.photo_id] = _unknown("image_unavailable")
        except (OSError, RuntimeError, ValueError, KeyError):
            results = {photo.entry.photo_id: dict(status="unavailable", reason="model_unavailable", method=METHOD)
                       for photo in detected}
    for ordinal, photo in enumerate(photos, 1):
        result = results.get(photo.entry.photo_id, dict(status="no_face", reason="no_face", method=METHOD))
        observations.append({**result, "photo_id": photo.entry.photo_id, "photo_number": ordinal})
    return observations


def coverage_exclusions(axes: Mapping[str, Any], frontal: list[dict] | None = None) -> set[str]:
    harmony = axes.get("harmony")
    observations = harmony.details.get("automatic_forehead", []) if harmony else []
    observations = observations if isinstance(observations, list) else []
    covered = {r["photo_id"] for r in observations if isinstance(r, dict)
               and isinstance(r.get("photo_id"), str) and isinstance(r.get("status"), str)
               and r["status"] in EXPLANATIONS and r["status"] != "forehead_clear"}
    overrides = harmony.details.get("forehead_overrides", {}) if harmony else {}
    for photo, status in overrides.items() if isinstance(overrides, dict) else []:
        if not isinstance(photo, str) or not isinstance(status, str):
            continue
        if status in {"forehead_hidden", "uncertain"}:
            covered.add(photo)
        elif status in {"visible", "hairline_hidden"}:
            covered.discard(photo)
    for review in frontal or []:
        photo = review.get("evidence_photo_id")
        if not isinstance(photo, str):
            continue
        status = review.get("forehead_visibility")
        if status in {"forehead_hidden", "uncertain"}:
            covered.add(photo)
        elif status in {"visible", "hairline_hidden"} and not review.get("visibility_only"):
            covered.discard(photo)
    return covered


def public_forehead(axes: Mapping[str, Any]) -> list[dict]:
    harmony = axes.get("harmony")
    raw = harmony.details.get("automatic_forehead", []) if harmony else []
    overrides = harmony.details.get("forehead_overrides", {}) if harmony else {}
    rows = []
    for item in raw[:10000] if isinstance(raw, list) else []:
        if (not isinstance(item, dict) or not isinstance(item.get("status"), str)
                or item["status"] not in EXPLANATIONS
                or type(item.get("photo_number")) is not int or not 1 <= item["photo_number"] <= 10000):
            continue
        row = dict(photo_number=item["photo_number"], status=item["status"], method=METHOD,
                   explanation=EXPLANATIONS[item["status"]],
                   reason=REASONS.get(item.get("reason") if isinstance(item.get("reason"), str)
                                      else "", "Automatic forehead assessment"))
        override = (overrides.get(item["photo_id"]) if isinstance(overrides, dict)
                    and isinstance(item.get("photo_id"), str) else None)
        if isinstance(override, str) and override in {"visible", "hairline_hidden", "forehead_hidden", "uncertain"}:
            row["manual_override"] = override
            row["explanation"] += " A confirmed user visibility correction is applied to regional photo selection."
        for key in ("hair_fraction", "skin_fraction"):
            value = item.get(key)
            if type(value) in (int, float) and isfinite(value) and 0 <= value <= 1:
                row[key] = value
        rows.append(row)
    return rows


def render_automatic_forehead(rows: list[dict]) -> str:
    if not rows:
        return ""
    body = "".join(
        f'<tr><th>Photo {r["photo_number"]}</th><td>{html.escape(r["reason"])}</td>'
        f'<td>{format(100 * r["hair_fraction"], ".1f") + "%" if "hair_fraction" in r else "Unavailable"}</td>'
        f'<td>{html.escape(r["explanation"])}</td></tr>' for r in rows
    )
    return ('<section class="feature-chapter"><h2>Automatic hair and forehead check</h2>'
            '<p>Local hair/skin segmentation checks each photo automatically. Percentages '
            'describe classified pixels in a forehead band, not measurement accuracy. '
            'Clear skin does not prove where an anatomical hairline lies behind a fringe. '
            'Uncertain forehead readings use other clear photos when available; other '
            'regions continue. The experimental geometry score still uses its original '
            'mesh instrument and is not a hairline-corrected score. Optional landmark '
            'review can correct a mistaken visibility assessment.</p>'
            '<div class="table-wrap"><table><thead><tr><th>Photo</th><th>Automatic finding</th>'
            '<th>Confident hair pixels</th><th>Meaning / next photo</th></tr></thead>'
            f'<tbody>{body}</tbody></table></div></section>')
