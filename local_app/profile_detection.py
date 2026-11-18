"""Normalize inference orientation without changing stored images or frontal inputs."""

from __future__ import annotations

import math
from numbers import Real

from .profile_proposals import HEURISTICS


def profile_detector_inputs(path, record, width, height):
    pose = {key: record.get(key) for key in ("yaw", "pitch", "roll")}
    yaw = pose["yaw"]
    try:
        numeric_yaw = (
            float(yaw) if isinstance(yaw, Real) and not isinstance(yaw, bool) else None
        )
    except (OverflowError, ValueError):
        numeric_yaw = None
    metadata = dict(
        mirrored_for_inference=False,
        canonical_attempted=False,
        coordinate_mapping="original normalized image",
        normalization_reason="retained original detector output",
    )
    if (
        numeric_yaw is None
        or not math.isfinite(numeric_yaw)
        or numeric_yaw >= HEURISTICS["canonical_left_trigger_yaw_degrees"]
    ):
        return record.get("landmarks"), pose, metadata
    import cv2
    import numpy as np
    import analyze_averageness2 as detector

    image = cv2.imread(str(path))
    if image is None or image.shape[:2] != (height, width):
        return (
            None,
            pose,
            {
                **metadata,
                "normalization_reason": "original image unavailable or dimensions changed",
            },
        )
    try:
        metadata["canonical_attempted"] = True
        with detector.make_landmarker() as landmarker:
            metadata["mirrored_for_inference"] = True
            result = landmarker.detect(
                detector.mp.Image(
                    image_format=detector.mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB),
                )
            )
        if not result.face_landmarks or not result.facial_transformation_matrixes:
            return (
                None,
                pose,
                {
                    **metadata,
                    "normalization_reason": "canonical detector found no compatible face",
                },
            )
        points = np.array(
            [
                [(1 - point.x) * width, point.y * height]
                for point in result.face_landmarks[0][:468]
            ],
            dtype=np.float64,
        )
        pitch, canonical_yaw, roll = (
            float(v)
            for v in detector.pose_angles(result.facial_transformation_matrixes[0])
        )
        return (
            points,
            dict(pitch=pitch, yaw=-canonical_yaw, roll=-roll),
            dict(
                mirrored_for_inference=True,
                canonical_attempted=True,
                coordinate_mapping="x_original = (1 - x_canonical) * width; y unchanged",
                normalization_reason="face-right inference mapped back to original frame",
            ),
        )
    except (OSError, RuntimeError, ValueError):
        return (
            None,
            pose,
            {
                **metadata,
                "normalization_reason": "canonical profile inference unavailable",
            },
        )
