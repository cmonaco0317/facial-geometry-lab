"""Head pose and 3D landmarks for one photograph. The part of `frontalize.py`
that was worth keeping.

WHY THIS MODULE EXISTS UNDER A NEW NAME
---------------------------------------
`frontalize.py` was a pipeline STAGE: it counter-rotated MediaPipe's 3D landmark
cloud to a canonical-frontal orientation, re-measured thirds/fifths/symmetry, and
reported the before/after delta as "how much of the measurement was pose
artifact". Phase D retired it, for the reason its own report section already
admitted in print:

    "No metric in this report is pose-corrected; frontalisation is run as a
     diagnostic only."

A stage whose output nothing consumes is not a measurement, it is a number that
has to be explained away — and explaining it cost the report a long paragraph
about why its crude 8-pair symmetry figure was NOT comparable to the 42-pair
figure in Part C1, after an earlier version had wrongly claimed the two agreed.
The multi-view fit solves pose as a nuisance parameter and removes it properly,
which is what frontalisation was gesturing at and could not deliver from one
photograph.

WHAT SURVIVED, AND WHY IT HAD TO
--------------------------------
Eight modules never wanted frontalisation at all — they wanted `detect_3d` and
`euler`, i.e. "what pose is this photograph, measured rather than assumed". That
is the pose gate's input, the intake set's diversity signal, and the axis every
pose-sensitivity result in this project is plotted against. It lives here, named
for what it does, so nothing imports a module called `frontalize` to do something
that is not frontalisation.

⚠️ THE POSE IS MEDIAPIPE'S, and it rides the same learned monocular-depth prior
as the landmarks themselves. It is not an independent measurement of head
orientation, and two procedures that share a prior agreeing is consistency, not
corroboration. The one place this was checked against something else — the
constant-face burst — put the landmark yaw proxy at r = -0.996 against it, which
says the two agree, not that either is right.
"""

import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def detect_3d(path):
    """3D landmark cloud + head-pose matrix for one image.

    Returns (P (478,3), M (4,4), (w, h), img). P's z uses roughly the same
    normalised scale as x, so it is multiplied by the image WIDTH -- not the
    height, which would silently anisotropically scale depth on a non-square
    frame.

    mediapipe is imported lazily so that importing this module stays cheap for
    callers that only want `euler`, and so a module that merely mentions pose
    does not drag the whole vision stack into a synthetic-only test.
    """
    import mediapipe as mp
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision

    opts = vision.FaceLandmarkerOptions(
        base_options=mpt.BaseOptions(
            model_asset_path=os.path.join(HERE, "face_landmarker.task")
        ),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        output_facial_transformation_matrixes=True,
    )
    img = cv2.imread(path)
    h, w = img.shape[:2]
    with vision.FaceLandmarker.create_from_options(opts) as lm:
        res = lm.detect(
            mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
            )
        )
    lms = res.face_landmarks[0]
    P = np.array([[p.x * w, p.y * h, p.z * w] for p in lms], dtype=np.float64)
    M = np.array(res.facial_transformation_matrixes[0])
    return P, M, (w, h), img


def euler(R):
    """(pitch, yaw, roll) in degrees from a rotation matrix."""
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    return (
        np.degrees(np.arctan2(R[2, 1], R[2, 2])),
        np.degrees(np.arctan2(-R[2, 0], sy)),
        np.degrees(np.arctan2(R[1, 0], R[0, 0])),
    )
