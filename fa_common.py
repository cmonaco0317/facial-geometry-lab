"""Shared constants, landmark indices, and helpers for the facial harmony analysis.

All landmark indices follow the MediaPipe FaceMesh canonical topology
(478 points with refine_landmarks=True). Index assignments and ideal values
follow the analysis specification.
"""

import os

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task"
)

# ----------------------------------------------------------------------------
# Landmark indices (MediaPipe FaceMesh, refine_landmarks=True -> 478 points)
# ----------------------------------------------------------------------------
# Vertical anchors (thirds)
TRICHION = 10  # forehead / hairline
BROW_L = 55  # left brow head (medial)
BROW_R = 285  # right brow head (medial)
SUBNASALE = 2  # nose base / columella
MENTON = 152  # chin bottom

# Horizontal anchors (fifths / face width = zygion points)
ZYGION_L = 234
ZYGION_R = 454

# Eye canthi
L_CANTHUS_LAT = 33  # left lateral canthus
L_CANTHUS_MED = 133  # left medial canthus
R_CANTHUS_MED = 362  # right medial canthus
R_CANTHUS_LAT = 263  # right lateral canthus

# Eye vertical extents
L_EYE_TOP = 159
L_EYE_BOT = 145
R_EYE_TOP = 386
R_EYE_BOT = 374

# Iris centers (only present with refine_landmarks=True)
L_IRIS = 468
R_IRIS = 473

# Eye aperture rings (indices 0..467, so a mesh WITHOUT iris landmarks has them).
# Same index sets analyze_skin uses for its eye masks.
LEFT_EYE_RING = [
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
]  # fmt: skip
RIGHT_EYE_RING = [
    362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,
]  # fmt: skip

# Nose / mouth / lips
ALA_L = 129
ALA_R = 358
MOUTH_L = 61
MOUTH_R = 291
LABRALE_SUP = 0  # upper lip top (labrale superius)
LABRALE_INF = 17  # lower lip bottom (labrale inferius)

# Brow lateral
BROW_LAT_L = 46
BROW_LAT_R = 276

# Jaw
JAW_L = 172
JAW_R = 397

# Midline-defining landmarks
MIDLINE_IDS = [10, 4, 2, 152]

# Symmetry pairs: (left_index, right_index, label)
SYMMETRY_PAIRS = [
    (L_CANTHUS_MED, R_CANTHUS_MED, "Eye medial canthi"),
    (L_CANTHUS_LAT, R_CANTHUS_LAT, "Eye lateral canthi"),
    (BROW_L, BROW_R, "Brow medial"),
    (BROW_LAT_L, BROW_LAT_R, "Brow lateral"),
    (ALA_L, ALA_R, "Nose ala"),
    (MOUTH_L, MOUTH_R, "Mouth corners"),
    (ZYGION_L, ZYGION_R, "Zygion / cheek"),
    (JAW_L, JAW_R, "Jaw"),
]

# Landmarks that get an explicit labelled marker on the annotated image
KEY_LANDMARKS = {
    TRICHION: "10 trichion",
    BROW_L: "55",
    BROW_R: "285",
    SUBNASALE: "2 subnasale",
    MENTON: "152 menton",
    ZYGION_L: "234",
    ZYGION_R: "454",
    L_CANTHUS_LAT: "33",
    L_CANTHUS_MED: "133",
    R_CANTHUS_MED: "362",
    R_CANTHUS_LAT: "263",
    ALA_L: "129",
    ALA_R: "358",
    MOUTH_L: "61",
    MOUTH_R: "291",
    LABRALE_SUP: "0",
    LABRALE_INF: "17",
    L_IRIS: "468",
    R_IRIS: "473",
}

# Colors (BGR for OpenCV)
GREEN = (0, 180, 0)
ORANGE = (0, 140, 255)
BLUE = (255, 60, 0)
RED = (40, 40, 230)
YELLOW = (0, 215, 255)
WHITE = (255, 255, 255)
CYAN = (220, 200, 0)
TILT = (255, 255, 0)
MAGENTA = (200, 0, 200)


# ----------------------------------------------------------------------------
# Landmark detection
# ----------------------------------------------------------------------------
def detect_landmarks(img_path, min_conf=0.5, model_path=MODEL_PATH):
    """Run the FaceLandmarker (478-point attention mesh) at full resolution.

    Uses the MediaPipe Tasks FaceLandmarker, which is the current equivalent of
    the legacy ``solutions.face_mesh.FaceMesh`` with ``refine_landmarks=True``:
    the ``face_landmarker.task`` bundle returns the 468 face + 10 iris = 478
    attention-mesh landmarks. ``RunningMode.IMAGE`` == ``static_image_mode=True``
    and ``min_face_detection_confidence`` == ``min_detection_confidence``.

    Returns (pts, img_bgr, (w, h)) where pts is an (N, 2) float array of pixel
    coordinates, or (None, img_bgr, (w, h)) if no face was detected.
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)
    h, w = img.shape[:2]
    options = vision.FaceLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=min_conf,
        min_face_presence_confidence=min_conf,
    )
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        res = landmarker.detect(mp_image)
    if not res.face_landmarks:
        return None, img, (w, h)
    lm = res.face_landmarks[0]
    pts = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float64)
    return pts, img, (w, h)


# ----------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------
def dist(pts, a, b):
    """Euclidean distance between two landmarks."""
    return float(np.linalg.norm(pts[a] - pts[b]))


def vdist(pts, a, b):
    """Absolute vertical (y) distance between two landmarks."""
    return float(abs(pts[a][1] - pts[b][1]))


def hdist(pts, a, b):
    """Absolute horizontal (x) distance between two landmarks."""
    return float(abs(pts[a][0] - pts[b][0]))


def iris_centres_from_mesh(pts):
    """Estimate the two iris centres from the 468-point mesh alone.

    WHY THIS EXISTS -- A RULER DEFECT, MEASURED 2026-08-04 (ruler)
    -----------------------------------------------------
    Every asymmetry figure in this project is a percentage OF THE IPD, and the
    IPD comes from `pts[468] - pts[473]` -- landmarks that exist only on a
    photograph. A fitted multi-view identity is a 468-point mesh and has no iris,
    so `multiview_probe.published_symmetry` substituted the OUTER CANTHI (33,263)
    for them. That is a different length: measured on this project's own frames it
    is 1.402x the iris IPD inside the pose gate (range 1.309-1.605 across +-70deg).

    So the fitted readings were divided by a normaliser ~40% larger than the one
    the single-photo baseline used, making them ~29% smaller for free -- and the
    SD ratio that headlines the multi-view case inflated by the same factor. The
    function whose docstring warns that comparing different rulers "inflates any
    apparent improvement" was itself doing it, one level down.

    THE PROXY, AND WHY THIS ONE
    ---------------------------
    Candidates buildable from indices 0..467, scored against the detected iris
    centres on 10 real frames (9 burst + the report's own frontal):

        eye-aperture ring centroids  0.9863 x truth  (in-gate SD 0.0080, CV 0.81%)
        canthal midpoints            0.9524 x truth  (in-gate SD 0.0068, CV 0.71%)
        outer canthi (what was used) 1.4023 x truth  (in-gate SD 0.0082, CV 0.58%)

    All three are equally STABLE; only the ring centroids are already at the right
    LEVEL, so they need no conversion constant that could rot. The residual 1.37%
    shortfall biases the resulting percentage slightly HIGH -- the conservative
    direction for a figure this project publishes about a person.

    SCOPE: in-gate. Ungated the proxy degrades to CV 3.56% and drifts to 1.08x at
    -63deg, because the far eye foreshortens. The pose gates that already guard
    every asymmetry figure keep this inside the measured band.
    """
    pts = np.asarray(pts, dtype=float)
    return pts[LEFT_EYE_RING].mean(0), pts[RIGHT_EYE_RING].mean(0)


def mesh_ipd(pts):
    """IPD estimated from the 468-point mesh -- the ruler a fitted shape can use.

    Same quantity as `dist(pts, L_IRIS, R_IRIS)` to within 1.4% in-gate. Use this
    to normalise anything measured on a fitted identity, so the number is
    comparable to one measured on a photograph. See `iris_centres_from_mesh`.
    """
    left, right = iris_centres_from_mesh(pts)
    return float(np.linalg.norm(left - right))


def with_mesh_iris(pts):
    """A 478-row landmark array whose iris slots are filled from the mesh.

    For arrays that must satisfy the 478-point contract (`landmarks_frontal.csv`,
    anything handed to `analyze_advanced.symmetry`) but were produced without a
    photograph. Only the two CENTRES (468, 473) are meaningful; the remaining
    iris-ring slots are filled with the same centre so nothing reads as a stray
    point at the origin. No stage measures iris DIAMETER from the CSV -- the
    px/mm scale is fitted on the photograph in `fa_refine.iris_scale`.
    """
    pts = np.asarray(pts, dtype=float)
    out = np.zeros((478, 2), dtype=float)
    out[: min(len(pts), 468)] = pts[:468]
    left, right = iris_centres_from_mesh(pts)
    out[468:473] = left
    out[473:478] = right
    return out


def angle_3pt(a, b, c):
    """Interior angle at vertex b formed by points a-b-c, in degrees."""
    a, b, c = np.asarray(a, float), np.asarray(b, float), np.asarray(c, float)
    ba, bc = a - b, c - b
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))


# ----------------------------------------------------------------------------
# Drawing helpers (OpenCV has no native dashed lines)
# ----------------------------------------------------------------------------
def dashed_hline(img, y, x0, x1, color, thickness=2, dash=22, gap=16):
    y = int(round(y))
    x = int(x0)
    while x < x1:
        x_end = min(x + dash, int(x1))
        cv2.line(img, (x, y), (x_end, y), color, thickness, cv2.LINE_AA)
        x += dash + gap


def dashed_vline(img, x, y0, y1, color, thickness=2, dash=22, gap=16):
    x = int(round(x))
    y = int(y0)
    while y < y1:
        y_end = min(y + dash, int(y1))
        cv2.line(img, (x, y), (x, y_end), color, thickness, cv2.LINE_AA)
        y += dash + gap


def dotted_vline(img, x, y0, y1, color, radius=3, gap=20):
    x = int(round(x))
    for y in range(int(y0), int(y1), gap):
        cv2.circle(img, (x, y), radius, color, -1, cv2.LINE_AA)


def label(img, text, org, color=WHITE, scale=1.0, thick=2, bg=None):
    org = (int(org[0]), int(org[1]))
    if bg is not None:
        (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        cv2.rectangle(
            img, (org[0] - 3, org[1] - th - 5), (org[0] + tw + 3, org[1] + bl), bg, -1
        )
    cv2.putText(
        img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA
    )


# ----------------------------------------------------------------------------
# Verdict / grading helpers
# ----------------------------------------------------------------------------
def grade(dev_abs, green_max, yellow_max):
    """Return 'green' | 'yellow' | 'red' from an absolute deviation."""
    if dev_abs <= green_max:
        return "green"
    if dev_abs <= yellow_max:
        return "yellow"
    return "red"


# ---------------------------------------------------------------------------
# Stale-intermediate guard
# ---------------------------------------------------------------------------
# Every stage after the first reads landmarks_frontal.csv / metrics_frontal.json
# rather than the photograph. If frontal.png is replaced and stage 1 is not re-run,
# those files still describe the PREVIOUS photo and every downstream stage will
# happily produce a confident, wrong report -- with no error anywhere.
#
# This is not hypothetical: it happened during the 2026-08-01 skin-repeatability
# work. The photo was swapped for a test, restored afterwards, and the skin stage
# was then run against the test frame's landmarks. It reported a different acne
# grade and a different face resolution and looked entirely normal doing it.
#
# run_pipeline.py already halts on a stage that FAILS. This covers the other half:
# a stage that succeeds against the wrong inputs.
def image_fingerprint(path):
    """Content hash of an input image, stamped by stage 1 and checked downstream."""
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:32]


# ---------------------------------------------------------------------------
# The two landmark files, and why there are two
# ---------------------------------------------------------------------------
# Until the multi-view fit existed there was one landmark file and no ambiguity:
# `landmarks_frontal.csv` held the points detected on `frontal.png`, so measuring
# geometry and sampling pixels were the same coordinates.
#
# The set path breaks that. `landmarks_frontal.csv` becomes a POSE-FREE RENDER of
# a shape fitted across several photographs -- correct for geometry and wrong for
# pixels, because no photograph was ever taken from that angle. A stage that
# samples skin colour at those coordinates reads the wrong pixels of the right
# image, silently, with the staleness guard satisfied because the file is fresh.
#
#   GEOMETRY_CSV  what to MEASURE  -- fitted and pose-free in set mode
#   PHOTO_CSV     where to LOOK    -- always detected on frontal.png as it is
#
# In the two-photo path the two files are byte-identical, so the split costs
# nothing there and cannot drift.
GEOMETRY_CSV = "landmarks_frontal.csv"
PHOTO_CSV = "landmarks_photo.csv"


def load_landmark_csv(kind="geometry", here=None):
    """Load a 478x2 landmark array. `kind` is 'geometry' or 'photo'.

    Deliberately not defaulted per-caller: every call site states which of the two
    it means, because the wrong one produces plausible numbers rather than an error.
    """
    if kind not in ("geometry", "photo"):
        raise ValueError(f"kind must be 'geometry' or 'photo', got {kind!r}")
    here = here or os.path.dirname(os.path.abspath(__file__))
    name = GEOMETRY_CSV if kind == "geometry" else PHOTO_CSV
    path = os.path.join(here, name)
    if not os.path.exists(path):
        raise SystemExit(
            f"missing {name} -- run analyze_frontal.py (or run_pipeline.py) first. "
            "Not falling back to the other landmark file: they differ in the set "
            "path, and guessing would measure the wrong thing without an error."
        )
    pts = np.zeros((478, 2))
    with open(path) as fh:
        next(fh)
        for line in fh:
            i, x, y = line.split(",")
            pts[int(i)] = [float(x), float(y)]
    return pts


def photograph_pose(pose):
    """The pose of the PHOTOGRAPH, whatever the geometry was derived from.

    `metrics_frontal.json["pose"]` describes what was MEASURED. In the set path
    that is the fit, whose yaw is 0 by construction because pose was solved per
    frame and removed. Any statement about the photograph itself -- "this frame
    was +4.1 deg off axis", capture advice, or a gate gating a stage that read
    pixels -- must not inherit that 0, or the report tells a reader a casual snap
    was perfectly aligned. Stage 1 stores the frame's own pose alongside it.
    """
    pose = pose or {}
    return pose.get("photo_pose") or pose


def set_fingerprint(paths):
    """One content hash over a whole photo SET, order-independent.

    `assert_fresh` guards a stage against measuring the previous PHOTOGRAPH. The
    set path needs the same guard one level up: the fitted landmarks describe a
    directory of photographs, and swapping any file in it must invalidate them.
    Hashing the sorted per-file hashes means adding, removing or editing any photo
    changes the fingerprint, while re-ordering or re-listing does not.
    """
    import hashlib

    h = hashlib.sha256()
    for digest in sorted(image_fingerprint(p) for p in paths):
        h.update(digest.encode())
    return h.hexdigest()[:32]


def assert_fresh_set(fit_path, set_dir, stage=""):
    """Raise if the fitted landmarks were not derived from the set on disk now."""
    import glob
    import json

    if not (os.path.exists(fit_path) and os.path.isdir(set_dir)):
        return
    try:
        with open(fit_path) as f:
            stamped = json.load(f).get("_meta", {}).get("set_sha256")
    except (OSError, ValueError):
        return
    if stamped is None:
        return
    paths = sorted(
        p
        for ext in ("png", "jpg", "jpeg", "PNG", "JPG", "JPEG")
        for p in glob.glob(os.path.join(set_dir, f"*.{ext}"))
    )
    actual = set_fingerprint(paths)
    if stamped != actual:
        raise SystemExit(
            f"STALE FIT{' in ' + stage if stage else ''}: "
            f"{os.path.basename(fit_path)} was fitted from a different set of "
            f"photographs in {set_dir} (stamped {stamped[:12]}, on disk {actual[:12]}, "
            f"{len(paths)} files). Re-run fit_identity_stage.py -- or run_pipeline.py "
            "--set -- before this stage. Continuing would measure the previous set."
        )


def assert_fresh(metrics_path, image_path, stage=""):
    """Raise if metrics_frontal.json was not derived from the image on disk now.

    Silent when the stamp is absent, so a metrics file written before this guard
    existed does not break the pipeline -- absence of evidence is not staleness.
    """
    import json
    import os

    if not (os.path.exists(metrics_path) and os.path.exists(image_path)):
        return
    try:
        with open(metrics_path) as f:
            stamped = json.load(f).get("_meta", {}).get("source_sha256")
    except (OSError, ValueError):
        return
    if stamped is None:
        return
    actual = image_fingerprint(image_path)
    if stamped != actual:
        raise SystemExit(
            f"STALE INTERMEDIATES{' in ' + stage if stage else ''}: "
            f"{os.path.basename(metrics_path)} was derived from a different "
            f"{os.path.basename(image_path)} (stamped {stamped[:12]}, on disk "
            f"{actual[:12]}). Re-run analyze_frontal.py -- or run_pipeline.py -- "
            "before this stage. Continuing would measure the previous photograph."
        )
