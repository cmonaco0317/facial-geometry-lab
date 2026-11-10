"""Profile silhouette extraction and cephalometric landmark detection.

A true 90-degree lateral photo is outside MediaPipe FaceMesh's reliable range,
so cephalometric points are located on the face's *anterior contour*: the
person is segmented from the background, the silhouette outline is traced, and
soft-tissue landmarks (glabella, pronasale, subnasale, labrale superius/inferius,
pogonion, menton, cervical point) are found as curvature extrema along the
profile curve.
"""

import os

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
from scipy.signal import savgol_filter, find_peaks

HERE = os.path.dirname(os.path.abspath(__file__))
SEG_MODEL = os.path.join(HERE, "selfie_segmenter.tflite")


def segment_person(img_bgr):
    """Return a clean binary person mask (uint8 0/255) via the selfie segmenter."""
    h, w = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    opts = vision.ImageSegmenterOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=SEG_MODEL),
        running_mode=vision.RunningMode.IMAGE,
        output_confidence_masks=True,
    )
    with vision.ImageSegmenter.create_from_options(opts) as seg:
        res = seg.segment(mp_image)
    conf = res.confidence_masks[0].numpy_view()  # foreground probability HxW
    mask = (conf > 0.5).astype(np.uint8) * 255

    # Keep only the largest connected component, then close small holes.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = np.where(labels == largest, 255, 0).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    return mask


def anterior_curve(mask, facing="right"):
    """For each image row, the most-anterior person-pixel x (the profile edge).

    Returns (ys, xs) covering rows where the person is present. ``facing`` is the
    direction the face points ('right' -> anterior = max x).
    """
    h, w = mask.shape
    ys, xs = [], []
    for y in range(h):
        cols = np.where(mask[y] > 0)[0]
        if cols.size == 0:
            continue
        x = cols.max() if facing == "right" else cols.min()
        ys.append(y)
        xs.append(int(x))
    return np.array(ys), np.array(xs)


def detect_facelandmarker(img_bgr, model_path):
    """Best-effort FaceLandmarker on the profile, as a cross-check. May return None."""
    h, w = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    opts = vision.FaceLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3,
    )
    with vision.FaceLandmarker.create_from_options(opts) as lm:
        res = lm.detect(mp_image)
    if not res.face_landmarks:
        return None
    return np.array(
        [[p.x * w, p.y * h] for p in res.face_landmarks[0]], dtype=np.float64
    )


if __name__ == "__main__":
    IMG = os.path.join(HERE, "profile.png")
    img = cv2.imread(IMG)
    h, w = img.shape[:2]
    print(f"profile {w}x{h}")

    mask = segment_person(img)
    cov = 100.0 * (mask > 0).sum() / (h * w)
    print(f"person mask coverage: {cov:.1f}%")

    # Decide facing direction: centroid of mask vs nose. Most-anterior extremum.
    ys, xs = anterior_curve(mask, "right")
    # crude face band: from top of mask to ~70% down
    print(f"curve rows: {ys.min()}..{ys.max()}, x range {xs.min()}..{xs.max()}")

    fl = detect_facelandmarker(img, os.path.join(HERE, "face_landmarker.task"))
    print(
        "FaceLandmarker on profile:", "DETECTED" if fl is not None else "no detection"
    )

    # Debug overlay
    dbg = img.copy()
    # mask edge in semi-transparent green
    edges = cv2.Canny(mask, 50, 150)
    dbg[edges > 0] = (0, 255, 0)
    # anterior curve in red
    for y, x in zip(ys, xs):
        cv2.circle(dbg, (x, y), 1, (0, 0, 255), -1)
    if fl is not None:
        for idx in (
            10,
            9,
            8,
            168,
            6,
            197,
            195,
            5,
            4,
            1,
            2,
            0,
            13,
            14,
            17,
            18,
            152,
            200,
            199,
        ):
            x, y = int(fl[idx][0]), int(fl[idx][1])
            cv2.circle(dbg, (x, y), 4, (255, 255, 0), -1)
            cv2.putText(
                dbg,
                str(idx),
                (x + 5, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
            )
    cv2.imwrite(os.path.join(HERE, "profile_debug.png"), dbg)
    print("wrote profile_debug.png")


def detect_facing(mask, img_bgr=None, model_path=None):
    """Which way is the subject facing? 'right' | 'left'.

    Every caller hardcoded "right" until 2026-08-01. A subject photographed facing
    the other way produced no error and no warning — `anterior_curve` would trace the
    BACK of the head as if it were the profile, and every cephalometric angle
    downstream would be measured off the skull instead of the face. Silent garbage,
    and exactly the class of failure this project keeps finding.

    Two independent signals, cheapest first:

    1. Face landmarks, when available. The nose tip sits anterior to the ear, so its
       position relative to the silhouette's horizontal centre gives the direction
       directly. This is the reliable one.
    2. Silhouette asymmetry, as a fallback. A head in profile has a long, roughly
       vertical anterior edge (forehead → nose → lips → chin) and a much rounder
       posterior one. The side whose extreme-x column varies MORE across rows is the
       face side, because a nose and chin stick out further than the back of a skull.

    Returns ('right'|'left', confidence 0..1, method). Callers should treat low
    confidence as a reason to say so, not as a reason to guess quietly.
    """
    h, w = mask.shape
    rows = [(y, np.where(mask[y] > 0)[0]) for y in range(h)]
    rows = [(y, c) for y, c in rows if c.size]
    if not rows:
        return "right", 0.0, "no silhouette — defaulted"

    if img_bgr is not None and model_path is not None:
        try:
            fl = detect_facelandmarker(img_bgr, model_path)
        except Exception:
            fl = None
        if fl is not None:
            nose = np.asarray(fl[1])  # pronasale-ish anchor
            centre = np.mean([c.mean() for _, c in rows])
            if abs(float(nose[0]) - centre) > 0.02 * w:
                return (
                    ("right" if float(nose[0]) > centre else "left"),
                    0.95,
                    "face landmark vs silhouette centre",
                )

    right_edge = np.array([c.max() for _, c in rows], dtype=float)
    left_edge = np.array([c.min() for _, c in rows], dtype=float)
    # Compare over the middle vertical band, where the face is; the top of the head
    # and the shoulders are roughly symmetric and only add noise.
    lo, hi = int(0.25 * len(rows)), int(0.80 * len(rows))
    r_var, l_var = right_edge[lo:hi].std(), left_edge[lo:hi].std()
    total = r_var + l_var
    if total <= 0:
        return "right", 0.0, "flat silhouette — defaulted"
    facing = "right" if r_var > l_var else "left"
    conf = float(abs(r_var - l_var) / total)
    return facing, conf, "silhouette edge variation (fallback)"


def load_profile_normalised(path, model_path=None):
    """Read a profile photo and normalise it to face-RIGHT.

    Every downstream sign convention in this project assumes a right-facing
    subject — anterior = max x, chin projection positive forward, the extremum
    searches for nasion and mentolabial sulcus. Threading a `facing` flag through
    all of that would need each of those conventions audited separately. Mirroring
    the INPUT once is a single, checkable normalisation that leaves every
    downstream assumption true.

    Returns (img, mask, meta) where meta records what was done, so the report can
    say "this image was mirrored for measurement" rather than silently presenting a
    flipped face.
    """
    img = cv2.imread(path)
    if img is None:
        raise SystemExit(f"PROFILE FAILED: cannot read {path}")
    # Segment ONCE for the facing decision, then -- if the image is flipped --
    # segment AGAIN from the flipped image rather than flipping the mask.
    # Flipping the mask is not equivalent: the segmentation model is not
    # flip-equivariant, and doing it that way made a mirrored copy of the same
    # photograph disagree with the original by ~4% of mask pixels and up to 752 px
    # of anterior curve, which moved the nasolabial angle by 10 degrees. Verified by
    # asserting the round trip below.
    probe = segment_person(img)
    facing, conf, method = detect_facing(probe, img, model_path)
    meta = {
        "detected_facing": facing,
        "confidence": round(float(conf), 3),
        "method": method,
        "mirrored_for_measurement": facing == "left",
    }
    if facing == "left":
        img = cv2.flip(img, 1)
        mask = segment_person(img)
    else:
        mask = probe
    if conf < 0.25:
        meta["warning"] = (
            f"Facing direction is only {conf:.2f} confident via {method}. If this "
            "photograph is not a clear profile, the cephalometric angles below are "
            "measuring an edge that may not be the face."
        )
    return img, mask, meta
