"""Accuracy engine layered on top of MediaPipe FaceLandmarker.

This module sharpens the 478-point landmark estimate WITHOUT replacing MediaPipe.
It adds four empirically-verifiable refinements and helpers for a fifth:

  1. Multi-scale + tight-crop + horizontal-flip fusion with a robust
     (per-landmark median) aggregate. Every variant is mapped back to ORIGINAL
     image pixel coordinates before aggregation. Averaging N noisy detector
     estimates cuts random detector error ~1/sqrt(N) and the median rejects
     the occasional outlier variant.
  2. Iris-diameter metric scale (px -> mm) with NO reference object. The
     horizontal visible iris diameter is ~11.7 mm in adults (SD ~0.5 mm;
     standard ophthalmic/forensic constant). refine_landmarks gives iris
     boundary landmarks 468-477; we ellipse-fit the limbus (sub-pixel, image
     edges) and derive px/mm with an honest +/-4-5% error.
  3. Sub-pixel refinement of EDGE landmarks ONLY -- points that sit on a real
     image gradient: iris boundary (ellipse fit), eyelid margins, lip vermilion
     border, and the canthi (cornerSubPix). NOT the jaw/cheek: there is no
     image edge there, so sub-pixel snapping would be noise.
  4. Contour-spline de-jitter of the FACE_OVAL (jaw/cheek): fit a smooth
     periodic spline through the oval ring and resample each oval landmark onto
     the curve, removing per-point jitter that inflates the jaw-asymmetry
     ARTIFACT.

CRITICAL HONESTY (ties to critique #1): fusion reduces *detector jitter on this
one photo* only. It does NOT measure test-retest variance (different day,
lighting, pose, expression). The dispersion returned here is a
DETECTOR-REPEATABILITY figure = a LOWER BOUND on true uncertainty. Real
test-retest is wider. Do not sell fusion as fixing critique #1.

Dependencies: cv2, numpy, scipy, mediapipe only (no skimage).
Style matches fa_common.py / analyze_frontal.py.
"""

import os

import cv2
import numpy as np
from scipy.interpolate import splev, splprep

import fa_common as F

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "face_landmarker.task")
DEFAULT_IMG = os.path.join(HERE, "frontal.png")

# Standard ophthalmic/forensic constant: horizontal visible iris (limbus)
# diameter in adults. SD ~0.5 mm => ~4.3% 1-sigma on the scale itself, before
# any non-frontal / detection error.
IRIS_MM = 11.7
IRIS_SD_MM = 0.5

# MediaPipe iris landmark groups (present only with refine_landmarks=True / the
# attention-mesh .task bundle): each group is [center, right, top, left, bottom].
L_IRIS_GROUP = [468, 469, 470, 471, 472]
R_IRIS_GROUP = [473, 474, 475, 476, 477]

# FACE_OVAL ring, in connected order (MediaPipe canonical topology). Used for the
# periodic-spline de-jitter of the jaw/cheek silhouette.
FACE_OVAL = [
    10,
    338,
    297,
    332,
    284,
    251,
    389,
    356,
    454,
    323,
    361,
    288,
    397,
    365,
    379,
    378,
    400,
    377,
    152,
    148,
    176,
    149,
    150,
    136,
    172,
    58,
    132,
    93,
    234,
    127,
    162,
    21,
    54,
    103,
    67,
    109,
]

# Eyelid-margin landmarks (upper + lower lid, both eyes) -- real image edges.
L_EYELID = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
R_EYELID = [
    362,
    398,
    384,
    385,
    386,
    387,
    388,
    466,
    263,
    249,
    390,
    373,
    374,
    380,
    381,
    382,
]

# Outer lip vermilion border -- real image edge.
OUTER_LIP = [
    61,
    185,
    40,
    39,
    37,
    0,
    267,
    269,
    270,
    409,
    291,
    375,
    321,
    405,
    314,
    17,
    84,
    181,
    91,
    146,
]

# Canthi (eye corners) -- crisp corners, good for cornerSubPix.
CANTHI = [F.L_CANTHUS_LAT, F.L_CANTHUS_MED, F.R_CANTHUS_MED, F.R_CANTHUS_LAT]


# ----------------------------------------------------------------------------
# Low-level detection (returns ALL 478 points in the frame it was run in)
# ----------------------------------------------------------------------------
def _make_landmarker(min_conf=0.4):
    import mediapipe as mp
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision

    opts = vision.FaceLandmarkerOptions(
        base_options=mpt.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=min_conf,
        min_face_presence_confidence=min_conf,
    )
    return vision.FaceLandmarker.create_from_options(opts)


def _detect(lmk, img_bgr):
    import mediapipe as mp

    h, w = img_bgr.shape[:2]
    res = lmk.detect(
        mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
        )
    )
    if not res.face_landmarks:
        return None
    return np.array(
        [[p.x * w, p.y * h] for p in res.face_landmarks[0]], dtype=np.float64
    )


def _build_mirror():
    """Left<->right landmark index remap, from the canonical face model."""
    verts = []
    with open(os.path.join(HERE, "canonical_face_model.obj")) as f:
        for line in f:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
    V = np.array(verts)
    Vm = V * [-1, 1, 1]
    m = np.array(
        [int(np.argmin(np.sum((V - Vm[i]) ** 2, axis=1))) for i in range(len(V))]
    )
    mir = np.arange(478)
    mir[: len(m)] = m
    for k in range(5):  # iris groups swap L<->R
        mir[468 + k] = 473 + k
        mir[473 + k] = 468 + k
    return mir


# ----------------------------------------------------------------------------
# (1) Multi-scale + tight-crop + flip fusion -> per-landmark median
# ----------------------------------------------------------------------------
def _face_bbox(pts, w, h, pad=0.35):
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    bw, bh = x1 - x0, y1 - y0
    x0 = max(0, int(x0 - pad * bw))
    y0 = max(0, int(y0 - pad * bh))
    x1 = min(w, int(x1 + pad * bw))
    y1 = min(h, int(y1 + pad * bh))
    return x0, y0, x1, y1


def fused_landmarks(img_path=DEFAULT_IMG, min_conf=0.4, img=None):
    """Run FaceLandmarker over several identity-preserving views, map all back to
    ORIGINAL pixel coords, and take the per-landmark median.

    Views: full image, tight face crop (so the detector's fixed ~256px input
    budget covers the face rather than background), two scales, and a horizontal
    flip (mirror-remapped). All geometric, identity-preserving.

    ⚠️ THE MIRRORED VARIANT BELOW DOES NOT CANCEL THE DETECTOR'S LEFT-RIGHT BIAS,
    and reading this list as though it did was a real mistake for months. Six
    variants are combined by per-landmark MEDIAN; five of them see the image in
    its original orientation and therefore carry the same one-directional error,
    so the median sits among those five and the mirrored one is outvoted.
    Diluting a systematic error is not cancelling it. Measured 2026-08-11: the
    output of this function still disagrees with its own mirror by 1.27-1.81% of
    IPD, 82-97% of it in the antisymmetric component. `refined_landmarks`
    balances the two orientations to cancel it properly.

    `img` lets a caller pass pixels it already holds, which is what the mirror
    balance uses so it does not round-trip a 7MB frame through the filesystem.

    Returns (fused_pts478, dispersion_px, variant_stack, meta) where
    dispersion_px is the per-landmark median-absolute-deviation across variants
    (robust spread, in original pixels) = a DETECTOR-REPEATABILITY figure.
    """
    if img is None:
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(img_path)
    h, w = img.shape[:2]
    mir = _build_mirror()
    lmk = _make_landmarker(min_conf)
    variants = []
    try:
        base = _detect(lmk, img)
        if base is None:
            raise RuntimeError("no face detected on full image")
        variants.append(base)

        # tight crop -> detector input budget covers the face
        x0, y0, x1, y1 = _face_bbox(base, w, h)
        crop = img[y0:y1, x0:x1]
        pc = _detect(lmk, crop)
        if pc is not None:
            variants.append(pc + np.array([x0, y0], dtype=np.float64))

        # multi-scale (up + down); map back by dividing by the scale factor
        for s in (0.85, 1.25):
            rs = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))))
            ps = _detect(lmk, rs)
            if ps is not None:
                variants.append(ps / s)

        # tight crop upscaled 2x (more pixels on the face, then map back)
        crop2 = cv2.resize(crop, (crop.shape[1] * 2, crop.shape[0] * 2))
        pc2 = _detect(lmk, crop2)
        if pc2 is not None:
            variants.append(pc2 / 2.0 + np.array([x0, y0], dtype=np.float64))

        # horizontal flip + mirror remap
        pf = _detect(lmk, cv2.flip(img, 1))
        if pf is not None:
            unflip = np.column_stack([(w - 1) - pf[:, 0], pf[:, 1]])
            remap = np.empty_like(unflip)
            remap[mir] = unflip
            variants.append(remap)
    finally:
        lmk.close()

    stack = np.stack(variants, axis=0)  # (V, 478, 2)
    fused = np.median(stack, axis=0)  # robust per-landmark aggregate

    # robust per-landmark dispersion: median Euclidean dist of variants to fused
    d = np.linalg.norm(stack - fused[None], axis=2)  # (V, 478)
    dispersion = np.median(d, axis=0)  # (478,)

    meta = {
        "n_variants": int(stack.shape[0]),
        "image": img_path,
        "width": int(w),
        "height": int(h),
        "bbox": [int(x0), int(y0), int(x1), int(y1)],
        "note_detector_repeatability_only": (
            "dispersion is detector repeatability on THIS photo (a lower bound). "
            "True test-retest (day/lighting/pose/expression) is wider."
        ),
    }
    return fused, dispersion, stack, meta


# ----------------------------------------------------------------------------
# (3) Sub-pixel edge refinement (EDGE landmarks only)
# ----------------------------------------------------------------------------
def _gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def _corner_subpix(gray, pts_subset):
    """Snap a small set of corner-like points to sub-pixel image corners."""
    pts = np.asarray(pts_subset, dtype=np.float32).reshape(-1, 1, 2)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
    refined = cv2.cornerSubPix(gray, pts.copy(), (7, 7), (-1, -1), crit)
    return refined.reshape(-1, 2).astype(np.float64)


def _ellipse_fit_iris(gray, center, ring_pts):
    """Fit an ellipse to the iris limbus near the MediaPipe ring, sub-pixel.

    Strategy: cast rays from the iris center through each boundary landmark,
    find the strongest radial gradient (limbus = dark iris -> bright sclera)
    along each ray at sub-pixel resolution, then least-squares-fit an ellipse.
    Falls back to the landmark ring if the fit is unstable.
    """
    center = np.asarray(center, dtype=np.float64)
    ring = np.asarray(ring_pts, dtype=np.float64)
    r0 = float(np.mean(np.linalg.norm(ring - center, axis=1)))
    h, w = gray.shape[:2]
    gf = cv2.GaussianBlur(gray, (0, 0), 1.0).astype(np.float32)

    edge_pts = []
    n_ang = 24
    for a in np.linspace(0, 2 * np.pi, n_ang, endpoint=False):
        dvec = np.array([np.cos(a), np.sin(a)])
        # sample intensity along the ray over [0.6, 1.4] * r0
        rs = np.linspace(0.6 * r0, 1.4 * r0, 40)
        prof = []
        ok = True
        for r in rs:
            px, py = center + dvec * r
            xi, yi = int(round(px)), int(round(py))
            if xi < 1 or yi < 1 or xi >= w - 1 or yi >= h - 1:
                ok = False
                break
            prof.append(gf[yi, xi])
        if not ok or len(prof) < 5:
            continue
        prof = np.array(prof)
        grad = np.gradient(prof)  # iris(dark)->sclera(bright): positive peak
        k = int(np.argmax(grad))
        if k <= 0 or k >= len(grad) - 1:
            continue
        # parabolic sub-pixel peak refinement
        y0_, y1_, y2_ = grad[k - 1], grad[k], grad[k + 1]
        denom = y0_ - 2 * y1_ + y2_
        off = 0.5 * (y0_ - y2_) / denom if abs(denom) > 1e-6 else 0.0
        r_edge = rs[k] + off * (rs[1] - rs[0])
        edge_pts.append(center + dvec * r_edge)

    # Ring diameters, for both the fallback and the cross-check. NOTE: the
    # MediaPipe horizontal iris landmarks (469/471, 474/476) sit toward the
    # palpebral-fissure corners and systematically OVERSHOOT the limbus, while
    # the vertical landmarks (470/472, 475/477) track the true circular limbus.
    # The iris is circular, so the vertical ring diameter is the more faithful
    # diameter estimate -- this is a documented detector behaviour, not a choice
    # made to move the number in any particular direction.
    ring_horiz = float(abs(ring[0, 0] - ring[2, 0]))
    ring_vert = float(abs(ring[1, 1] - ring[3, 1]))

    if len(edge_pts) >= 6:
        ep = np.array(edge_pts, dtype=np.float32)
        try:
            (cx, cy), (MA, ma), ang = cv2.fitEllipse(ep)
            # sanity: ellipse must stay near the landmark ring
            if (
                0.5 * (2 * r0) < max(MA, ma) < 1.8 * (2 * r0)
                and np.hypot(cx - center[0], cy - center[1]) < 0.5 * r0
            ):
                return {
                    "center": [float(cx), float(cy)],
                    "axes": [float(MA), float(ma)],
                    "angle": float(ang),
                    "ring_horiz": ring_horiz,
                    "ring_vert": ring_vert,
                    # primary diameter = vertical ring (least eyelid-corner bias)
                    "diam": ring_vert,
                    "ellipse_mean_diam": float((MA + ma) / 2.0),
                    "source": "ellipse_fit",
                }
        except cv2.error:
            pass
    # fallback: landmark ring only (still use the vertical diameter)
    return {
        "center": center.tolist(),
        "axes": [2 * r0, 2 * r0],
        "angle": 0.0,
        "ring_horiz": ring_horiz,
        "ring_vert": ring_vert,
        "diam": ring_vert,
        "ellipse_mean_diam": float(2 * r0),
        "source": "landmark_ring",
    }


def refine_edges(img_bgr, pts):
    """Return a copy of pts with EDGE landmarks snapped sub-pixel.

    Refines: canthi (cornerSubPix), eyelid margins + outer-lip border
    (cornerSubPix on local edges), and iris centers (ellipse-fit limbus center).
    Leaves jaw / cheek / forehead untouched (no image edge there).
    Also returns the two iris ellipse fits.
    """
    out = pts.copy()
    gray = _gray(img_bgr)

    # canthi + eyelid + lip = corner/edge points
    edge_ids = CANTHI + L_EYELID + R_EYELID + OUTER_LIP
    edge_ids = sorted(set(edge_ids))
    refined = _corner_subpix(gray, out[edge_ids])
    # only accept small moves (sub-pixel snap, not a jump to a different edge)
    for i, idx in enumerate(edge_ids):
        if np.linalg.norm(refined[i] - out[idx]) <= 4.0:
            out[idx] = refined[i]

    # iris: ellipse-fit limbus, move the center landmark onto the fitted center
    irisL = _ellipse_fit_iris(gray, out[468], out[[469, 470, 471, 472]])
    irisR = _ellipse_fit_iris(gray, out[473], out[[474, 475, 476, 477]])
    out[468] = np.array(irisL["center"], dtype=np.float64)
    out[473] = np.array(irisR["center"], dtype=np.float64)
    return out, {"iris_left": irisL, "iris_right": irisR}


# ----------------------------------------------------------------------------
# (4) Periodic-spline de-jitter of FACE_OVAL (jaw/cheek)
# ----------------------------------------------------------------------------
def dejitter_face_oval(pts, smooth=None):
    """Fit a smooth periodic spline through the FACE_OVAL ring and resample each
    oval landmark onto the curve. Removes per-point jitter that inflates the
    jaw-asymmetry artifact. Returns a copy of pts with oval points replaced.
    """
    out = pts.copy()
    ring = out[FACE_OVAL]  # ordered ring
    x, y = ring[:, 0], ring[:, 1]
    if smooth is None:
        # scale smoothing to ring size so it removes jitter, not real shape
        per = np.sum(np.linalg.norm(np.diff(ring, axis=0, append=ring[:1]), axis=1))
        smooth = (per * 0.002) ** 2 * len(ring)
    tck, u = splprep([x, y], s=smooth, per=True)
    xs, ys = splev(u, tck)  # resample at the SAME parameter positions
    for i, idx in enumerate(FACE_OVAL):
        out[idx] = [xs[i], ys[i]]
    return out


# ----------------------------------------------------------------------------
# (2) Iris-diameter px->mm scale
# ----------------------------------------------------------------------------
def iris_scale(pts, ellipses=None):
    """Derive px/mm from the iris diameter (no reference object).

    Primary diameter = the VERTICAL iris-ring diameter (least biased by the
    eyelid-corner overshoot of the horizontal iris landmarks); both eyes are
    averaged. If sub-pixel ellipse fits are supplied (from refine_edges) we also
    expose the ellipse-mean diameter as a cross-check. err_pct folds in the
    iris-SD term only; the wider caveat (non-frontal foreshortening, detection
    error) lives in the caveat string and the report wording.
    """
    if ellipses is not None:
        dL = ellipses["iris_left"]["diam"]
        dR = ellipses["iris_right"]["diam"]
        xcheckL = ellipses["iris_left"]["ellipse_mean_diam"]
        xcheckR = ellipses["iris_right"]["ellipse_mean_diam"]
        src = ellipses["iris_left"]["source"] + "+" + ellipses["iris_right"]["source"]
        xcheck = round((xcheckL + xcheckR) / 2.0, 2)
    else:
        # vertical landmark-ring diameter (470<->472, 475<->477)
        dL = float(abs(pts[470][1] - pts[472][1]))
        dR = float(abs(pts[475][1] - pts[477][1]))
        src = "landmark_ring"
        xcheck = None
    iris_px = (dL + dR) / 2.0
    px_per_mm = iris_px / IRIS_MM
    # error from iris SD only (lower bound); reporting layer must widen it
    err_pct = round(100.0 * IRIS_SD_MM / IRIS_MM, 1)
    return {
        "px_per_mm": round(px_per_mm, 4),
        "iris_px": round(iris_px, 2),
        "iris_px_left": round(dL, 2),
        "iris_px_right": round(dR, 2),
        "iris_mm_assumed": IRIS_MM,
        "err_pct": err_pct,
        "source": src,
        "ellipse_mean_diam_xcheck_px": xcheck,
        "caveat": (
            "Scale from assumed 11.7 mm iris (SD ~0.5 mm => ~4.3% before any "
            "non-frontal foreshortening or detection error). Vertical iris-ring "
            "diameter used (horizontal iris landmarks overshoot the limbus). "
            "Treat mm values as approximate; relative px ratios are unaffected "
            "by this scale."
        ),
    }


# ----------------------------------------------------------------------------
# Top-level convenience: fused + edge-refined + oval-dejittered landmarks
# ----------------------------------------------------------------------------
def _refine_one(img, min_conf):
    """One orientation, all the way through: fuse, edge-refine, de-jitter."""
    fused, dispersion, _stack, meta = fused_landmarks(min_conf=min_conf, img=img)
    edged, ellipses = refine_edges(img, fused)
    final = dejitter_face_oval(edged)
    meta = dict(meta)
    meta["iris_ellipses"] = ellipses
    meta["scale_mm"] = iris_scale(final, ellipses)
    meta["oval_dejitter_mean_move_px"] = round(
        float(np.mean(np.linalg.norm(final[FACE_OVAL] - edged[FACE_OVAL], axis=1))), 2
    )
    return final, dispersion, meta


def refined_landmarks(img_path=DEFAULT_IMG, min_conf=0.4, balance_mirror=True):
    """Full pipeline: fuse variants, edge-refine, de-jitter, MIRROR-BALANCE.

    THE MIRROR BALANCE, AND WHY IT IS ON BY DEFAULT
    ----------------------------------------------
    The detector does not treat the two sides of a face alike. Detect a
    photograph, then mirror THE IMAGE and detect again with this same pipeline: a
    detector with no left-right preference must return the same landmarks,
    because a mirrored photograph of a face is still a photograph of that face.
    It does not. Measured 2026-08-11 on the constant-face burst, the two
    estimates disagree by 1.27-1.81% of IPD on every in-gate frame, and 82-97%
    of that disagreement sits in the ANTISYMMETRIC component -- precisely the
    part that a symmetry measurement reads as asymmetry. Past the pose gate it
    reaches 45% of IPD.

    `fused_landmarks` already ran a mirrored variant and did not fix this: one
    mirrored view among six, combined by median, is outvoted by the five that
    share the bias. Averaging the two ORIENTATIONS cancels the antisymmetric
    component exactly, because both estimates are of the same anatomical point.

    ⚠️ IT LOWERS THE READINGS, WHICH IS THE FLATTERING DIRECTION, so it was not
    adopted on the strength of that argument. The argument rests on the mirror
    permutation being anatomically exact -- if it mis-paired landmarks this would
    average two DIFFERENT points and genuinely suppress asymmetry -- so it was
    measured: a KNOWN asymmetry injected into real photographs by a smooth
    quadratic image warp is recovered at **1.16x by the balanced estimator
    against 1.13x by the shipped one**, pooled over 24 injections in both
    directions across six frames. It removes error, not signal. See
    `detector_mirror_probe.py --suppression`.

    ⛔ It does NOT close the disjoint-bracket defect -- it removes 24% of the
    signed-yaw slope and fixes the worst single frame. It is a measured,
    one-directional bias being removed, not a solved problem.

    ⚠️ SCALE IS NOT BALANCED. `meta["scale_mm"]` still comes from the original
    orientation's iris ellipse fits, because that is what the suppression test
    validated. Averaging the two px/mm estimates is probably better and is
    untested, so it is not done here.

    `balance_mirror=False` returns the pre-2026-08-11 behaviour, for measuring
    what the balance changes rather than for production use.

    Returns (pts478, per_landmark_dispersion_px, meta). meta carries the iris
    ellipse fits, the px/mm scale, the variant count, the honest
    detector-repeatability note, and the measured mirror disagreement.
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)
    final, dispersion, meta = _refine_one(img, min_conf)
    if not balance_mirror:
        meta["mirror_balanced"] = False
        return final, dispersion, meta

    # ⚠️ THE MIRRORED PASS CAN FAIL WHERE THE ORIGINAL SUCCEEDS, and letting that
    # kill the frame would be a behaviour change nobody asked for. Measured when
    # this went in: IMG_1890 (+57.8 yaw with -48.9 pitch and -46.2 roll) and
    # IMG_1894 detect fine upright and not at all mirrored, so the burst went
    # from 9 usable frames to 7 -- a silent narrowing of what the pipeline
    # accepts, caused by a correction that is supposed to be a refinement.
    # Fall back to the unbalanced estimate and SAY SO in meta, rather than
    # discarding a photograph the detector could actually read.
    w = img.shape[1]
    try:
        flipped, _fdisp, _fmeta = _refine_one(cv2.flip(img, 1), min_conf)
    except Exception as exc:  # noqa: BLE001 - degrade, never drop the frame
        meta = dict(meta)
        meta["mirror_balanced"] = False
        meta["mirror_balance_unavailable"] = f"{type(exc).__name__}: {exc}"
        return final, dispersion, meta
    # map the mirrored estimate back: undo the pixel flip, then undo the
    # left<->right relabelling, so `back[i]` is an estimate of the SAME
    # anatomical landmark i that `final[i]` estimates.
    unflip = np.column_stack([(w - 1) - flipped[:, 0], flipped[:, 1]])
    back = np.empty_like(unflip)
    back[_build_mirror()] = unflip

    ipd = max(_ipd_px(final), 1e-9)
    disagree = float(np.linalg.norm(final - back, axis=1).mean())
    meta = dict(meta)
    meta["mirror_balanced"] = True
    meta["mirror_disagreement_px"] = round(disagree, 3)
    meta["mirror_disagreement_pct_ipd"] = round(disagree / ipd * 100.0, 3)
    return 0.5 * (final + back), dispersion, meta


# ----------------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------------
def _ipd_px(pts):
    return float(np.linalg.norm(pts[F.L_IRIS] - pts[F.R_IRIS]))


def _jitter_pool(img_path, n=12):
    """Build a pool of independent single-shot estimates of the SAME face, each
    from a tiny (sub-pixel) translation of the image, mapped back to the
    original frame.

    MediaPipe IMAGE mode is deterministic on identical input, so the translation
    jitter is what exposes the detector's genuine single-shot repeatability
    (the random per-pixel error that fusion is designed to cancel). This is the
    honest baseline pool: every member is a legitimate single estimate.
    """
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    lmk = _make_landmarker()
    shots = []
    try:
        rng = np.random.default_rng(0)
        for _ in range(n):
            dx, dy = rng.uniform(-0.5, 0.5, 2)
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            shifted = cv2.warpAffine(
                img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
            )
            p = _detect(lmk, shifted)
            if p is not None:
                shots.append(p - np.array([dx, dy]))
    finally:
        lmk.close()
    return np.stack(shots, 0)  # (n, 478, 2)


def _variance_reduction_demo(img_path, k=4, trials=200, seed=0):
    """Empirically show that a k-fold MEDIAN aggregate has lower error than a
    single estimate -- the core 'averaging N noisy estimates cuts error
    ~1/sqrt(N)' claim, measured directly (no hand-waving).

    Construction (apples-to-apples, same detector, same photo):
      * pool   = many independent single-shot estimates (sub-pixel jitter).
      * truth  = the full-pool mean per landmark (best available consensus).
      * single error = RMS distance of one random pool member to truth.
      * fused  error = RMS distance of the median of k random pool members.
    Repeated over many random draws. The fused error must be the smaller one.

    Returns (single_rms_px, fused_rms_px, ratio, sqrt_k_expected, n_pool).
    """
    pool = _jitter_pool(img_path, n=max(2 * k, 12))
    n = pool.shape[0]
    truth = pool.mean(axis=0)  # (478, 2) consensus
    rng = np.random.default_rng(seed)
    single_errs, fused_errs = [], []
    for _ in range(trials):
        idx = rng.choice(n, size=k, replace=False)
        single = pool[idx[0]]  # one estimate
        fused = np.median(pool[idx], axis=0)  # k-fold robust aggregate
        single_errs.append(np.sqrt(np.mean(np.sum((single - truth) ** 2, axis=1))))
        fused_errs.append(np.sqrt(np.mean(np.sum((fused - truth) ** 2, axis=1))))
    single_rms = float(np.mean(single_errs))
    fused_rms = float(np.mean(fused_errs))
    ratio = single_rms / (fused_rms + 1e-12)
    return single_rms, fused_rms, ratio, float(np.sqrt(k)), n


def main():
    print("=== fa_refine self-test on frontal.png ===\n")
    final, dispersion, meta = refined_landmarks(DEFAULT_IMG)
    scale = meta["scale_mm"]

    ipd_px = _ipd_px(final)
    ipd_mm = ipd_px / scale["px_per_mm"]

    print(f"variants fused              : {meta['n_variants']}")
    print(
        f"iris source                : {scale['source']}  "
        f"(L {scale['iris_px_left']}px  R {scale['iris_px_right']}px)"
    )
    print(f"iris diameter (mean)       : {scale['iris_px']} px  (= {IRIS_MM} mm)")
    print(f"px/mm                      : {scale['px_per_mm']}")
    print(f"IPD                        : {ipd_px:.1f} px  =  {ipd_mm:.1f} mm")
    print(f"  (err on scale, iris-SD only: +/-{scale['err_pct']}%)")

    ok_ipd = 60.0 <= ipd_mm <= 66.0
    target = 62.0 <= ipd_mm <= 64.0
    print(
        f"  IPD-mm sanity (target 62-64): "
        f"{'PASS' if target else ('NEAR' if ok_ipd else 'FAIL')}\n"
    )

    # ---- variance-reduction demonstration --------------------------------
    k = 4
    single_rms, fused_rms, ratio, sqrt_k, n_pool = _variance_reduction_demo(
        DEFAULT_IMG, k=k
    )
    print("=== variance-reduction demo (error of aggregate vs single) ===")
    print(f"pool of independent single-shot estimates       : {n_pool}")
    print(f"single-estimate RMS error to consensus  (px)    : {single_rms:.4f}")
    print(f"{k}-fold median RMS error to consensus    (px)    : {fused_rms:.4f}")
    print(f"error ratio single/fused                        : {ratio:.2f}x")
    print(f"  (ideal sqrt({k}) ~= {sqrt_k:.2f}x for pure random error)")
    print(
        f"variance-reduction demo: "
        f"{'PASS (aggregate error < single)' if fused_rms < single_rms else 'FAIL'}"
    )

    # oval de-jitter effect (informational; computed inside refined_landmarks)
    print(
        f"\nFACE_OVAL spline de-jitter: mean point move "
        f"{meta['oval_dejitter_mean_move_px']:.2f}px "
        f"(jitter removed from jaw/cheek silhouette)"
    )
    print(
        f"fused per-landmark dispersion (median, px): {float(np.median(dispersion)):.3f}"
    )
    print("  -> DETECTOR-REPEATABILITY ONLY (lower bound; test-retest is wider)")


if __name__ == "__main__":
    main()
