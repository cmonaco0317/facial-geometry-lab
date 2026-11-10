#!/usr/bin/env python3
"""Multi-view identity fit — P0 experiment for the 2026-08-02 re-scope.

THE QUESTION THIS ANSWERS
-------------------------
The re-scoped goal claims that 5-6 casual photos beat 2 ideal ones because the
identity shape is CONSTANT across views while pose VARIES, which turns pose from
a confound into a solvable nuisance parameter. That is an identifiability claim,
and it is testable on data this project already has.

`pose_probe.py` established the single-view baseline on the 2026-08-01 burst: on
a face that could not have changed in 26 seconds, published symmetry ran
1.18% -> 27.29% of IPD (23x) and tracked |yaw| at r = 0.995. This probe asks
whether fitting ONE 3D shape across several of those same frames collapses that
spread.

METHOD
------
Weak-perspective (scaled orthographic) bundle adjustment over the 468 mesh
landmarks, by alternating least squares:

    x_ij ~= s_i * (R_i @ S_j)[:2] + t_i

  * POSE STEP   - given shape S, solve each frame's 2x3 camera M_i and offset
                  t_i by linear least squares, then project M_i to the nearest
                  scaled-rotation (SVD). Closed form, no initialisation needed.
  * SHAPE STEP  - given all cameras, solve each landmark's 3D position by a
                  weighted 3x3 normal equation. Independent per point.
  * VISIBILITY  - MediaPipe returns all 478 points even when a landmark is
                  physically occluded; it fills the far side from its mesh prior.
                  Fitting to those points fits the PRIOR, not the face. So each
                  observation is weighted by how much its surface normal faces
                  the camera, recomputed from the current shape every iteration.

Gauge freedom (S is defined only up to a similarity) is fixed by Procrustes-
aligning S to the canonical model and renormalising scale each iteration.

WHAT IS DELIBERATELY NOT CLAIMED
--------------------------------
  * Weak perspective ignores true perspective. The burst barely varies subject
    distance (IPD 283-337 px), so this is defensible HERE and is exactly the
    axis the backlog records as never measured.
  * The model is RIGID. Expression is not separated, so any expression change
    across frames lands in the shape residual. Over 26 seconds that is small but
    it is not zero, and it inflates the multi-view error rather than flattering
    it.
  * n=1 subject. This measures whether the METHOD identifies a constant face,
    not whether it generalises to other faces.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)

# MediaPipe-dependent imports are DEFERRED into the functions that need them, so
# the fitter's mathematics can be exercised on synthetic shapes with numpy alone.
# selfcheck_multiview.py depends on this: it proves fit_identity recovers a known
# deformation without a photograph, which is the guard that makes the reproducibility
# result mean anything (see BACKLOG "P0 RESULT", hypothesis H2). That result has been
# corrected three times -- 6.7x a sign bug, 2.6x a mixed ruler, 1.9x an inverted
# occlusion weighting; it stands at 4.7x.

N_MESH = 468  # the 468 face points; 468-477 are iris and excluded from shape

# MEASURED, and RE-DERIVED 2026-08-04 -- run `bracket_margin_calibration.py`.
#
# The bias bracket [corrected, raw] brackets SYSTEMATIC bias but not sampling
# scatter, so on its own it does not contain the truth often enough. Widening both
# ends by this fraction of the estimated noise floor is the SMALLEST margin
# reaching 100% containment over 6 view configurations x 15 subjects, which then
# held 90/90 on subjects it was not chosen on.
#
# Re-derived TWICE on 2026-08-04 as the floor underneath it was corrected:
#   0.10  original, against a floor ~6x too large in SD (naive (A'WA)^-1 instead
#         of the sandwich estimator, and a squared 2D residual used as a per-axis
#         sigma). With those fixed it contained only 68/90.
#   0.50  against the corrected floor but with the visibility weighting still
#         INVERTED, so the fit was being dragged by prior-filled occluded points.
#   0.20  current. Visibility sign fixed and the degrees-of-freedom factor added,
#         which together make the propagated bars calibrated at ~1.00x of truth.
# A margin that shrinks as the underlying estimate gets honest is what should
# happen: less of the interval is doing the work of an error nobody had measured.
#
# The derivation is a script now, not a comment. The old one lived only here, so
# when its input changed nothing could re-run it -- the same lesson
# `subset_intersection.py` exists for.
BRACKET_MARGIN = 0.20
CANONICAL_OBJ = os.path.join(HERE, "canonical_face_model.obj")
MIRROR_PERM = os.path.join(HERE, "mirror_perm.npy")


# ---------------------------------------------------------------------------
# Canonical model + mesh topology
# ---------------------------------------------------------------------------
def load_canonical(path: str = CANONICAL_OBJ) -> tuple[np.ndarray, np.ndarray]:
    """Return (verts (N,3), faces (M,3)) from the canonical face OBJ."""
    verts: list[list[float]] = []
    faces: list[list[int]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "v":
                verts.append([float(v) for v in parts[1:4]])
            elif parts[0] == "f":
                idx = [int(p.split("/")[0]) - 1 for p in parts[1:4]]
                faces.append(idx)
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def vertex_normals(shape: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals for a triangle mesh."""
    v0, v1, v2 = shape[faces[:, 0]], shape[faces[:, 1]], shape[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)  # face normal, magnitude = 2*area
    normals = np.zeros_like(shape)
    for k in range(3):
        np.add.at(normals, faces[:, k], fn)
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(norm, 1e-12)


# ---------------------------------------------------------------------------
# Bundle adjustment
# ---------------------------------------------------------------------------
def solve_pose(
    shape: np.ndarray, obs: np.ndarray, w: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Weak-perspective camera for one frame.

    Returns (R (3,3), t (2,), scale). Solves the 2x3 affine map by weighted
    least squares, then snaps it to the nearest scaled rotation via SVD.
    """
    wc = w[:, None]
    cs = (shape * wc).sum(0) / wc.sum()
    co = (obs * wc).sum(0) / wc.sum()
    sc, oc = shape - cs, obs - co
    # weighted linear solve for the 2x3 map M
    a = (sc * wc).T @ sc  # 3x3
    b = (sc * wc).T @ oc  # 3x2
    m = np.linalg.solve(a + 1e-9 * np.eye(3), b).T  # 2x3
    u, d, vt = np.linalg.svd(m, full_matrices=False)
    scale = float(d.mean())
    r2 = u @ vt  # nearest orthonormal 2x3
    r3 = np.cross(r2[0], r2[1])
    rot = np.vstack([r2, r3 / np.linalg.norm(r3)])
    t = co - scale * (rot[:2] @ cs)
    return rot, t, scale


def solve_shape(
    obs: np.ndarray, rots: np.ndarray, ts: np.ndarray, scales: np.ndarray, w: np.ndarray
) -> np.ndarray:
    """Solve every landmark's 3D position given all frame cameras.

    obs (F,P,2), rots (F,3,3), ts (F,2), scales (F,), w (F,P) -> (P,3)
    Each point is independent: a weighted 3x3 normal equation.
    """
    n_f, n_p, _ = obs.shape
    ata = np.zeros((n_p, 3, 3))
    ata2 = np.zeros((n_p, 3, 3))  # the same sum in w^2 -- see the sandwich below
    atb = np.zeros((n_p, 3))
    for i in range(n_f):
        m = scales[i] * rots[i][:2]  # 2x3
        resid = obs[i] - ts[i]  # (P,2)
        wi = w[i][:, None, None]
        ata += wi * (m.T @ m)[None]
        ata2 += (wi**2) * (m.T @ m)[None]
        atb += (w[i][:, None]) * (resid @ m)
    eye = 1e-6 * np.eye(3)[None]
    inv = np.linalg.inv(ata + eye)
    shape = (inv @ atb[..., None])[..., 0]
    # THE SANDWICH, AND WHY THE NAIVE FORM WAS WRONG BY 4.3x
    # ------------------------------------------------------
    # Cov = sigma^2 (A'WA)^-1 holds only when W is the INVERSE NOISE COVARIANCE.
    # W here is a VISIBILITY weight -- how much a landmark's surface normal faces
    # the camera -- which is a robustness device, not a noise model. For a general
    # W the covariance is the sandwich estimator
    #
    #     Cov = sigma^2 (A'WA)^-1 (A'W^2 A) (A'WA)^-1
    #
    # which is what this computes. The difference is not academic: measured on a
    # real 6-photograph fit the mean visibility weight was 0.054, because most of
    # the mesh sits near the 0.05 floor. The naive form divides by that weight and
    # inflated the variance ~18x (4.3x in SD), while the sandwich is invariant to a
    # uniform rescaling of W -- as it must be, since scaling every weight equally
    # cannot change how well the data determine a point.
    cov_unit = inv @ ata2 @ inv
    unit_var = np.einsum("pii->p", cov_unit)
    # IN-PLANE ONLY, KEPT SEPARATELY -- and the two are not interchangeable.
    #
    # `unit_var` is the TRACE, i.e. E[|3D error|^2], which is what
    # symmetry_3d_debiased needs because it compares against a squared 3D
    # distance. Anything measured on the FRONTAL RENDER lives in x,y and never
    # sees depth. Measured on a 36-degree synthetic set, the split is x 5.1%,
    # y 4.2%, z 90.7% -- depth is where a 4-view fit is weak, by an order of
    # magnitude. Spreading the trace isotropically therefore overstates the
    # in-plane SD by 2.67x, which is the difference between an error bar a reader
    # can use and one so wide the number beside it stops meaning anything.
    unit_var_xy = np.einsum("pii->pi", cov_unit)[:, :2].mean(1)
    return shape, unit_var, unit_var_xy, cov_unit


def procrustes_to(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Similarity-align src onto dst (Umeyama). Returns the transformed src."""
    cs, cd = src.mean(0), dst.mean(0)
    s0, d0 = src - cs, dst - cd
    u, sv, vt = np.linalg.svd(s0.T @ d0)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    scale = sv.sum() / max((s0**2).sum(), 1e-12)
    return (s0 @ r) * scale + cd


def fit_identity(
    obs: np.ndarray,
    canonical: np.ndarray,
    faces: np.ndarray,
    iters: int = 40,
    use_visibility: bool = True,
    basis: np.ndarray | None = None,
    visibility: np.ndarray | None = None,
) -> dict:
    """Alternating least squares for one shape across F views.

    obs (F,P,2) pixel landmarks. Returns the fitted shape and diagnostics.
    """
    n_f, n_p, _ = obs.shape
    shape = canonical.copy()
    w = np.ones((n_f, n_p))
    rots = np.zeros((n_f, 3, 3))
    ts = np.zeros((n_f, 2))
    scales = np.ones(n_f)
    hist: list[float] = []

    for it in range(iters):
        for i in range(n_f):
            rots[i], ts[i], scales[i] = solve_pose(shape, obs[i], w[i])
        if basis is None:
            shape, unit_var, unit_var_xy, cov_unit = solve_shape(
                obs, rots, ts, scales, w
            )
        else:
            # P2: solve 3k deformation coefficients instead of 1404 free
            # coordinates. unit_var is filled after the loop from the
            # coefficient covariance, which is the basis-aware noise floor.
            import shape_basis as SB

            coeff = SB.solve_coefficients(obs, canonical, basis, rots, ts, scales, w)
            shape = canonical + basis @ coeff
            unit_var = unit_var_xy = cov_unit = None
        pre_gauge_scale = np.linalg.norm(shape - shape.mean(0), axis=1).mean()
        shape = procrustes_to(shape, canonical)  # fix gauge
        # the gauge fix rescales the shape; carry the variance through the same
        # scale change or the noise floor would be quoted in the wrong units
        post_gauge_scale = np.linalg.norm(shape - shape.mean(0), axis=1).mean()
        gauge_ratio = (post_gauge_scale / max(pre_gauge_scale, 1e-12)) ** 2

        if visibility is not None:
            # Caller supplied per-view weights. Used when the point set has no
            # triangulation of its own -- e.g. P8 fits the 105 FLAME-embedded
            # landmarks, whose occlusion is computed from the full 5023-vertex
            # mesh and then sampled down. Computing normals over 105 scattered
            # points instead would be meaningless.
            w = visibility
        elif use_visibility:
            normals = vertex_normals(shape, faces)
            for i in range(n_f):
                # Direction TO the camera, expressed in shape space.
                #
                # SIGN IS LOAD-BEARING AND WAS WRONG TWICE. Read this before touching it.
                #
                # In the canonical model +z points OUT of the face (nose tip z=7.59
                # vs chin z=4.26, 100% of vertices n_z > 0), so in SHAPE space the
                # direction to a camera in front is [0, 0, +1]. That much was right,
                # and it is what the 2026-08-02 fix verified.
                #
                # What it did not check is that `rots[i]` is not a rotation between
                # two right-handed frames. The observations are IMAGE coordinates,
                # y DOWN, while the shape is y UP -- so the fitted 2x3 carries that
                # flip in its second row, and the third axis built from it by
                # cross(r2[0], r2[1]) comes out pointing AWAY from the camera.
                # Measured on real frames: r2[1] = [0.011, -0.996, -0.089] and
                # view = [0.339, 0.088, -0.937] at +12.9 deg of yaw, so `facing` was
                # negated and only 0.9-4.1% of the mesh counted as visible. 97.3% of
                # every frame's landmarks sat on the 0.05 floor.
                #
                # The weighting was therefore doing the exact thing its own comment
                # warned about -- favouring the self-occluded side and flooring the
                # visible one, upweighting the prior-filled points it exists to
                # suppress. Found 2026-08-04 while tracing why in-gate asymmetry
                # brackets are disjoint; the fit was being dragged by MediaPipe's
                # hallucinated far side, most at the jaw. Negating restores 97.6-99.1%
                # facing at frontal. Guarded in selfcheck_multiview.
                view = -(rots[i].T @ np.array([0.0, 0.0, 1.0]))
                facing = normals @ view
                w[i] = np.clip(facing, 0.0, 1.0) ** 0.5 + 0.05

        # weighted RMS reprojection error, in pixels
        err, tot = 0.0, 0.0
        for i in range(n_f):
            pred = scales[i] * (shape @ rots[i][:2].T) + ts[i]
            d2 = ((pred - obs[i]) ** 2).sum(1)
            err += float((w[i] * d2).sum())
            tot += float(w[i].sum())
        hist.append(float(np.sqrt(err / max(tot, 1e-9))))

    # Per-landmark positional variance: unit_var scales with the PER-AXIS residual
    # variance sigma_px^2, then through the gauge rescale into shape units.
    #
    # sigma^2 IS HALF THE SQUARED RMS, NOT THE WHOLE OF IT. `hist` accumulates
    # d2 = |pred - obs|^2, a SQUARED 2D DISTANCE, so hist[-1]^2 estimates
    # E[|residual vector|^2] = 2 sigma^2 for per-axis noise sigma. Using it
    # directly inflated every variance 2x (1.41x in SD). Confirmed against
    # injected noise: hist[-1] = 4.085 px where sigma was 3.0, and
    # 4.085/sqrt(2) = 2.888.
    # ...AND THE RESIDUAL OF A FITTED MODEL UNDERSTATES SIGMA. Each landmark's 3
    # coordinates are solved from its 2F observations, so the residual carries only
    # (2F-3)/(2F) of the true variance -- the ordinary degrees-of-freedom
    # correction. Predicted 0.79x under-report at F=4 (sqrt(5/8) = 0.79) and
    # measured at exactly 0.79/0.79/0.78 across 26/36/60 deg before this factor
    # went in. Under-reporting is the dangerous direction for an error bar, so a
    # known, derivable bias does not get left in.
    dof = max(2 * n_f - 3, 1)
    sigma2 = (hist[-1] ** 2) / 2.0 * (2 * n_f) / dof
    if basis is None:
        landmark_var = unit_var * sigma2 * gauge_ratio
        landmark_var_xy = unit_var_xy * sigma2 * gauge_ratio
        # The FULL 3x3 per-landmark covariance, which the in-plane summary above
        # cannot substitute for. Depth is where a few-view fit is weakest, and the
        # gauge fix Procrustes-aligns every shape to the canonical model -- so a
        # depth error tilts the whole shape and LANDS IN THE RENDERED x,y. Measured:
        # a 2-degree clustered set moves the rendered landmarks 3.01 px where a
        # 60-degree set moves them 1.54, while the in-plane posterior alone claims
        # the clustered set is the better-determined one. Propagating in 3D through
        # the real gauge-and-render path is the only way to see that.
        landmark_cov = cov_unit * sigma2 * gauge_ratio
    else:
        import shape_basis as SB

        landmark_var_xy = landmark_cov = None
        landmark_var = (
            SB.coefficient_variance(
                obs, canonical, basis, rots, ts, scales, w, hist[-1]
            )
            * gauge_ratio
        )
    return {
        "shape": shape,
        "rots": rots,
        "scales": scales,
        "ts": ts,
        "weights": w,
        "rms_px": hist[-1],
        "history": hist,
        "landmark_var": landmark_var,
        # In-plane only -- what anything measured on the frontal render needs.
        # None under a basis fit: SB.coefficient_variance returns the 3D trace and
        # has no per-axis form, so the caller must fall back rather than silently
        # inherit a number that is 2.67x too wide.
        "landmark_var_xy": landmark_var_xy,
        "landmark_cov": landmark_cov,
    }


# ---------------------------------------------------------------------------
# 3D symmetry on a fitted identity
# ---------------------------------------------------------------------------
def symmetry_3d(shape: np.ndarray, perm: np.ndarray, ipd: float) -> float:
    """Mean landmark asymmetry as % of IPD, measured on a 3D shape.

    Fits the midsagittal plane as the one minimising disagreement between each
    point and its mirror partner, then reports mean residual. This is the 3D
    analogue of analyze_advanced.symmetry's 42-pair figure.
    """
    mirrored = shape[perm]
    # the plane normal is the direction in which point and partner most differ
    diff = shape - mirrored
    u, _s, _vt = np.linalg.svd(diff - diff.mean(0), full_matrices=False)
    normal = np.linalg.svd((diff - diff.mean(0)).T, full_matrices=False)[0][:, 0]
    normal = normal / np.linalg.norm(normal)
    centroid = shape.mean(0)

    def reflect(p: np.ndarray) -> np.ndarray:
        d = p - centroid
        return centroid + d - 2.0 * np.outer(d @ normal, normal)

    resid = np.linalg.norm(reflect(shape) - mirrored, axis=1)
    return float(resid.mean() / ipd * 100.0)


def symmetry_3d_debiased(
    shape: np.ndarray,
    perm: np.ndarray,
    ipd: float,
    landmark_var: np.ndarray,
) -> dict:
    """Asymmetry with the measurement noise floor removed.

    THE BIAS THIS FIXES, AND WHY IT IS NOT OPTIONAL
    -----------------------------------------------
    Asymmetry is a MAGNITUDE: |reflect(p) - mirror(p)|. Random landmark error
    can only ADD to a magnitude, never cancel, so any noisy estimate of it is
    biased HIGH. Measured on synthetic faces where the truth is known, the raw
    figure ran +0.70pp / +12.5% high with the same sign on 20 of 20 subjects --
    systematic, not scatter.

    For this product that is the harmful direction: it tells a person they are
    more asymmetric than they are. The June research found harm in this category
    occurs independently of how gently the result is worded, so a number that is
    wrong in the damaging direction cannot be dressed up in softer prose.

    THE CORRECTION. Under independent per-landmark noise, the observed squared
    residual for a mirror pair carries the true squared residual plus both
    landmarks' positional variance:

        E[observed^2] = true^2 + var_i + var_mirror(i)

    so true = sqrt(max(0, observed^2 - var_i - var_mirror(i))). The variances
    come from the fit's own normal equations (`landmark_var`), so this is
    measured per subject rather than assumed.

    WHY THIS RETURNS AN INTERVAL AND NOT A NUMBER
    ---------------------------------------------
    The subtraction OVERSHOOTS, because the fit's residual carries systematic
    model error (rigid-body mismatch, prior-filled occluded points) on top of
    independent noise, so the floor it implies is too big. Measured: raw ran
    +10% high, fully-corrected -20% low.

    The obvious repair -- scale the floor by a fitted constant k -- was tried and
    REJECTED. Solving for the k that zeroes the bias gives 0.204 to 0.461 across
    noise levels and view counts, a 2.3x spread. A constant that needs retuning
    per condition is a fudge, not a calibration, and shipping it as a point
    estimate would manufacture false precision.

    So both ends are kept: `raw` is biased HIGH, `corrected` is biased LOW, and
    the truth sits between. That is this project's existing `bounded` state
    rather than a new invention. The interval is widened by a measured margin
    (see BRACKET_MARGIN) because the raw bias-bracket alone contained truth only
    93% of the time -- it brackets systematic bias but ignores sampling scatter.

    Clamping at zero makes the lower end conservative in the SAFE direction: a
    pair whose asymmetry is indistinguishable from noise contributes zero rather
    than contributing noise as a finding. `n_below_floor` says how often that
    happened, so a figure resting mostly on floored pairs is visible as such.
    """
    mirrored = shape[perm]
    diff = shape - mirrored
    normal = np.linalg.svd((diff - diff.mean(0)).T, full_matrices=False)[0][:, 0]
    normal = normal / np.linalg.norm(normal)
    centroid = shape.mean(0)

    d = shape - centroid
    reflected = centroid + d - 2.0 * np.outer(d @ normal, normal)
    obs2 = ((reflected - mirrored) ** 2).sum(axis=1)

    noise2 = landmark_var + landmark_var[perm]
    true2 = np.maximum(obs2 - noise2, 0.0)

    raw = float(np.sqrt(obs2).mean() / ipd * 100.0)
    corrected = float(np.sqrt(true2).mean() / ipd * 100.0)
    floor = float(np.sqrt(noise2).mean() / ipd * 100.0)
    margin = BRACKET_MARGIN * floor
    return {
        "raw_pct_ipd": raw,
        "corrected_pct_ipd": corrected,
        "noise_floor_pct_ipd": floor,
        "lower_pct_ipd": max(0.0, corrected - margin),
        "upper_pct_ipd": raw + margin,
        "n_below_floor": int((obs2 <= noise2).sum()),
        "n_pairs": int(len(obs2)),
    }


def asymmetry_interval(
    shape: np.ndarray,
    perm: np.ndarray,
    ipd: float,
    landmark_var: np.ndarray,
    frame_yaws: "list[float] | np.ndarray",
) -> dict:
    """The asymmetry bracket, with its validity conditioned on the photo set's pose.

    WHY THE BRACKET ALONE IS NOT ENOUGH — MEASURED, NOT ARGUED
    ----------------------------------------------------------
    The 2026-08-01 burst is one face over 26 seconds, so it cannot have changed,
    so every subset's interval must contain the SAME value. All 126 four-frame
    subsets must therefore share a common point. They do not:

        all 126 subsets          common intersection EMPTY
        the 15 subsets whose frames are all within +-45deg   EMPTY TOO

    🔴 THE GATE NO LONGER RESCUES IT. Re-derived 2026-08-04 by
    `subset_intersection.py` after the landmark covariance was corrected (sandwich
    estimator + per-axis sigma), which shrank the noise floor ~6x in SD. The
    earlier in-gate intersections -- [0.17, 2.21], then [0.00, 3.81] -- overlapped
    only because the floor was too large and every bracket was ~6x too wide.

    With a calibrated floor the in-gate brackets are disjoint by 4.34pp. Two
    subsets sharing THREE of their four frames read [0.00, 3.49] and
    [7.83, 11.42]: swapping ONE in-gate photograph moves the bracket clean off its
    own previous range, on a face that could not have changed. So the bracket
    does not cover on real photographs even inside the pose gate, and the pose
    gate is not the missing ingredient. Do not ship this figure until it does.

    The bracket is honest INSIDE the regime P1's intake gates for, and dishonest
    outside it. The cause is systematic, not scatter: including a single frame
    past 45deg shifts the estimate +5.68pp (2.59% -> 8.28%), and the estimate
    tracks the set's configuration (r=+0.80 with angular spread, r=+0.57 with
    max |yaw|, r=-0.62 with the smallest IPD in the set). Random-noise widening
    cannot fix a systematic term, so the bracket declares itself invalid instead.

    This is the project's existing three-state vocabulary applied to a SET rather
    than a photo: measured inside the gate, refused outside it.

    LIMIT, STATED: the 45deg threshold and the +5.68pp shift come from ONE
    subject's burst. Directional and measured, but n=1 -- a second subject could
    move both.
    """
    yy = np.abs(np.asarray(frame_yaws, dtype=float))
    out_of_gate = int((yy > 45.0).sum())
    base = symmetry_3d_debiased(shape, perm, ipd, landmark_var)

    if out_of_gate == 0:
        base["pose_condition"] = "measured"
        base["valid"] = True
        base["note"] = (
            "all frames within the +-45deg regime the bracket was verified in"
        )
    else:
        base["pose_condition"] = "refused"
        base["valid"] = False
        base["lower_pct_ipd"] = float("nan")
        base["upper_pct_ipd"] = float("nan")
        base["note"] = (
            f"{out_of_gate} frame(s) past 45deg. Measured on a constant face, "
            f"including such a frame shifts asymmetry +5.68pp systematically, and "
            f"intervals from different subsets stop overlapping. Refused rather "
            f"than reported with a caveat."
        )
    base["n_frames_out_of_gate"] = out_of_gate
    return base


def frontal_render(shape3d: np.ndarray, ref2d: np.ndarray) -> np.ndarray:
    """Orthographic frontal projection of a fitted shape, in a frame's pixels.

    Exists so the fitted identity can be scored with the PUBLISHED 2D metric
    rather than a bespoke 3D one. Comparing a 3D 468-point measure against
    analyze_advanced.symmetry's 42-pair figure compares different rulers and
    inflates any apparent improvement -- measured, not hypothetical: the first
    run of this probe reported 52x that way and 6.7x once corrected.

    Placed in pixels by matching the MESH IPD, which is the same quantity every
    downstream metric normalises by (fa_common.mesh_ipd). It used to match the
    outer-canthal distance instead -- see `published_symmetry` for why that was
    not a free choice.
    """
    xy = flat_xy(shape3d)
    xy = xy * render_scale(shape3d, ref2d)
    return xy - xy.mean(0) + ref2d.mean(0)


def flat_xy(shape3d: np.ndarray) -> np.ndarray:
    """The shape's x,y in IMAGE convention (y down), unscaled and unplaced."""
    xy = shape3d[:, :2].copy()
    # Y-DOWN. THE CANONICAL MODEL IS Y-UP AND IMAGE COORDINATES ARE NOT.
    #
    # The gauge fix Procrustes-aligns every fitted shape to canonical_face_model.obj,
    # where the top of the head has the LARGER y (trichion +8.26, menton -9.40).
    # Pixel coordinates run the other way. Slicing shape3d[:, :2] straight out
    # therefore returned an UPSIDE-DOWN face -- verified against a real photograph:
    # the render's y correlated -0.9995 with the detection, and negating y dropped
    # the mean disagreement from 305 px (99% of an IPD) to 9.4 px.
    #
    # It survived undetected because the only consumer was `published_symmetry`,
    # and bilateral symmetry about a vertical axis is INVARIANT to a vertical flip
    # -- `symmetry()` even re-orients its own axis. So every reproducibility number
    # this module has ever produced is unaffected, and every y-asymmetric metric
    # (facial thirds, eye position, philtrum, jaw) would have been measured on an
    # inverted face the moment the render reached them. Found 2026-08-04 by wiring
    # it into the pipeline; fixed here, guarded in selfcheck_setpath.
    xy[:, 1] *= -1.0
    return xy


def render_scale(shape3d: np.ndarray, ref2d: np.ndarray) -> float:
    """The exact pixel scale `frontal_render` applies. ONE definition, two callers.

    Phase B needs it to express the fit's per-landmark variance -- which lives in
    shape units -- as a positional SD in the same pixels as the rendered landmark
    file. Recomputing that factor at the call site is how two slightly different
    scales end up in one report, so it is defined once and used by both.
    """
    import fa_common as F  # lazy: this module stays importable without cv2/mediapipe

    return F.mesh_ipd(ref2d) / max(F.mesh_ipd(flat_xy(shape3d)), 1e-9)


def published_symmetry(shape3d: np.ndarray, ref2d: np.ndarray) -> float:
    """Score a fitted 3D identity with analyze_advanced.symmetry (the shipped metric).

    ONE RULER, AND IT HAD TO BE MEASURED TO GET IT (fixed 2026-08-04 (ruler))
    ----------------------------------------------------------------
    `AA.symmetry` divides by `|pts[468] - pts[473]|`. A fitted shape has no iris
    landmarks, so this used to write the OUTER CANTHI into those slots -- a length
    measured at 1.402x the iris IPD inside the pose gate. The single-photo baseline
    kept the true iris IPD, so the two sides of the headline comparison were
    divided by normalisers 40% apart, and the fitted side came out ~29% lower for
    no reason but the ruler. Effect on the published figure: the SD ratio fell from
    2.61x to 1.85x once both sides were measured the same way.

    `fa_common.with_mesh_iris` fills the slots from the eye-aperture ring
    centroids instead, which reproduce the detected iris IPD to 1.4%.
    """
    import analyze_advanced as AA
    import fa_common as F

    xy = frontal_render(shape3d, ref2d)
    full = F.with_mesh_iris(xy)
    return float(AA.symmetry(full, yaw_deg=0.0)["mean_asym_pct_ipd"])


def deform_shape(canonical: np.ndarray, amount: float, seed: int = 7) -> np.ndarray:
    """A smooth, deterministic, ASYMMETRIC deformation of the canonical face.

    Shared by selfcheck_multiview (correctness) and intake_calibration (threshold
    derivation) so the two cannot drift apart. Smooth (low spatial frequency) so
    it reads as a plausible face rather than noise, and asymmetric so a fit that
    collapsed to the symmetric prior would visibly fail to reproduce it.
    """
    rng = np.random.default_rng(seed)
    centred = canonical - canonical.mean(0)
    radius = np.linalg.norm(centred, axis=1).mean()
    basis = rng.normal(size=(3, 3))
    phase = rng.uniform(0, 2 * np.pi, size=3)
    field = np.sin(centred @ basis / radius * 1.7 + phase)
    field[:, 0] += 0.6 * (centred[:, 0] > 0)  # one-sided: breaks mirror symmetry
    return canonical + amount * radius * field / 3.0


def prior_divergence(shape3d: np.ndarray, canonical: np.ndarray) -> float:
    """Mean |fit - canonical| as % of face radius.

    GUARD against the failure mode that would make this whole experiment a lie:
    the gauge fix Procrustes-aligns to the canonical model, and the canonical
    face is symmetric, so a weak shape step would drift toward it and produce a
    number that is both low and stable FOR THE WRONG REASON.
    """
    radius = np.linalg.norm(canonical - canonical.mean(0), axis=1).mean()
    return float(np.linalg.norm(shape3d - canonical, axis=1).mean() / radius * 100.0)


# ---------------------------------------------------------------------------
# Experiment driver
# ---------------------------------------------------------------------------
def load_frames(session: str) -> list[dict]:
    """Detect landmarks + pose for every frame in a session directory."""
    import analyze_advanced as AA
    import fa_refine
    import pose_detect as FZ

    paths = sorted(
        p
        for ext in ("png", "jpg", "jpeg")
        for p in glob.glob(os.path.join(session, f"*.{ext}"))
    )
    out: list[dict] = []
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            _p3, mat, _wh, _img = FZ.detect_3d(path)
            pitch, yaw, roll = FZ.euler(mat[:3, :3])
            pts, _disp, _meta = fa_refine.refined_landmarks(path)
        except Exception as exc:  # noqa: BLE001 - report, never silently skip
            print(f"  {name}: FAILED ({type(exc).__name__}: {exc})")
            continue
        ipd = float(np.linalg.norm(pts[468] - pts[473]))
        sym = AA.symmetry(pts, yaw_deg=float(yaw))
        out.append(
            {
                "name": name,
                "pts2d": pts[:N_MESH],
                "ipd_px": ipd,
                "yaw": float(yaw),
                "pitch": float(pitch),
                "roll": float(roll),
                "sym_single_view": float(sym["mean_asym_pct_ipd"]),
            }
        )
    return out


def subsets(n: int, size: int, rng: np.random.Generator, count: int) -> list[list[int]]:
    """Random frame subsets of the given size, without replacement within each."""
    return [
        sorted(rng.choice(n, size=size, replace=False).tolist()) for _ in range(count)
    ]


def main() -> None:
    import fa_common as FC

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("session_dir", help="directory of frames from ONE sitting")
    ap.add_argument("--subset-size", type=int, default=4)
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260802)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument(
        "--baseline-gate",
        type=float,
        default=20.0,
        help="deg; single-view baseline uses only frames within this |yaw|, so the "
        "comparison is against photos the pipeline would actually accept",
    )
    ap.add_argument(
        "--no-visibility",
        action="store_true",
        help="ablation: fit without occlusion weighting",
    )
    ap.add_argument(
        "--basis-k",
        type=int,
        default=0,
        help="spectral deformation modes to fit. Default 0 (free fit) is the P2 "
        "head-to-head winner: given each model its OWN calibrated margin, the free "
        "fit produced the narrowest honest bracket on real photos (2.64pp vs "
        "spectral's 4.67pp) and the only common intersection that excludes zero. "
        "Use 80 for the spectral basis, which is near-unbiased but wider; never "
        "below MIN_SAFE_K=80, which suppresses localised asymmetry.",
    )
    args = ap.parse_args()

    canonical, faces = load_canonical()
    canonical = canonical[:N_MESH]
    faces = faces[(faces < N_MESH).all(axis=1)]
    perm = np.load(MIRROR_PERM)[:N_MESH]

    print("=" * 78)
    print("LOADING FRAMES")
    print("=" * 78)
    frames = load_frames(os.path.abspath(args.session_dir))
    if len(frames) < args.subset_size + 1:
        raise SystemExit(f"need > {args.subset_size} usable frames; got {len(frames)}")
    print(f"  usable: {len(frames)}")
    for f in frames:
        print(
            f"    {f['name']:<10} yaw {f['yaw']:>7.1f}  IPD {f['ipd_px']:>6.1f}px"
            f"   single-view symmetry {f['sym_single_view']:>6.2f}%"
        )

    obs_all = np.stack([f["pts2d"] for f in frames])
    ipds = np.array([f["ipd_px"] for f in frames])

    single = np.array([f["sym_single_view"] for f in frames])
    yaws = np.array([f["yaw"] for f in frames])
    ref2d = obs_all[
        int(np.argmin(np.abs(yaws)))
    ]  # straightest frame: scale/centre only

    # THE FAIR BASELINE. Scoring multi-view against ALL frames including +-70deg
    # ones flatters it enormously, because the pipeline's own gate REFUSES those.
    # Currently 47.6x that way, 4.7x restricted to what the pipeline accepts. (The
    # 52x/6.7x pair this comment used to quote was the same contrast before a sign
    # bug and then a ruler error were found -- the CONTRAST is the durable part.)
    gate = args.baseline_gate
    accepted = np.abs(yaws) <= gate
    if accepted.sum() < 2:
        raise SystemExit(
            f"need >=2 frames within |yaw|<={gate}deg for a baseline SD; got {accepted.sum()}"
        )
    single_gated = single[accepted]

    print()
    print("=" * 78)
    print("1. SINGLE-VIEW BASELINE — the number to beat")
    print("=" * 78)
    print(
        f"  ALL frames        n={len(single):<3} range {single.min():7.2f}-{single.max():7.2f}%"
        f"  SD {single.std(ddof=1):6.2f}   <- NOT the fair comparison"
    )
    print(
        f"  |yaw|<={gate:g}deg      n={accepted.sum():<3} range"
        f" {single_gated.min():7.2f}-{single_gated.max():7.2f}%"
        f"  SD {single_gated.std(ddof=1):6.2f}   <- what the pipeline accepts"
    )
    print(
        "\n  The gated row is the honest baseline: the ungated one includes frames the\n"
        "  project's own pose gate refuses, so charging them to single-view while\n"
        "  letting multi-view USE them as information is not apples-to-apples."
    )

    rng = np.random.default_rng(args.seed)
    picks = subsets(len(frames), args.subset_size, rng, args.trials)

    print()
    print("=" * 78)
    print(
        f"2. MULTI-VIEW FIT — one identity per random {args.subset_size}-frame subset"
    )
    print("=" * 78)
    if args.basis_k > 0:
        import shape_basis as SB

        basis = SB.spectral_basis(faces, N_MESH, args.basis_k)
        print(f"  fitting {args.basis_k} spectral modes (not 1404 free coordinates)")
        if args.basis_k < 80:
            print(
                f"  ⚠ k={args.basis_k} is below MIN_SAFE_K=80; measured to SUPPRESS "
                f"localised asymmetry — narrower intervals around a wrong value"
            )
    else:
        basis = None
        print("  free fit: 1404 free coordinates")

    results = []
    for k, idx in enumerate(picks):
        fit = fit_identity(
            obs_all[idx],
            canonical,
            faces,
            iters=args.iters,
            use_visibility=not args.no_visibility,
            basis=basis,
        )
        sym = published_symmetry(fit["shape"], ref2d)  # SAME ruler as the baseline
        drift = prior_divergence(fit["shape"], canonical)
        # mesh IPD, not the outer-canthal distance: the bracket is published as a
        # percentage OF THE IPD next to photo-derived figures, and the canthal
        # length is 1.40x the iris IPD (fa_common.iris_centres_from_mesh).
        ipd_fit = FC.mesh_ipd(fit["shape"])
        deb = asymmetry_interval(
            fit["shape"],
            perm,
            ipd_fit,
            fit["landmark_var"],
            [frames[i]["yaw"] for i in idx],
        )
        sub_yaws = [frames[i]["yaw"] for i in idx]
        spread = max(sub_yaws) - min(sub_yaws)
        results.append(
            {
                "frames": [frames[i]["name"] for i in idx],
                "yaw_spread_deg": round(spread, 1),
                "rms_px": round(fit["rms_px"], 2),
                "symmetry_pct_ipd": round(sym, 3),
                "prior_divergence_pct": round(drift, 2),
                "asymmetry_interval_pct_ipd": [
                    round(deb["lower_pct_ipd"], 2),
                    round(deb["upper_pct_ipd"], 2),
                ],
                "asymmetry_pose_condition": deb["pose_condition"],
                "noise_floor_pct_ipd": round(deb["noise_floor_pct_ipd"], 2),
            }
        )
        interval = (
            f"[{deb['lower_pct_ipd']:>5.2f},{deb['upper_pct_ipd']:>5.2f}]%"
            if deb["valid"]
            else f"REFUSED ({deb['n_frames_out_of_gate']} frame(s) >45deg)"
        )
        print(
            f"  subset {k + 1}: {'+'.join(frames[i]['name'][-4:] for i in idx)}"
            f"   yaw spread {spread:>5.1f}deg   reproj {fit['rms_px']:>6.2f}px"
            f"   symmetry {sym:>6.2f}%   asym {interval}"
        )

    multi = np.array([r["symmetry_pct_ipd"] for r in results])
    drifts = np.array([r["prior_divergence_pct"] for r in results])

    # GUARD: the canonical model scored through the identical path. If the fit
    # collapsed to the prior, every subset would return this number.
    floor = published_symmetry(canonical, ref2d)
    print()
    print("=" * 78)
    print("3. GUARD — is this just the canonical mesh prior coming back out?")
    print("=" * 78)
    print(f"  canonical model, same scoring path : {floor:.2f}%   <- collapse floor")
    print(
        f"  fitted identities                  : {multi.min():.2f}-{multi.max():.2f}%"
    )
    print(f"  mean |fit - canonical|             : {drifts.mean():.1f}% of face radius")
    collapsed = abs(multi.mean() - floor) < 0.5 and drifts.mean() < 2.0
    print(
        f"  VERDICT: {'COLLAPSED — result is meaningless' if collapsed else 'not collapsed; the fit carries subject-specific shape'}"
    )

    print()
    print("=" * 78)
    print("4. VERDICT — one identity across views vs one photo, SAME METRIC")
    print("=" * 78)
    print(f"  {'':26}{'single-view':>14}{'multi-view':>14}")
    print(f"  {f'n':26}{accepted.sum():>14}{len(multi):>14}")
    print(
        f"  {'range':26}{single_gated.min():>7.2f}-{single_gated.max():<6.2f}{multi.min():>7.2f}-{multi.max():<6.2f}"
    )
    print(f"  {'SD':26}{single_gated.std(ddof=1):>13.2f}%{multi.std(ddof=1):>13.2f}%")
    print(
        f"  {'CV':26}{100 * single_gated.std(ddof=1) / single_gated.mean():>13.1f}%"
        f"{100 * multi.std(ddof=1) / multi.mean():>13.1f}%"
    )
    improvement = single_gated.std(ddof=1) / max(multi.std(ddof=1), 1e-9)
    print()
    print(f"  SD ratio (gated single / multi): {improvement:.1f}x")
    print(
        "\n  A constant face was photographed, so ALL spread on both sides is\n"
        "  measurement error by construction. Lower is strictly better.\n"
        "  NOT established here: which side is closer to the TRUE asymmetry. A\n"
        "  frontal 2D view is blind to antero-posterior asymmetry by construction,\n"
        "  so multi-view reading higher is consistent with it seeing more, AND with\n"
        "  it having its own floor. This data cannot separate those."
    )

    # ---------------------------------------------------------------------
    # What does the fit actually want from a photo set? This is P1's spec.
    # ---------------------------------------------------------------------
    print()
    print("=" * 78)
    print("5. WHAT THE FIT WANTS FROM A PHOTO SET — the intake spec")
    print("=" * 78)
    extreme_yaw = 45.0
    n_extreme = np.array(
        [sum(abs(frames[i]["yaw"]) > extreme_yaw for i in idx) for idx in picks],
        dtype=float,
    )
    spreads = np.array([r["yaw_spread_deg"] for r in results])
    min_ipd = np.array([min(frames[i]["ipd_px"] for i in idx) for idx in picks])
    reproj = np.array([r["rms_px"] for r in results])

    def corr(a: np.ndarray, b: np.ndarray) -> float:
        if a.std() < 1e-9 or b.std() < 1e-9:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    print(f"  n={len(picks)} subsets. Correlation with the fitted symmetry value:")
    print(f"    yaw spread (deg)              r = {corr(spreads, multi):+.3f}")
    print(
        f"    frames beyond |yaw|>{extreme_yaw:g}deg      r = {corr(n_extreme, multi):+.3f}"
    )
    print(f"    smallest IPD in the subset    r = {corr(min_ipd, multi):+.3f}")
    print(f"  and with reprojection error:")
    print(
        f"    frames beyond |yaw|>{extreme_yaw:g}deg      r = {corr(n_extreme, reproj):+.3f}"
    )
    print(
        "\n  READ THIS CAREFULLY — it inverts the naive design. More angular spread is\n"
        "  NOT monotonically better. Beyond roughly 45deg the frames that add the most\n"
        "  'diversity' also lose resolution (IPD 83-101px vs 283-337px), self-occlude\n"
        "  the far side, and break the weak-perspective assumption. The intake should\n"
        "  ask for MODERATE, WELL-SPREAD angles -- not the most extreme ones available."
    )

    out = {
        "_what": "P0 multi-view identity-fit experiment (2026-08-02 re-scope)",
        "_metric": (
            "analyze_advanced.symmetry mean_asym_pct_ipd, both sides, both normalised "
            "by the IPD — the single-photo side by the detected iris centres and the "
            "fitted side by fa_common.mesh_ipd, which reproduces them to 1.4%. Before "
            "2026-08-04 (ruler) the fitted side used the outer canthi (1.402x) and this field "
            "claimed one ruler while the code used two"
        ),
        "intake_signals": {
            "extreme_yaw_threshold_deg": extreme_yaw,
            "r_symmetry_vs_yaw_spread": round(corr(spreads, multi), 3),
            "r_symmetry_vs_n_extreme": round(corr(n_extreme, multi), 3),
            "r_symmetry_vs_min_ipd": round(corr(min_ipd, multi), 3),
            "r_reproj_vs_n_extreme": round(corr(n_extreme, reproj), 3),
        },
        "session": os.path.abspath(args.session_dir),
        "visibility_weighting": not args.no_visibility,
        "subset_size": args.subset_size,
        "baseline_gate_deg": gate,
        "seed": args.seed,
        "collapse_floor_pct": round(float(floor), 3),
        "collapsed": bool(collapsed),
        "single_view": {
            "per_frame": {f["name"]: f["sym_single_view"] for f in frames},
            "gated_n": int(accepted.sum()),
            "gated_sd": round(float(single_gated.std(ddof=1)), 4),
            "ungated_sd": round(float(single.std(ddof=1)), 4),
        },
        "multi_view": {
            "trials": results,
            "sd": round(float(multi.std(ddof=1)), 4),
            "cv_pct": round(float(100 * multi.std(ddof=1) / multi.mean()), 2),
            "mean_prior_divergence_pct": round(float(drifts.mean()), 2),
        },
        "sd_ratio_gated_single_over_multi": round(float(improvement), 3),
        "mean_ipd_px": round(float(ipds.mean()), 1),
    }
    dest = os.path.join(os.path.abspath(args.session_dir), "multiview_probe.json")
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n  wrote {dest}")


if __name__ == "__main__":
    main()
