#!/usr/bin/env python3
"""P2 — multi-view identity fitting against the FLAME 2023 Open shape prior.

WHY A STATISTICAL PRIOR AND NOT THE SPECTRAL BASIS
---------------------------------------------------
`shape_basis.py` constrains the fit to smooth deformations of the canonical
mesh. That fixed the conditioning, but a smoothness prior knows what is SMOOTH,
not what is FACIAL: it will happily fit a smooth shape no face has ever had.
FLAME's identity space is learned from 4D scans of real heads, so it constrains
the fit to shapes faces actually take.

WHAT THIS FITS
--------------
    vertices(beta) = v_template + shapedirs[:, :, :K] @ beta

which is LINEAR in beta, so the same alternating scheme already used elsewhere
applies: solve each view's weak-perspective camera in closed form, then solve
all K identity coefficients by one linear least squares. Pose blendshapes and
jaw articulation are deliberately left at zero -- this fits NEUTRAL IDENTITY,
and letting the jaw move would let expression absorb identity.

THE TRUNCATION TRAP — THE WHOLE REASON THIS FILE IS CAREFUL
------------------------------------------------------------
PCA over faces concentrates asymmetry in the HIGH components, because asymmetry
is low-variance. Measured on this model:

    identity components   0-10    4.7% antisymmetric energy
    identity components 200-300  42.0%

    share of representable asymmetry in the first N components
      N=50    11.4%      N=100   27.5%      N=200   62.6%

Standard FLAME practice is 50-100 identity components (DECA uses 100). Here that
would discard 72-89% of the asymmetry the model can express -- flattering every
subject on the exact metric this pipeline reports. `K_DEFAULT` is therefore the
full 300, and `selfcheck_flame.py` asserts that truncating suppresses asymmetry
rather than assuming it does not.

This is the same failure as spectral k=20, which is why it was looked for.

WHAT IS DELIBERATELY NOT CLAIMED
--------------------------------
  * Only 105 of MediaPipe's 468 landmarks embed into FLAME's surface, so this
    sees under a quarter of the observations the free/spectral fits use. Roughly
    12% of each mode's mesh motion is visible at those points -- uniformly across
    all 300 components, with none invisible, so there is no systematic blind
    spot; but the system is thin (105 x 2 x n_views observations against 300
    coefficients) and leans on the prior to stay conditioned.
  * The expression basis is tagged `supr_gender = male`. Identity fitting does
    not use it, but the caveat travels with the model.
  * Metric scale is not established here. FLAME is in its own units.
"""

from __future__ import annotations

import os
import pickle

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "models", "flame", "flame2023_Open.pkl")
EMBEDDING = os.path.join(HERE, "models", "flame", "mediapipe_landmark_embedding.npz")

N_IDENTITY = 300  # shapedirs is 400 = 300 identity + 100 expression (verified)
K_DEFAULT = 300  # see "THE TRUNCATION TRAP" above -- do not lower without measuring


def load_flame(path: str = MODEL, embedding: str = EMBEDDING) -> dict:
    """Load FLAME plus the MediaPipe landmark embedding.

    Returns the pieces this project needs, with the barycentric machinery already
    resolved into index/weight arrays.

    ON THE PICKLE. Unpickling executes arbitrary code, so it is only acceptable
    when the file's provenance is known. This one's is: FLAME is distributed by
    the Max Planck Institute for Intelligent Systems ONLY as a `.pkl` -- there is
    no JSON or safetensors alternative to prefer -- and this copy was fetched
    over HTTPS from flame.is.tue.mpg.de on 2026-08-02 behind an authenticated
    session, with the archive's sha256 recorded in NOTICE section 2b
    (a6b4c3dc...87a91). Re-verify that hash before trusting a replacement file,
    and never point this at a `.pkl` from anywhere else.
    """
    with open(path, "rb") as fh:
        m = pickle.load(fh, encoding="latin1")  # noqa: S301 - provenance above
    emb = np.load(embedding)
    faces = np.asarray(m["f"]).astype(np.int64)
    tri = faces[emb["lmk_face_idx"]]  # (105, 3) vertex ids per landmark
    mp_index = np.asarray(emb["landmark_indices"], dtype=np.int64)
    # Positions of MediaPipe's outer eye corners (33, 263) WITHIN the 105-point
    # set, so callers can reference IPD rather than face width. Both are present.
    pos = {int(v): i for i, v in enumerate(mp_index)}
    ipd_pair = (pos[33], pos[263])
    return {
        "ipd_pair": ipd_pair,
        "v_template": np.asarray(m["v_template"], dtype=np.float64),
        "shapedirs": np.asarray(m["shapedirs"], dtype=np.float64)[:, :, :N_IDENTITY],
        "faces": faces,
        "lmk_tri": tri,
        "lmk_bary": np.asarray(emb["lmk_b_coords"], dtype=np.float64),
        "mp_index": np.asarray(emb["landmark_indices"], dtype=np.int64),
    }


def landmark_basis(flame: dict, k: int = K_DEFAULT) -> tuple[np.ndarray, np.ndarray]:
    """Rest landmark positions and their deformation basis.

    Returns (L0 (105,3), B (105,3,k)) so that landmarks(beta) = L0 + B @ beta.
    Collapses the barycentric interpolation once, up front, rather than per call.
    """
    tri, bc = flame["lmk_tri"], flame["lmk_bary"]
    l0 = (flame["v_template"][tri] * bc[:, :, None]).sum(axis=1)
    basis = (flame["shapedirs"][tri][:, :, :, :k] * bc[:, :, None, None]).sum(axis=1)
    return l0, basis


def solve_beta(
    obs: np.ndarray,
    l0: np.ndarray,
    basis: np.ndarray,
    rots: np.ndarray,
    ts: np.ndarray,
    scales: np.ndarray,
    w: np.ndarray,
    prior_weight: float = 1.0,
) -> np.ndarray:
    """Identity coefficients by weighted linear least squares across all views.

    `prior_weight` is the FLAME PCA prior: shapedirs are scaled so a coefficient
    of 1 is one standard deviation of the training population, which makes a
    plain ridge the statistically correct regulariser rather than a fudge. It is
    what keeps a 300-coefficient fit conditioned against 105 landmarks.
    """
    n_f, n_p, _ = obs.shape
    k = basis.shape[2]
    a_rows, b_rows = [], []
    for i in range(n_f):
        m = scales[i] * rots[i][:2]  # (2,3)
        base = l0 @ m.T + ts[i]  # (105,2) prediction at beta = 0
        resid = obs[i] - base
        sw = np.sqrt(w[i])[:, None]
        # d(pred)/d(beta_a) = m @ basis[:, :, a]
        blk = np.einsum("de,pea->pda", m, basis).reshape(n_p * 2, k)
        a_rows.append(blk * np.repeat(sw, 2, axis=0))
        b_rows.append((resid * sw).reshape(-1))
    a = np.vstack(a_rows)
    b = np.concatenate(b_rows)
    ata = a.T @ a + prior_weight * np.eye(k)
    return np.linalg.solve(ata, a.T @ b), np.linalg.inv(ata)


def fit(
    obs_468: np.ndarray,
    flame: dict,
    k: int = K_DEFAULT,
    iters: int = 30,
    prior_weight: float = 1.0,
) -> dict:
    """Fit one FLAME identity across N views of MediaPipe landmarks.

    obs_468 is (F, 468, 2) in pixels -- the same array the other fitters take.
    Only the 105 embedded landmarks are used; the caller does not have to know.
    """
    import multiview_probe as MV

    sub = obs_468[:, flame["mp_index"], :]  # (F, 105, 2)
    l0, basis = landmark_basis(flame, k)
    n_f, n_p, _ = sub.shape

    beta = np.zeros(k)
    w = np.ones((n_f, n_p))
    rots = np.zeros((n_f, 3, 3))
    ts = np.zeros((n_f, 2))
    scales = np.ones(n_f)
    hist: list[float] = []

    beta_cov_unit = np.zeros((k, k))
    for _ in range(iters):
        lm = l0 + basis @ beta
        for i in range(n_f):
            rots[i], ts[i], scales[i] = MV.solve_pose(lm, sub[i], w[i])
        beta, beta_cov_unit = solve_beta(
            sub, l0, basis, rots, ts, scales, w, prior_weight
        )
        lm = l0 + basis @ beta
        err, tot = 0.0, 0.0
        for i in range(n_f):
            pred = scales[i] * (lm @ rots[i][:2].T) + ts[i]
            d2 = ((pred - sub[i]) ** 2).sum(1)
            err += float((w[i] * d2).sum())
            tot += float(w[i].sum())
        hist.append(float(np.sqrt(err / max(tot, 1e-9))))

    verts = flame["v_template"] + flame["shapedirs"][:, :, :k] @ beta

    # Per-landmark positional variance, so the asymmetry bracket works here too.
    # Uncertainty lives in the coefficients; propagate it to landmarks through the
    # basis: cov(l_j) = B_j @ cov(beta) @ B_j^T, and the bracket wants its trace.
    #
    # UNITS, AND A BUG THAT WAS CAUGHT BY THE BRACKET COLLAPSING TO ZERO WIDTH.
    # The design matrix A already carries the pose scale (its entries are
    # d(pixel)/d(beta)), so inv(A'A) is already in beta^2 per sigma_px^2 and
    # B inv(A'A) B' lands in FLAME units squared. Dividing by the pose scale as
    # well double-counted it and shrank the variance by ~1e7, producing a
    # zero-width bracket that contained the truth 0% of the time. Multiply by the
    # pixel residual variance and nothing else.
    sigma2 = hist[-1] ** 2
    cov = beta_cov_unit * sigma2
    landmark_var = np.einsum("pda,ab,pdb->p", basis, cov, basis)

    return {
        "beta": beta,
        "landmarks": l0 + basis @ beta,
        "landmark_var": landmark_var,
        "vertices": verts,
        "rots": rots,
        "ts": ts,
        "scales": scales,
        "rms_px": hist[-1],
        "history": hist,
        "k": k,
    }


def landmark_mirror_map(flame: dict) -> np.ndarray:
    """Mirror partner index for each of the 105 embedded landmarks.

    Built from the FLAME template rather than assumed: reflect across x and take
    the nearest neighbour. Verified 105/105 clean at a median distance of 0.0009
    against a face width of 0.115.
    """
    from scipy.spatial import cKDTree

    l0, _ = landmark_basis(flame, 1)
    flipped = l0.copy()
    flipped[:, 0] *= -1
    _dist, idx = cKDTree(l0).query(flipped)
    return idx


def mp105_mirror(flame: dict, mesh_perm: np.ndarray) -> np.ndarray:
    """Mirror map over the 105 embedded landmarks, in MediaPipe index order.

    EXISTS TO PREVENT A REPEAT OF A MISTAKE THIS PROJECT ALREADY MADE.
    FLAME reports asymmetry over its own 105 surface points scaled by its own
    width; the mesh fits report over 468 points scaled by IPD. Comparing those
    two numbers compares different rulers -- exactly hypothesis H1 from the P0
    work, which inflated an improvement 52x before it was caught.

    The 105 embedded MediaPipe landmarks are mirror-closed under the mesh's own
    permutation (verified: 0 missing partners, the sub-permutation is an
    involution, 11 midline fixed points), so they form a ruler every model can
    be measured on: FLAME reports these points natively, and a mesh fit is
    simply indexed down to them.
    """
    mp = flame["mp_index"]
    pos = {int(v): i for i, v in enumerate(mp)}
    missing = [int(v) for v in mp if int(mesh_perm[v]) not in pos]
    if missing:
        raise ValueError(f"landmark set is not mirror-closed; {len(missing)} unpaired")
    return np.array([pos[int(mesh_perm[v])] for v in mp], dtype=np.int64)


def asymmetry(landmarks: np.ndarray, mirror: np.ndarray) -> float:
    """Mean landmark asymmetry as % of interocular distance, on FLAME landmarks.

    Same construction as the mesh-space measure: fit the midsagittal plane, then
    compare each point with its reflected mirror partner.
    """
    mir = landmarks[mirror]
    diff = landmarks - mir
    normal = np.linalg.svd((diff - diff.mean(0)).T, full_matrices=False)[0][:, 0]
    normal = normal / np.linalg.norm(normal)
    centre = landmarks.mean(0)
    d = landmarks - centre
    reflected = centre + d - 2.0 * np.outer(d @ normal, normal)
    resid = np.linalg.norm(reflected - mir, axis=1)
    # scale by the widest horizontal extent, the landmark-set analogue of IPD
    scale = np.ptp(landmarks[:, 0])
    return float(resid.mean() / max(scale, 1e-12) * 100.0)
