#!/usr/bin/env python3
"""Densify the MediaPipe↔FLAME correspondence from 105 landmarks to all 468.

WHY THIS EXISTS — THE 315-vs-1404 GAP
--------------------------------------
Every ground-truth figure this project holds was measured on a **105-landmark /
315-dof** fit, because `mediapipe_landmark_embedding.npz` embeds only 105 of
MediaPipe's 468 points into FLAME's surface. Production fits **468 / 1404**.
Same algorithm, 4.5x more free parameters from the same views. `p8_validate`
says so in print, and no claim covered the shipping size.

More scans cannot close that: the shortfall is not subjects, it is that a FLAME
mesh yields no position for 363 of the 468 points the product actually solves
for. This file supplies those positions, so the SAME real scanned humans can be
rendered as 468-point observations and the production-sized estimator scored
against them.

HOW, AND WHY IT IS THIS SIMPLE
-------------------------------
    1. Similarity-align the MediaPipe canonical mesh onto the FLAME template
       using the 105 known pairs (Umeyama: rotation, translation, uniform scale).
       Residual after alignment is 3.3% of IPD -- the two canonical faces are
       genuinely different faces, which is what step 2 absorbs.
    2. A thin-plate spline through those 105 pairs (3D biharmonic kernel r),
       exact at the anchors, evaluated at all 468 vertices. TPS is the minimum
       bending-energy interpolant, so between anchors it does the least
       structured thing available rather than inventing anatomy.
    3. Project each warped vertex onto the nearest point of the FLAME FACE
       surface -- exact point-to-triangle, brute force over every allowed
       triangle, no k-nearest shortcut that could silently miss. The result is
       (triangle, barycentric), so ANY mesh in FLAME topology yields 468 points
       that lie exactly on its surface.

⛔ THE 105 ANCHORS KEEP THEIR OFFICIAL EMBEDDING VERBATIM, and that is not a
detail. It makes the 105-point set an exact SUBSET of the 468-point set, so
"fit 468 points, then score the error on the 105" is measured with the same
ruler as the 105-dof run -- the one comparison that isolates the effect of the
extra degrees of freedom from the effect of scoring a different point set.
This project has already inflated a result 52x and another 2.6x by comparing
two numbers computed on different rulers; that is why the anchors are pinned.

WHAT THIS IS AND IS NOT
------------------------
✅ A geometrically consistent 468-point set on the real scanned surface, spread
   like MediaPipe's, usable as GROUND TRUTH for the 1404-dof estimator: truth
   and observations both derive from these points, so where exactly they sit
   cannot flatter the recovery measurement.
⛔ NOT a claim that these are the points MediaPipe's detector would return on a
   photograph of that person. Held-out anchors land 1.2% of IPD from their
   known position (p90 2.0%), so a densified point is placed to about the
   detector's own noise -- fine for "recover this point set", NOT fine for
   "this is where the detector would look". No metric derived from a single
   densified point's absolute position should be read as MediaPipe's.
⛔ NOT a fix for anything the 105-point validation already reported.

The `.npz` this writes is DELIBERATELY UNTRACKED, like every other derived
artifact here — the repo is deny-by-default for data files and this one rebuilds
deterministically in ~14 seconds from the FLAME model, which `selfcheck_dof_gap`
does before it checks anything.
"""

from __future__ import annotations

import json
import os
import pickle  # noqa: S403 - FLAME_masks.pkl, same MPI archive as flame_fit.py
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import flame_fit as FF  # noqa: E402
import multiview_probe as MV  # noqa: E402

MASKS = os.path.join(HERE, "models", "flame", "FLAME_masks.pkl")
OUT_NPZ = os.path.join(HERE, "mp468_embedding.npz")
OUT_JSON = os.path.join(HERE, "mp468_embedding.json")

# FLAME regions a MediaPipe face point may legitimately land on. Ears, scalp,
# neck, eyeballs and the mesh boundary are EXCLUDED so a point near the face
# oval cannot snap round onto the back of an ear or inside a socket -- a failure
# that would be invisible in an aggregate distance and catastrophic in one view.
FACE_REGIONS = (
    "face",
    "forehead",
    "eye_region",
    "left_eye_region",
    "right_eye_region",
    "nose",
    "lips",
)


def allowed_triangles(flame: dict) -> np.ndarray:
    """Triangles of the FLAME template whose three vertices are all face region."""
    with open(MASKS, "rb") as fh:
        masks = pickle.load(fh, encoding="latin1")  # noqa: S301 - provenance above
    keep = np.zeros(len(flame["v_template"]), dtype=bool)
    for name in FACE_REGIONS:
        keep[np.asarray(masks[name], dtype=np.int64)] = True
    faces = flame["faces"]
    return np.flatnonzero(keep[faces].all(axis=1))


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def similarity_align(
    src: np.ndarray, src_idx: np.ndarray, tgt: np.ndarray
) -> np.ndarray:
    """Umeyama-align `src` so its `src_idx` rows best match `tgt`. All rows move."""
    a, b = src[src_idx], tgt
    ca, cb = a.mean(0), b.mean(0)
    a0, b0 = a - ca, b - cb
    u, sv, vt = np.linalg.svd(a0.T @ b0)
    rot = u @ vt
    if np.linalg.det(rot) < 0:
        u[:, -1] *= -1
        rot = u @ vt
    scale = sv.sum() / max((a0**2).sum(), 1e-18)
    return (src - ca) @ rot * scale + cb


def tps_fit(p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Thin-plate spline mapping p -> q exactly. 3D biharmonic kernel phi(r) = r."""
    n = len(p)
    k = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
    pm = np.hstack([np.ones((n, 1)), p])
    m = np.zeros((n + 4, n + 4))
    m[:n, :n] = k
    m[:n, n:] = pm
    m[n:, :n] = pm.T
    rhs = np.zeros((n + 4, 3))
    rhs[:n] = q
    sol = np.linalg.solve(m + 1e-12 * np.eye(n + 4), rhs)
    return sol[:n], sol[n:]


def tps_apply(
    x: np.ndarray, ctrl: np.ndarray, c: np.ndarray, a: np.ndarray
) -> np.ndarray:
    d = np.linalg.norm(x[:, None, :] - ctrl[None, :, :], axis=2)
    return d @ c + np.hstack([np.ones((len(x), 1)), x]) @ a


def closest_on_triangles(
    pts: np.ndarray, verts: np.ndarray, tris: np.ndarray, chunk: int = 48
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact closest point on a triangle soup, for every point.

    Brute force over EVERY candidate triangle rather than a k-nearest shortlist.
    On a mesh with a fold -- a nostril rim, a lip seam -- the nearest triangle by
    centroid is not always the nearest triangle, and this project's standing
    lesson is that a shortcut which is right on average is exactly how one
    landmark ends up inside the mouth with nothing in the aggregate to show it.

    Returns (tri_index (n,), barycentric (n,3), distance (n,)).
    """
    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    ab, ac = b - a, c - a
    out_tri = np.zeros(len(pts), dtype=np.int64)
    out_bary = np.zeros((len(pts), 3))
    out_dist = np.zeros(len(pts))

    for s in range(0, len(pts), chunk):
        p = pts[s : s + chunk][:, None, :]  # (k,1,3)
        ap = p - a[None]
        d1 = (ab[None] * ap).sum(-1)
        d2 = (ac[None] * ap).sum(-1)
        bp = p - b[None]
        d3 = (ab[None] * bp).sum(-1)
        d4 = (ac[None] * bp).sum(-1)
        cp = p - c[None]
        d5 = (ab[None] * cp).sum(-1)
        d6 = (ac[None] * cp).sum(-1)
        va = d3 * d6 - d5 * d4
        vb = d5 * d2 - d1 * d6
        vc = d1 * d4 - d3 * d2

        # Barycentric (v, w) of the closest point, by Voronoi region.
        # Interior first, then each vertex and edge region overrides it.
        denom = 1.0 / np.maximum(va + vb + vc, 1e-30)
        v = vb * denom
        w = vc * denom

        at_a = (d1 <= 0) & (d2 <= 0)
        at_b = (d3 >= 0) & (d4 <= d3)
        at_c = (d6 >= 0) & (d5 <= d6)
        on_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
        on_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
        on_bc = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)

        t_ab = d1 / np.where((d1 - d3) == 0, 1e-30, d1 - d3)
        t_ac = d2 / np.where((d2 - d6) == 0, 1e-30, d2 - d6)
        t_bc = (d4 - d3) / np.where(
            ((d4 - d3) + (d5 - d6)) == 0, 1e-30, (d4 - d3) + (d5 - d6)
        )

        v = np.where(on_ab, t_ab, v)
        w = np.where(on_ab, 0.0, w)
        v = np.where(on_ac, 0.0, v)
        w = np.where(on_ac, t_ac, w)
        v = np.where(on_bc, 1.0 - t_bc, v)
        w = np.where(on_bc, t_bc, w)
        v = np.where(at_a, 0.0, v)
        w = np.where(at_a, 0.0, w)
        v = np.where(at_b, 1.0, v)
        w = np.where(at_b, 0.0, w)
        v = np.where(at_c, 0.0, v)
        w = np.where(at_c, 1.0, w)

        proj = a[None] + ab[None] * v[..., None] + ac[None] * w[..., None]
        dist = np.linalg.norm(proj - p, axis=-1)  # (k, m)
        best = dist.argmin(axis=1)
        rows = np.arange(len(best))
        out_tri[s : s + chunk] = best
        bv, bw = v[rows, best], w[rows, best]
        out_bary[s : s + chunk] = np.stack([1.0 - bv - bw, bv, bw], axis=1)
        out_dist[s : s + chunk] = dist[rows, best]
    return out_tri, out_bary, out_dist


# ---------------------------------------------------------------------------
# The embedding
# ---------------------------------------------------------------------------
def build(flame: dict, anchors: np.ndarray | None = None) -> dict:
    """Embed all 468 MediaPipe canonical vertices into FLAME's face surface.

    `anchors` selects which of the 105 official landmarks drive the spline;
    None means all of them. Passing a subset is what makes leave-one-out
    possible, and leave-one-out is the only measurement of this file's accuracy
    that uses a point whose answer was not given to it.
    """
    mp_index = flame["mp_index"]
    l0, _ = FF.landmark_basis(flame, 1)
    src, _mp_faces = MV.load_canonical()
    src = src[: MV.N_MESH]

    sel = np.arange(len(mp_index)) if anchors is None else np.asarray(anchors)
    aligned = similarity_align(src, mp_index[sel], l0[sel])
    ctrl = aligned[mp_index[sel]]
    c, a = tps_fit(ctrl, l0[sel])
    warped = tps_apply(aligned, ctrl, c, a)

    tri_ids = allowed_triangles(flame)
    faces = flame["faces"][tri_ids]
    tri, bary, dist = closest_on_triangles(warped, flame["v_template"], faces)

    out_tri = faces[tri]  # (468,3) vertex ids
    out_bary = bary

    if anchors is None:
        # The official embedding wins at every anchor, so the 105-point set is an
        # exact subset. Overwriting AFTER the spline (not before) keeps the
        # spline itself honest -- it was still asked to predict them.
        out_tri = out_tri.copy()
        out_bary = out_bary.copy()
        out_tri[mp_index] = flame["lmk_tri"]
        out_bary[mp_index] = flame["lmk_bary"]

    return {
        "tri": out_tri,
        "bary": out_bary,
        "warped": warped,
        "aligned": aligned,
        "proj_dist": dist,
        "align_residual": np.linalg.norm(aligned[mp_index[sel]] - l0[sel], axis=1),
    }


def landmarks_from_mesh(verts: np.ndarray, tri: np.ndarray, bary: np.ndarray):
    """The 468 MediaPipe-semantic points of any mesh in FLAME topology."""
    return (verts[tri] * bary[:, :, None]).sum(axis=1)


def load(path: str = OUT_NPZ) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(path)
    return d["tri"], d["bary"]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def leave_one_out(flame: dict, l0: np.ndarray, ipd: float) -> np.ndarray:
    """Error, per anchor, when that anchor is withheld from the spline.

    THE ONLY HONEST ACCURACY NUMBER THIS FILE HAS. Every other diagnostic here
    measures self-consistency; this one measures prediction of a point whose
    true FLAME position is known and was not supplied. It is end-to-end --
    align, spline, project -- because that is what a densified point goes
    through, and projection can only help a point whose truth lies on the
    surface, so leaving it out would understate the method that ships.
    """
    n = len(flame["mp_index"])
    errs = np.zeros(n)
    for j in range(n):
        keep = np.delete(np.arange(n), j)
        got = build(flame, anchors=keep)
        pred = landmarks_from_mesh(flame["v_template"], got["tri"], got["bary"])[
            flame["mp_index"][j]
        ]
        errs[j] = np.linalg.norm(pred - l0[j])
    return errs / ipd * 100.0


def main() -> int:
    flame = FF.load_flame()
    l0, _ = FF.landmark_basis(flame, 1)
    ipd = float(np.linalg.norm(l0[flame["ipd_pair"][0]] - l0[flame["ipd_pair"][1]]))
    perm = np.load(MV.MIRROR_PERM)[: MV.N_MESH]

    print("=" * 78)
    print("MP468 — DENSIFYING THE MEDIAPIPE↔FLAME CORRESPONDENCE, 105 → 468")
    print("=" * 78)
    print(f"  FLAME IPD in template units      : {ipd:.4f}  (all % below are of this)")

    emb = build(flame)
    tri, bary = emb["tri"], emb["bary"]
    pts = landmarks_from_mesh(flame["v_template"], tri, bary)

    ar = emb["align_residual"] / ipd * 100.0
    print(
        f"  similarity-align anchor residual : mean {ar.mean():.2f}%  max {ar.max():.2f}%"
        "   (the two canonical faces differ; the spline absorbs it)"
    )
    pd = emb["proj_dist"] / ipd * 100.0
    print(
        f"  spline → surface projection      : mean {pd.mean():.2f}%  p95 "
        f"{np.quantile(pd, 0.95):.2f}%  max {pd.max():.2f}%"
    )

    # 1. anchors must be preserved EXACTLY, or the 105/468 comparison is not
    #    same-ruler and every conclusion drawn from it is about two point sets.
    anchor_err = np.linalg.norm(pts[flame["mp_index"]] - l0, axis=1).max()
    print(f"  anchors reproduced exactly       : max error {anchor_err:.2e}")

    # 2. THE CORRESPONDENCE MUST NOT FOLD. A warp that turns the mesh inside out
    #    somewhere would still project onto the surface and still look fine in
    #    every aggregate distance -- and production reads its OCCLUSION from
    #    exactly these triangles' normals, so a fold would hide the wrong half of
    #    the face. Checked under MediaPipe's own topology, after aligning back
    #    into its frame so the comparison is orientation-to-orientation.
    src, mp_faces = MV.load_canonical()
    src = src[: MV.N_MESH]
    mp_faces = mp_faces[(mp_faces < MV.N_MESH).all(axis=1)]

    def _normals_areas(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a, b, c = p[mp_faces[:, 0]], p[mp_faces[:, 1]], p[mp_faces[:, 2]]
        n = np.cross(b - a, c - a)
        return n, np.linalg.norm(n, axis=1) / 2.0

    n_src, area_src = _normals_areas(src)
    n_emb, area_emb = _normals_areas(MV.procrustes_to(pts, src))
    cos = (n_src * n_emb).sum(1) / np.maximum(
        np.linalg.norm(n_src, axis=1) * np.linalg.norm(n_emb, axis=1), 1e-30
    )
    n_flipped = int((cos < 0).sum())
    print(
        f"  triangles folded / degenerate    : {n_flipped} flipped of "
        f"{len(mp_faces)},  mean normal agreement {cos.mean():.3f},  smallest area "
        f"ratio {(area_emb / area_src).min():.3f}"
    )

    # 3. no two points may collapse. A degenerate pair is 3 free parameters the
    #    fit can trade against nothing, and it would read as an easy landmark.
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1) / ipd * 100.0
    ds = np.linalg.norm(src[:, None, :] - src[None, :, :], axis=2)
    np.fill_diagonal(ds, np.inf)
    nn_mp = ds.min(axis=1) / float(np.linalg.norm(src[33] - src[263])) * 100.0
    n_tight = int((nn < nn_mp.min()).sum())
    print(
        f"  nearest-neighbour spacing        : embedded min {nn.min():.2f}% "
        f"median {np.median(nn):.2f}%   vs MediaPipe's own min {nn_mp.min():.2f}% "
        f"median {np.median(nn_mp):.2f}%   —  {n_tight} points tighter than "
        "MediaPipe's own tightest pair"
    )
    print("     ^ the median matches MediaPipe's almost exactly; the tight tail is the")
    print(
        "       INNER LIP CONTOUR, where projecting onto a closed-mouth template pulls"
    )
    print("       the upper and lower seams together. Disclosed, not fixed: it makes a")
    print("       few truth points closer together than MediaPipe's, which cannot")
    print("       flatter a recovery scored against those same points.")

    # 4. the embedding must not manufacture asymmetry. Measured against the
    #    OFFICIAL 105 on the same template: a densification that is worse than
    #    what already ships would inflate every true-asymmetry figure.
    tmpl_asym_468 = MV.symmetry_3d(pts, perm, float(np.ptp(pts[:, 0])))
    perm105 = FF.mp105_mirror(flame, np.load(MV.MIRROR_PERM)[: MV.N_MESH])
    tmpl_asym_105 = MV.symmetry_3d(l0, perm105, float(np.ptp(l0[:, 0])))
    print(
        f"  asymmetry of the TEMPLATE itself : 468-point {tmpl_asym_468:.3f}%  vs "
        f"official 105-point {tmpl_asym_105:.3f}%   (artifact floor, not a face)"
    )

    # 5. leave-one-out — the only number here that predicts an unseen point.
    loo = leave_one_out(flame, l0, ipd)
    print(
        f"  LEAVE-ONE-OUT anchor error       : mean {loo.mean():.2f}%  median "
        f"{np.median(loo):.2f}%  p90 {np.quantile(loo, 0.9):.2f}%  max {loo.max():.2f}%"
    )
    print("     ^ compare with the detector noise this project measured, 1.5% of IPD:")
    print("       a densified point is placed to about detector precision, which is")
    print("       enough to BE a point set to recover and not enough to claim it is")
    print("       where MediaPipe would look.")

    np.savez(OUT_NPZ, tri=tri, bary=bary)
    payload = {
        "_what": "MediaPipe 468 canonical vertices embedded in FLAME topology",
        "_method": "Umeyama align on the 105 official pairs → exact TPS → "
        "closest point on the FLAME face surface; the 105 keep their official "
        "embedding verbatim so they remain an exact subset",
        "_licence": "derived from the FLAME 2023 Open model (CC-BY-4.0) and "
        "MediaPipe's canonical_face_model.obj (Apache-2.0). Contains no scan data.",
        "_not_claimed": [
            "that a densified point is where MediaPipe's detector would land: "
            "held-out anchors miss by 1.2% of IPD on average",
            "any absolute-position metric read off a single densified point",
        ],
        "ipd_template_units": ipd,
        "align_residual_pct_ipd": {"mean": float(ar.mean()), "max": float(ar.max())},
        "projection_pct_ipd": {
            "mean": float(pd.mean()),
            "p95": float(np.quantile(pd, 0.95)),
            "max": float(pd.max()),
        },
        "anchor_max_error": float(anchor_err),
        "nn_spacing_pct_ipd": {
            "embedded_min": float(nn.min()),
            "embedded_median": float(np.median(nn)),
            "mediapipe_min": float(nn_mp.min()),
            "mediapipe_median": float(np.median(nn_mp)),
            "n_tighter_than_mediapipe_min": n_tight,
        },
        "topology": {
            "n_triangles": int(len(mp_faces)),
            "n_flipped": n_flipped,
            "mean_normal_agreement": float(cos.mean()),
            "min_area_ratio": float((area_emb / area_src).min()),
        },
        "template_asymmetry_pct": {
            "embedded_468": float(tmpl_asym_468),
            "official_105": float(tmpl_asym_105),
        },
        "leave_one_out_pct_ipd": {
            "mean": float(loo.mean()),
            "median": float(np.median(loo)),
            "p90": float(np.quantile(loo, 0.9)),
            "max": float(loo.max()),
            "per_anchor": [round(float(x), 4) for x in loo],
        },
    }
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n  wrote {OUT_NPZ}")
    print(f"  wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
