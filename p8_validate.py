#!/usr/bin/env python3
"""Public implementation; private development notes omitted."""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import flame_fit as FF  # noqa: E402
import multiview_probe as MV  # noqa: E402

OUT = os.path.join(HERE, "p8_validation.json")

# MEASURED, from the real-photo fits: reprojection RMS 4.12-10.36 px at IPD
# 283-337, i.e. ~1.5% of IPD. Held in absolute pixels referenced to the
# unforeshortened frontal size (see intake_calibration for why that matters).
NOISE_FRAC = 0.015


def rotation(pitch: float, yaw: float) -> np.ndarray:
    p, y = np.radians([pitch, yaw])
    rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    return ry @ rx


def sample_identity(flame: dict, rng: np.random.Generator, scale: float) -> np.ndarray:
    """One synthetic subject from FLAME's identity prior.

    FLAME's shapedirs are scaled so a coefficient of 1 is one population standard
    deviation, so standard normal coefficients draw a plausible face. `scale`
    below 1 keeps subjects nearer the mean; 1.0 samples the population as
    modelled.
    """
    beta = rng.normal(scale=scale, size=flame["shapedirs"].shape[2])
    return flame["v_template"] + flame["shapedirs"] @ beta


def render_views(
    verts: np.ndarray,
    flame: dict,
    yaws: list[float],
    rng: np.random.Generator,
    px_scale: float = 4000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Render the 105 embedded landmarks from several views, with occlusion.

    Occlusion is computed on the FULL 5023-vertex mesh and then sampled at the
    landmarks -- the honest way, since 105 scattered points have no surface of
    their own. Occluded landmarks are replaced by the FLAME MEAN face under the
    same pose, because that is what MediaPipe does: it never reports a landmark
    missing, it fills the hidden side from its prior.
    """
    tri, bc = flame["lmk_tri"], flame["lmk_bary"]
    mean_v = flame["v_template"]
    normals = MV.vertex_normals(verts, flame["faces"])
    # Noise reference is the unforeshortened frontal IPD, because the measured
    # 1.5% figure is 1.5% OF IPD. Referencing face width instead would inflate it
    # 1.29x (IPD/width = 0.774 on this landmark set) and quietly exaggerate every
    # bias number this file reports.
    mean_l = (mean_v[tri] * bc[:, :, None]).sum(axis=1)
    ref_ipd = px_scale * float(
        np.linalg.norm(mean_l[flame["ipd_pair"][0]] - mean_l[flame["ipd_pair"][1]])
    )

    obs, vis = [], []
    for yaw in yaws:
        rot = rotation(0.0, yaw)
        facing = normals @ (rot.T @ np.array([0.0, 0.0, 1.0]))
        seen = facing > 0.0
        src = np.where(seen[:, None], verts, mean_v)
        lm = (src[tri] * bc[:, :, None]).sum(axis=1)
        xy = px_scale * (lm @ rot[:2].T)
        xy = xy + rng.normal(scale=NOISE_FRAC * ref_ipd, size=xy.shape)
        obs.append(xy + 500.0)
        # per-landmark visibility, sampled from the mesh through the same weights
        w = (facing[tri].clip(0, None) * bc).sum(axis=1)
        vis.append(np.clip(w / max(w.max(), 1e-12), 0.0, 1.0) ** 0.5 + 0.05)
    return np.stack(obs), np.stack(vis)


def load_ply_verts(path: str) -> np.ndarray:
    """Vertices from a binary little-endian PLY in FLAME topology."""
    with open(path, "rb") as f:
        hdr = b""
        while b"end_header" not in hdr:
            line = f.readline()
            if not line:
                raise ValueError(f"no end_header in {path}")
            hdr += line
        nv = int(
            [l for l in hdr.split(b"\n") if l.startswith(b"element vertex")][0].split()[
                -1
            ]
        )
        return (
            np.frombuffer(f.read(nv * 12), dtype="<f4")
            .reshape(nv, 3)
            .astype(np.float64)
        )


def real_identities(scan_dir: str, flame: dict) -> list[tuple[str, np.ndarray]]:
    """Real human meshes, one per (subject, sequence), newest-first by name.

    ⭐ THE POINT OF THIS PATH: every reproducibility and containment figure P8
    holds was measured on identities SAMPLED FROM FLAME'S OWN PRIOR. That is not
    circular -- the fit under test knows nothing about FLAME -- but it does mean
    the faces inherit FLAME's training population, which is exactly the caveat
    P8 records: "NOT demographic coverage in the human sense". These are real
    scanned humans instead, so that caveat lifts for whoever is in this set.

    ⛔ EACH MESH IS RIGIDLY ALIGNED TO THE FLAME TEMPLATE FIRST, and skipping
    that step is not a subtlety -- it produced a 24.94% shape error and 7.5%
    bracket containment on the first run, against 4.08% and 100% on synthetic
    identities.
    That looked like a devastating finding about real faces. It was the harness.

    D3DFACS registrations arrive in the SCANNER's coordinate frame, not FLAME's:
    the template is tallest in y (extent 0.218/0.319/0.228) while a raw
    registration is largest in Z (0.232/0.286/0.327) and its centroid sits 0.050
    off origin, which is 23% of face width. `render_views` rotates about the
    canonical y-axis to synthesise yaw, so on an already-rotated head every view
    it produced was meaningless.

    Umeyama similarity alignment removes rotation, translation and uniform scale
    -- the scanner's arbitrary frame -- and nothing else. Shape differences are
    what remain, which is exactly what is under test. ⚠️ This is the SECOND units
    -or-frame error in this file's history: the first reported 83% containment
    and a fairness gap, and was noise applied to face WIDTH where it was
    documented as 1.5% of IPD.
    """
    import glob

    template = flame["v_template"]
    out = []
    for p in sorted(glob.glob(os.path.join(scan_dir, "**", "*.ply"), recursive=True)):
        rel = os.path.relpath(p, scan_dir)
        subj = rel.split(os.sep)[0]
        v = MV.procrustes_to(load_ply_verts(p), template)
        out.append((f"{subj}/{os.path.basename(p)}", v))
    return out


def run(
    n_subjects: int,
    seed: int,
    yaws: list[float],
    scale: float,
    scans: list[tuple[str, np.ndarray]] | None = None,
) -> list[dict]:
    flame = FF.load_flame()
    perm105 = FF.mp105_mirror(flame, np.load(MV.MIRROR_PERM)[: MV.N_MESH])
    l0, _ = FF.landmark_basis(flame, 1)
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    n = len(scans) if scans else n_subjects
    for i in range(n):
        # Real scan when supplied, else FLAME's prior. Everything downstream is
        # byte-identical, which is what makes the two runs comparable.
        verts = scans[i][1] if scans else sample_identity(flame, rng, scale)
        truth = (verts[flame["lmk_tri"]] * flame["lmk_bary"][:, :, None]).sum(axis=1)
        true_scale = float(np.ptp(truth[:, 0]))
        true_asym = MV.symmetry_3d(truth, perm105, true_scale)

        obs, vis = render_views(verts, flame, yaws, rng)
        # THE FIT UNDER TEST: free, 315 dof (105 landmarks x 3), no FLAME
        # knowledge whatsoever -- no prior, no coefficients, no basis.
        #
        # ⚠️ CORRECTED 2026-08-12, and the correction matters beyond the digits.
        # This said "1404 dof", which is 468x3 -- the PRODUCTION fit's size, copied
        # into a context where it does not apply. This validation runs entirely in
        # the 105-landmark FLAME-embedding space, because a FLAME mesh natively
        # yields 105 MediaPipe-semantic landmarks.
        #
        # ⛔ SO P8 VALIDATES A DIFFERENT ESTIMATOR FROM THE ONE THAT SHIPS, and its
        # containment must not be read as a guarantee for the 468-point path.
        #
        # ✅ THAT GAP IS NOW MEASURED SEPARATELY -- `p8_dof_gap.py`, claim
        # `dof-gap-closed-on-real-humans`. `mp468_embedding.py` densifies the
        # correspondence to all 468 points, and on these same scans the 1404-dof
        # fit costs 0.99-1.00x the shape error of this one at the same 105 points.
        # ⭐ The "4.5x more free parameters" worry had the wrong mechanism in mind:
        # `solve_shape` solves each landmark from its own 3x3 normal equation, so
        # extra points bring their own observations and compete for nothing. What
        # DOES cost the production path is estimating its own occlusion (+1.27pp
        # across 468) -- a different defect from the one this comment feared.
        fit = MV.fit_identity(
            obs, l0, np.zeros((0, 3), dtype=np.int64), iters=40, visibility=vis
        )
        pts = fit["shape"]
        got_scale = float(np.ptp(pts[:, 0]))
        band = MV.symmetry_3d_debiased(pts, perm105, got_scale, fit["landmark_var"])

        aligned = MV.procrustes_to(pts, truth)
        radius = np.linalg.norm(truth - truth.mean(0), axis=1).mean()
        rows.append(
            {
                "subject": scans[i][0] if scans else i,
                "true_asym_pct": round(float(true_asym), 4),
                "raw_pct": round(float(band["raw_pct_ipd"]), 4),
                "lower_pct": round(float(band["lower_pct_ipd"]), 4),
                "upper_pct": round(float(band["upper_pct_ipd"]), 4),
                "contained": bool(
                    band["lower_pct_ipd"] <= true_asym <= band["upper_pct_ipd"]
                ),
                "shape_err_pct": round(
                    float(
                        np.linalg.norm(aligned - truth, axis=1).mean() / radius * 100
                    ),
                    4,
                ),
                "reproj_px": round(float(fit["rms_px"]), 3),
            }
        )
        if (i + 1) % 20 == 0:
            print(f"    {i + 1}/{n_subjects} subjects")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subjects", type=int, default=120)
    ap.add_argument("--seed", type=int, default=20260802)
    ap.add_argument("--scale", type=float, default=1.0, help="identity SD sampled")
    ap.add_argument(
        "--scans",
        default=None,
        help="directory of REAL FLAME-topology .ply meshes, used instead of sampling",
    )
    args = ap.parse_args()

    yaws = [-32.0, -11.0, 9.0, 28.0]  # in-gate, the regime the bracket is verified in
    print("=" * 78)
    print("P8 — VALIDATION ACROSS A POPULATION OF FACES, WITH GROUND TRUTH")
    print("=" * 78)
    print(
        f"  {args.subjects} synthetic identities sampled from FLAME's CC-BY-4.0 prior"
    )
    print(
        f"  views at {yaws} deg (all in-gate)   detector noise 1.5% of IPD (measured)"
    )
    print("  fit under test: FREE (315 dof, 105 landmarks) — knows nothing about FLAME")
    print("  is a real test rather than fitting FLAME to FLAME.\n")

    scans = real_identities(args.scans, FF.load_flame()) if args.scans else None
    if scans:
        subs = sorted({k.split("/")[0] for k, _ in scans})
        print(
            f'  REAL SCANS: {len(scans)} meshes from {len(subs)} subjects — {", ".join(subs)}'
        )
        print(
            "  ⛔ licensed, non-distributable: numbers may be published, geometry may not."
        )
    rows = run(args.subjects, args.seed, yaws, args.scale, scans)
    true_a = np.array([r["true_asym_pct"] for r in rows])
    raw = np.array([r["raw_pct"] for r in rows])
    err = np.array([r["shape_err_pct"] for r in rows])
    contained = np.array([r["contained"] for r in rows])
    width = np.array([r["upper_pct"] - r["lower_pct"] for r in rows])

    print("\n" + "=" * 78)
    print("RESULTS")
    print("=" * 78)
    print(f"  subjects                     : {len(rows)}")
    print(
        f"  true asymmetry across faces  : {true_a.min():.2f} – {true_a.max():.2f}%"
        f"  (mean {true_a.mean():.2f}, SD {true_a.std(ddof=1):.2f})"
    )
    print(
        f"  shape recovery error         : {err.mean():.2f}% ± {err.std(ddof=1):.2f}"
        f"   (worst {err.max():.2f}%)"
    )
    print(
        f"  raw asymmetry bias           : {(raw - true_a).mean():+.3f}pp"
        f"   (same sign on {int((np.sign(raw - true_a) == np.sign((raw - true_a).mean())).sum())}/{len(rows)})"
    )
    print(f"  BRACKET CONTAINMENT          : {contained.mean():.1%}")
    print(f"  mean bracket width           : {width.mean():.2f}pp")

    print("\n  DOES ACCURACY DEPEND ON WHO THE SUBJECT IS?")
    print(
        f"    corr(shape error, true asymmetry) = {np.corrcoef(err, true_a)[0, 1]:+.3f}"
    )
    print(
        f"    corr(bracket width, true asymmetry) = {np.corrcoef(width, true_a)[0, 1]:+.3f}"
    )
    q = np.quantile(true_a, [0.25, 0.75])
    lo_m, hi_m = true_a <= q[0], true_a >= q[1]
    print(
        f"    most symmetric quartile : containment {contained[lo_m].mean():.0%},"
        f" error {err[lo_m].mean():.2f}%"
    )
    print(
        f"    least symmetric quartile: containment {contained[hi_m].mean():.0%},"
        f" error {err[hi_m].mean():.2f}%"
    )
    print("    a gap here would mean the pipeline works better on some faces than")
    print("    others — a fairness problem, not just an accuracy one.")

    # PROVENANCE MUST TRACK WHICH IDENTITIES WERE ACTUALLY USED. Leaving the
    # FLAME-sampled wording in place for a real-scan run would mislabel
    # non-commercial licensed data as CC-BY-4.0 — the exact confusion between the
    # FLAME MODEL licence and the FLAME DATA licence this project already caught
    # once, and the one that kept these datasets out for ten days.
    payload = {
        "_what": (
            "P8 population validation against REAL human ground truth (D3DFACS)"
            if scans
            else "P8 population validation against FLAME-sampled ground truth"
        ),
        "_licence": (
            "REAL D3DFACS registrations — NON-COMMERCIAL research licence. Numbers may be "
            "published; the geometry may NOT be redistributed in any form, including rendered "
            "or modified. Cite the FLAME paper and Cosker et al. 2011."
            if scans
            else "identities sampled from FLAME 2023 Open (CC-BY-4.0, commercial). "
            "The FLAME DATASETS (D3DFACS/FaMoS/CoMA) are non-commercial."
        ),
        "_identity_source": ("d3dfacs_real_scans" if scans else "flame_prior_samples"),
        "_independent_subjects": (
            len({str(r["subject"]).split("/")[0] for r in rows}) if scans else len(rows)
        ),
        "_not_validated": [
            "real photographs — no real skin, lighting, expression drift or camera",
            (
                "the 468-point production path — this fits 105 landmarks / 315 dof; "
                "that gap is measured separately by p8_dof_gap.py"
                if scans
                else "phenotype coverage beyond FLAME's own training population"
            ),
            "the skin layer, whose measured defects stand untouched",
        ],
        "n_subjects": len(rows),
        "seed": args.seed,
        "yaws": yaws,
        "containment": float(contained.mean()),
        "mean_shape_error_pct": float(err.mean()),
        "raw_bias_pp": float((raw - true_a).mean()),
        "mean_width_pp": float(width.mean()),
        "subjects": rows,
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n  wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
