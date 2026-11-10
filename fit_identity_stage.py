#!/usr/bin/env python3
"""FRONT-END STAGE: a directory of casual photographs in, one fitted identity out.

WHAT THIS IS FOR
----------------
The product thesis is that 5-6 casual photographs beat 2 ideal ones, because the
identity shape is CONSTANT across them while pose, expression and lighting are
not -- which turns pose from a confound that cannot be removed from inside one
photograph into a nuisance parameter that can be solved for and subtracted.

Until this stage existed that was a claim in a document. `multiview_probe.py`
built and validated the fitter; nothing consumed it. The report still ran on one
frontal photograph plus one profile.

WHAT IT ACTUALLY DOES
---------------------
1. `intake.assess_set` gates the SET -- not each photo. It HALTS on a set that is
   not ready and prints the turn-direction advice, because a set of six photos
   all taken from the same angle carries the same confound six times and no
   amount of fitting recovers what none of them saw.
2. Detects landmarks per accepted frame, fits ONE shape across all of them, and
   projects that shape head-on.
3. Writes `landmarks_frontal.csv` -- the same 478-row contract every geometry
   stage already consumes -- plus `fit_diagnostics.json`.

So the thirteen downstream stages need no knowledge of any of this. That is the
whole trick, and it is why this is a stage rather than a rewrite.

WHAT IT DOES NOT DO, STATED PLAINLY
-----------------------------------
  * It does not improve MILLIMETRES. The fit is scale-free; the px->mm anchor is
    still one photograph's iris and still worth ~+-6%.
  * It does not improve SKIN. Appearance does not triangulate, and the skin stage
    deliberately keeps reading the photograph's own landmarks.
  * It is validated on synthetic deformations and ONE subject's burst. A better
    instrument is not a validation set.
  * The improvement it buys is 4.7x the reproducibility of the best single photo
    the pose gate allows -- measured, on one ruler, with the occlusion weighting
    finally the right way round. Much better, not good: its own CV is 36.3%.
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

import fa_common as F  # noqa: E402
import intake  # noqa: E402
import multiview_probe as MV  # noqa: E402

OUT_CSV = os.path.join(HERE, F.GEOMETRY_CSV)
OUT_DIAG = os.path.join(HERE, "fit_diagnostics.json")
OUT_POSTERIOR = os.path.join(HERE, "fit_posterior.npz")
OUT_REF = os.path.join(HERE, "frontal.png")
EXTS = ("png", "jpg", "jpeg", "PNG", "JPG", "JPEG")

# Below this intake weight a photo is not fitted. Not a pose gate -- intake
# already applied that -- but a floor for frames that are unreadable, tiny in
# frame, or turned past where the detector is trustworthy at all.
MIN_WEIGHT = 0.2

# Posterior draws written for Phase B. Pure numpy, so the cost is negligible;
# fixed seed so a report is reproducible from the same photographs.
N_POSTERIOR = 400
POSTERIOR_SEED = 20260804


def set_paths(set_dir: str) -> list[str]:
    """Every photograph in the set EXCEPT the lateral.

    `profile.*` is the silhouette input for `analyze_profile`, not a view for the
    fit. Two reasons it must be excluded rather than merely down-weighted, and the
    second is the one that matters:

      1. It is a different kind of input. The profile stage reads an outline; the
         fitter reads landmarks. Nothing is shared.
      2. A true lateral sits past 45 deg, and this project measured that frames
         past 45 deg degrade the fit rather than enriching it -- IPD collapses
         337 -> 83 px so absolute detector error dominates, the far side is
         self-occluded, and `asymmetry_interval` REFUSES any set containing one.
         Fitting the silhouette shot would have built the report's geometry from
         a set the project's own gate refuses, while the printed angular spread
         got wider and looked better for it.
    """
    keep = []
    for ext in EXTS:
        for p in glob.glob(os.path.join(set_dir, f"*.{ext}")):
            if os.path.splitext(os.path.basename(p))[0].lower() == "profile":
                continue
            keep.append(p)
    return sorted(keep)


def detect_frames(paths: list[str]) -> list[dict]:
    """Landmarks + head pose per frame, reusing the probe's validated path."""
    import fa_refine
    import pose_detect as FZ

    out: list[dict] = []
    for path in paths:
        name = os.path.basename(path)
        try:
            _p3, mat, _wh, _img = FZ.detect_3d(path)
            _pitch, yaw, _roll = FZ.euler(mat[:3, :3])
            pts, _disp, _meta = fa_refine.refined_landmarks(path)
        except Exception as exc:  # noqa: BLE001 - report, never silently skip
            print(f"    {name:<24} DROPPED ({type(exc).__name__}: {exc})")
            continue
        out.append(
            {
                "file": name,
                "path": path,
                "pts478": pts,
                "yaw": float(yaw),
                "ipd_px": float(np.linalg.norm(pts[468] - pts[473])),
            }
        )
        print(f"    {name:<24} yaw {float(yaw):+7.1f}   IPD {out[-1]['ipd_px']:6.1f}px")
    return out


def choose_reference(frames: list[dict]) -> dict:
    """The straightest frame. It becomes `frontal.png` -- the APPEARANCE source.

    The fit is pose-free and needs no reference, but four things still need an
    actual photograph: the skin layer, the px->mm iris anchor, every annotated
    overlay, and the averageness bank comparison. Using the straightest frame for
    those keeps them in the regime their own gates were measured in.
    """
    return min(frames, key=lambda f: abs(f["yaw"]))


def fit(frames: list[dict]) -> dict:
    canonical, faces = MV.load_canonical()
    canonical = canonical[: MV.N_MESH]
    faces = faces[(faces < MV.N_MESH).all(axis=1)]
    obs = np.stack([f["pts478"][: MV.N_MESH] for f in frames])
    return MV.fit_identity(obs, canonical, faces, iters=40, use_visibility=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("set_dir", help="directory of photographs of ONE person")
    ap.add_argument(
        "--min-frames",
        type=int,
        default=3,
        help="below this the fit is under-determined regardless of what intake says",
    )
    args = ap.parse_args()

    set_dir = os.path.abspath(args.set_dir)
    if not os.path.isdir(set_dir):
        print(f"!! not a directory: {set_dir}")
        return 1
    paths = set_paths(set_dir)
    if not paths:
        print(f"!! no photographs in {set_dir}")
        return 1
    # TWO LISTS, ON PURPOSE. `paths` is what intake judges and the fitter sees --
    # the lateral excluded, because it is a silhouette input rather than a view.
    # `all_paths` is what the staleness stamp covers, and it includes the lateral,
    # because swapping the profile shot changes the REPORT even though it cannot
    # change the fit. A guard that only watched the fit's inputs would let a new
    # profile through unnoticed.
    all_paths = sorted(
        p for ext in EXTS for p in glob.glob(os.path.join(set_dir, f"*.{ext}"))
    )

    # ---- THE SET GATE, and it HALTS ---------------------------------------
    # Deliberately the same shape as run_pipeline's capture gate: refuse before
    # measuring, rather than measure and caveat. A set that is not ready produces
    # a fit that looks like every other fit.
    print("=" * 78)
    print(f"INTAKE — {len(paths)} photo(s) in {os.path.basename(set_dir)}")
    print("=" * 78)
    verdict = intake.assess_set(paths)
    intake.render(verdict)
    # ⭐ NOT-READY IS NO LONGER ONE STATE. It was a single cliff at the 52-degree
    # floor, so a 49-degree set — which this project's OWN calibration puts at
    # 4.1% shape error and an 83% pass rate — produced nothing at all, while a
    # 53-degree set produced a full-confidence report. That cliff is not in the
    # measurements; it was in the gate.
    #
    # The calibration sweep (400 trials against known truth) by spread band:
    #     0-15 deg   11.09% error,  0% usable
    #     15-30 deg   6.48% error, 11% usable
    #     30-45 deg   4.56% error, 64% usable
    #     45-60 deg   4.08% error, 83% usable
    #
    # So below 30 degrees the geometry genuinely does not work and the refusal
    # stands. Between 30 and the floor it works with more error — which this
    # project already knows how to express: widen the brackets and say so. That
    # is DEGRADED, not REFUSED, and it is recorded in fit_diagnostics.json so
    # every downstream bracket can carry the penalty.
    cov = verdict["coverage"]
    spread = float(cov.get("effective_spread_deg") or 0.0)
    views = int(cov.get("n_usable") or 0)
    cal = verdict["calibration"]
    floor = float(cal["min_spread_deg"])
    HARD_FLOOR = 30.0

    degraded = None
    if not verdict["ready"]:
        blocking = views < int(cal["min_views"]) or spread < HARD_FLOOR
        if blocking:
            print()
            print(
                "!! HALTED before fitting anything: this photo set cannot support a report."
            )
            if spread < HARD_FLOOR:
                print(
                    f"   Angular spread is {spread:.0f} deg, under the hard floor of "
                    f"{HARD_FLOOR:.0f}. Below that the measured shape error runs 6.5-11% "
                    f"and almost no set comes out usable — this is the geometry failing, "
                    f"not a strict threshold."
                )
            print(
                "   Nothing was written. The advice above is specific — follow it and re-run."
            )
            return 1
        # Between the hard floor and the shipping floor: proceed, degraded.
        widen = floor / spread if spread > 0 else 1.0
        degraded = {
            "reason": "angular spread below the shipping floor",
            "effective_spread_deg": spread,
            "shipping_floor_deg": floor,
            "hard_floor_deg": HARD_FLOOR,
            "bracket_widen_factor": round(widen, 3),
            "what_it_means": (
                f"This set spans {spread:.0f} deg against a {floor:.0f} deg floor. It is "
                f"above the {HARD_FLOOR:.0f} deg hard floor where the geometry stops "
                f"working, so a report is produced — with every measured bracket widened "
                f"{widen:.2f}x and labelled. It is a real reading with more error, not a "
                f"full-confidence one."
            ),
        }
        print()
        print(f"!! DEGRADED, not refused: {degraded['what_it_means']}")
        print()

    # ---- Detection --------------------------------------------------------
    # FIT THE FRAMES INTAKE COUNTED, NOT THE ONES IT MERELY TOLERATED.
    # `intake.angular_coverage` already discards frames past MAX_USEFUL_ABS_YAW
    # when it computes the spread the readiness verdict is based on -- they "still
    # COUNT toward the set but earn no spread credit". The fit has to agree with
    # that, or it is fitting views the verdict that let it run never counted.
    accepted = [
        p["path"]
        for p in verdict["photos"]
        if p.get("weight", 0.0) > MIN_WEIGHT
        and p.get("yaw") is not None
        and abs(p["yaw"]) <= intake.MAX_USEFUL_ABS_YAW
    ]
    skipped = [
        p["file"]
        for p in verdict["photos"]
        if p.get("yaw") is not None and abs(p["yaw"]) > intake.MAX_USEFUL_ABS_YAW
    ]
    print()
    print("=" * 78)
    print(f"DETECTING — {len(accepted)} photo(s) intake accepted")
    print("=" * 78)
    if skipped:
        print(
            f"    not fitted, past {intake.MAX_USEFUL_ABS_YAW:.0f} deg: "
            f"{', '.join(skipped)} — measured to degrade the fit, and the set-level\n"
            f"    asymmetry bracket refuses any set containing one"
        )
    frames = detect_frames(accepted)
    if len(frames) < args.min_frames:
        print()
        print(
            f"!! HALTED: {len(frames)} frame(s) survived detection, need "
            f"{args.min_frames}. Intake judges a photo it can see a face in; the "
            f"refined detector is stricter and half a deliberate profile burst has "
            f"failed it before. Nothing was written."
        )
        return 1

    # ---- The fit ----------------------------------------------------------
    print()
    print("=" * 78)
    print(f"FITTING ONE IDENTITY across {len(frames)} views")
    print("=" * 78)
    result = fit(frames)
    ref = choose_reference(frames)
    rendered = MV.frontal_render(result["shape"], ref["pts478"][: MV.N_MESH])
    full = F.with_mesh_iris(rendered)

    yaws = [f["yaw"] for f in frames]
    spread = max(yaws) - min(yaws)
    drift = MV.prior_divergence(result["shape"], MV.load_canonical()[0][: MV.N_MESH])
    print(f"  reprojection RMS       : {result['rms_px']:.2f} px")
    print(f"  angular spread fitted  : {spread:.1f} deg over {len(frames)} views")
    print(f"  |fit - canonical|      : {drift:.1f}% of face radius")
    if drift < 2.0:
        # The gauge fix Procrustes-aligns to a SYMMETRIC canonical face, so a fit
        # that failed to move would return a low, stable, meaningless number.
        print(
            "  !! the fit barely moved off the canonical prior — treat with suspicion"
        )
    print(f"  reference photograph   : {ref['file']} (yaw {ref['yaw']:+.1f})")

    # ---- PHASE C: the asymmetry figure is a BRACKET from these photographs ---
    #
    # The single-photo path reports asymmetry as a POINT, and a point is the one
    # thing this measurement has never earned. It is a magnitude, so noise can
    # only add to it; the raw figure is biased HIGH and the noise-floor
    # subtraction overshoots low, so the truth is bracketed rather than known.
    # `asymmetry_interval` is that bracket plus the pose condition it is only
    # valid under.
    #
    # WHY IT IS SHIPPABLE HERE AND NOT BEFORE. The bracket has to COVER: on one
    # unchanging face, brackets from different subsets of the same sitting must
    # share a common point, or at least one of them is wrong. Measured
    # 2026-08-11 by `subset_intersection.py`, over the 13 four-frame subsets of
    # the 26-second burst that `intake.assess_set` ACCEPTS, they intersect at
    # [2.18, 2.57] %IPD. Over all 15 subsets inside the pose gate, including two
    # whose yaw spread is under intake's 26deg minimum, the intersection is
    # EMPTY -- but those are sets the product refuses, and the accept/refuse line
    # does no delicate work: refused spreads reach 21.9deg and accepted ones
    # start at 28.7deg, so any threshold in that gap gives the same split.
    #
    # ⛔ SO THE BRACKET IS ONLY HONEST BEHIND INTAKE. Emitting it for a set intake
    # would refuse is exactly the case that was measured to fail. `intake.ready`
    # is recorded beside it and the report must respect it.
    # ⚠️ n=1 subject, and 13 OVERLAPPING subsets of 6 photographs is 6
    # photographs of evidence, not 13 independent tests.
    perm = np.load(MV.MIRROR_PERM)[: MV.N_MESH]
    fit_ipd = F.mesh_ipd(result["shape"])
    asym = MV.asymmetry_interval(
        result["shape"], perm, fit_ipd, result["landmark_var"], yaws
    )
    asym["intake_ready"] = bool(verdict["ready"])
    if not asym["intake_ready"]:
        # The set that reaches here despite intake saying no (min-frames override,
        # or a caller that ignored the verdict) gets the same refusal the pose
        # gate gives, for the same reason: the bracket was not shown to cover.
        asym["valid"] = False
        asym["pose_condition"] = "refused"
        asym["lower_pct_ipd"] = float("nan")
        asym["upper_pct_ipd"] = float("nan")
        asym["note"] = (
            "intake does not accept this set, and the bracket was only shown to "
            "cover on sets it accepts. Refused rather than reported."
        )
    if asym["valid"]:
        print(
            f"  asymmetry bracket      : {asym['lower_pct_ipd']:.2f}"
            f"-{asym['upper_pct_ipd']:.2f}% of IPD"
            f"   (raw {asym['raw_pct_ipd']:.2f}, floor {asym['noise_floor_pct_ipd']:.2f})"
        )
    else:
        print(f"  asymmetry bracket      : REFUSED — {asym['note']}")

    # ---- SCALE-FREE ANGLES on the fitted 3D shape --------------------------
    #
    # The jaw angle is the axis the incumbent competitor leads on, and this
    # project REFUSED it: measured off the profile silhouette the soft-tissue
    # jawline angle ran 98.9-136.2 deg across four frames of one unchanging jaw
    # in a single sitting -- almost the whole 100-140 scale it is judged on --
    # and it does not track head turn, so it could not be corrected either.
    #
    # On the fitted shape it reproduces: measured 2026-08-11 across the 13
    # four-frame subsets of the burst that intake accepts, the mean mandibular
    # contour angle swings 3.4 deg (128.0-131.5), which is 11x better than the
    # silhouette and 9% of the scale. `angle_reproducibility_probe.py`.
    #
    # ⚠️ THE BAR THAT SHIPS IS THE POSTERIOR, NOT THE BURST SPREAD. A new subject
    # has one photo set and no subsets to disagree, so the only per-subject error
    # available is the fit's own. The burst measurement is what says that bar is
    # not understating: posterior SD 1.86 deg against an observed subset SD of
    # 1.07, i.e. conservative -- the safe direction.
    #
    # ⛔ The NASOLABIAL angle is computed and deliberately NOT shipped as a
    # number: it swings 10.8 deg (36% of its scale) on the same unchanging face,
    # and the fit's own posterior for it is 21 deg because the columella-to-
    # subnasale vector is short enough that landmark error dominates its
    # direction. It is kept only because it CORROBORATES: the mesh reads
    # 105.5-116.3 where the silhouette reads 127.9, and `analyze_profile` already
    # documents that silhouette values run ~15-25 deg high.
    import face_angles as FANG

    angles = FANG.all_angles(result["shape"], perm)
    ang_sd = None
    if result.get("landmark_cov") is not None:
        chol = np.linalg.cholesky(result["landmark_cov"] + 1e-12 * np.eye(3)[None])
        rng_a = np.random.default_rng(POSTERIOR_SEED)
        pool: dict[str, list[float]] = {}
        for _ in range(N_POSTERIOR // 2):
            z = rng_a.normal(size=(len(result["shape"]), 3))
            draw = MV.procrustes_to(
                result["shape"] + np.einsum("pij,pj->pi", chol, z),
                MV.load_canonical()[0][: MV.N_MESH],
            )
            for k_a, v_a in FANG.flatten(FANG.all_angles(draw, perm)).items():
                pool.setdefault(k_a, []).append(v_a)
        ang_sd = {
            k_a: round(float(np.std(v_a, ddof=1)), 4) for k_a, v_a in pool.items()
        }

    # BANK NORMS for the ratio family. Proposal trap #5 says detail without norms
    # is data rather than a report, and this is the part of that which CAN be
    # answered: the 243-face bank is 2D landmarks only, which is enough for a
    # frontal-plane ratio and not enough for a millimetre.
    # ⛔ Ratios only. Scoring a fitted subject against raw single-photo references
    # inflates any MAGNITUDE (5.1, measured); a ratio is perturbed symmetrically.
    ratio_pct: dict[str, float | None] = {}
    try:
        _dist = FANG.bank_ratio_distribution()
        for _k, _v in angles.get("ratios", {}).items():
            ratio_pct[_k] = FANG.bank_percentile(_k, _v, _dist)
    except Exception as _exc:  # noqa: BLE001 - norms are additive, never fatal
        print(f"  (bank norms unavailable: {type(_exc).__name__})")

    # ---- SAGITTAL, on the same fitted shape --------------------------------
    # Eight scale-free profile measurements the two-photo path never had: the
    # silhouette gave an outline, and an outline cannot be normed. They rank
    # against FLAME's identity population, and ONLY where the ten real scanned
    # humans corroborate that population for the metric in question -- the
    # corroboration flag is cached with the distribution so this call site cannot
    # accidentally rank against an unchecked norm.
    #
    # ⛔ THE CACHE, NOT THE SCANS. Building the bank reads the licensed D3DFACS
    # registrations; ranking against it needs only the numbers. The shipping
    # pipeline therefore never opens the licensed geometry at all, which is a
    # stronger guarantee than remembering not to.
    import profile_metrics as PM

    sagittal = PM.sagittal_metrics(result["shape"])
    _sag_cache = PM.load_cache()
    if _sag_cache is None:
        print(
            "  (no sagittal bank cache — run `python profile_metrics.py` where the "
            "scans are present; profile measurements will print without a placement)"
        )
    sagittal_pct = {
        k: PM.cached_percentile(k, v, _sag_cache) for k, v in sagittal.items()
    }

    # ---- THE TWO TIERS, joined to this subject's readings -------------------
    # `metric_tiers` decides what each measurement may SAY; this is where that
    # decision meets what this face actually reads. The placement interval is
    # computed here because it needs both the tier's bar and the reference
    # distributions, and a tier-1 metric still loses its placement word when this
    # particular reading's interval crosses a boundary -- the philtrum rule.
    measurements = None
    try:
        import metric_tiers as MT

        _tiers = MT.load()
        _flat = dict(angles.get("ratios", {}))
        _flat.update(sagittal)
        _flat.update(FANG.flatten(angles))

        def _rank(key: str, value: float):
            row = _tiers.get(key, {})
            if row.get("placed_in") == "clinical" and row.get("band"):
                return MT.band_position(value, row["band"][0], row["band"][1])
            if row.get("placed_in") == "sagittal":
                return PM.cached_percentile(key, value, _sag_cache)
            return FANG.bank_percentile(key, value, _dist)

        measurements = MT.measurement_rows(_tiers, _flat, _rank)
        _n = MT.counts({k: v for k, v in measurements.items()})
        print(
            f"  measurements           : {_n['tier1']} placed, {_n['tier2']} "
            f"printed without a placement, {_n['refused']} refused"
        )
    except Exception as _exc:  # noqa: BLE001 - the report refuses, never guesses
        print(
            f"  (!! no measurement tiers: {type(_exc).__name__}: {_exc})\n"
            "     the report will refuse the measurement section rather than "
            "print numbers with no stated basis"
        )
    jaw = angles["mandibular_contour_deg"]
    print(
        f"  mandibular contour     : {jaw['mean']:.1f} deg"
        + (f" +- {ang_sd['mandibular_mean']:.1f}" if ang_sd else "")
        + f"   (L {jaw['left']:.1f} / R {jaw['right']:.1f}, gap {jaw['left_right_gap']:.1f})"
    )

    # ---- POSTERIOR SAMPLES, drawn through the REAL render path -------------
    # Not a per-landmark SD applied independently afterwards. The gauge fix
    # Procrustes-aligns every shape to the canonical model, so a DEPTH error --
    # which is where a few-view fit is weakest -- tilts the whole shape and lands
    # in the rendered x,y as a correlated displacement. Measured on synthetic
    # truth: a 2-degree clustered set moves the rendered landmarks 3.01 px where a
    # 60-degree set moves them 1.54, while the in-plane posterior on its own
    # claims the clustered set is the BETTER determined of the two. Sampling in 3D
    # and pushing each draw through gauge-then-render is the only propagation that
    # sees this, and it is why the samples are written rather than a summary.
    cov = result.get("landmark_cov")
    if cov is None:
        # A basis fit reports only the 3D trace. Refusing is right: publishing a
        # bar known to be ~2.7x too wide is manufacturing a caveat, not measuring.
        print()
        print(
            "!! HALTED: this fit reports no per-landmark covariance (basis fits give "
            "the 3D trace only). Phase B's error bars would be uncalibrated, so they "
            "are refused rather than printed. Nothing was written."
        )
        return 1

    canonical_full, _ = MV.load_canonical()
    canonical_full = canonical_full[: MV.N_MESH]
    ref2d = ref["pts478"][: MV.N_MESH]
    rng = np.random.default_rng(POSTERIOR_SEED)
    # Cholesky per landmark, with a jitter floor: the depth direction can be so
    # poorly determined that the covariance is numerically singular, which is the
    # regime this exists to represent rather than a reason to give up.
    chol = np.linalg.cholesky(cov + 1e-12 * np.eye(3)[None])
    draws = np.empty((N_POSTERIOR, 478, 2))
    for k in range(N_POSTERIOR):
        z = rng.normal(size=(MV.N_MESH, 3))
        sample = result["shape"] + np.einsum("pij,pj->pi", chol, z)
        sample = MV.procrustes_to(sample, canonical_full)  # the gauge, re-applied
        draws[k] = F.with_mesh_iris(MV.frontal_render(sample, ref2d))
    sd_px = draws[:, : MV.N_MESH, :].std(axis=0, ddof=1).mean(axis=1)

    # ---- Orientation guard, before anything is written --------------------
    # The canonical model is Y-UP and pixels are Y-DOWN, so a render that skipped
    # the flip comes out upside down -- and a bilaterally symmetric face looks
    # entirely normal that way to every symmetry check in this project. The cheap
    # test is the one asymmetry that is not left-right: the top of the head must
    # sit above the chin.
    if not full[10, 1] < full[152, 1]:
        print()
        print(
            "!! HALTED: the fitted render is upside down (trichion y="
            f"{full[10, 1]:.1f} is not above menton y={full[152, 1]:.1f}). Nothing "
            "was written. See multiview_probe.frontal_render — the canonical model "
            "is Y-up and image coordinates are Y-down."
        )
        return 1

    # ---- Outputs ----------------------------------------------------------
    with open(OUT_CSV, "w") as fh:
        fh.write("index,x,y\n")
        for i, (x, y) in enumerate(full):
            fh.write(f"{i},{x:.2f},{y:.2f}\n")

    import shutil

    shutil.copyfile(ref["path"], OUT_REF)

    diag = {
        "_what": "per-subject uncertainty from the disagreement between REAL views",
        "_note": (
            "landmark_var is the fit's own per-landmark positional variance, in "
            "shape units. Unlike ensemble_uncertainty's 13 augmentations of one "
            "photograph, these views differ in pose, lighting and expression, so "
            "this is a test-retest spread rather than a detector-jitter floor. "
            "Phase B is what makes the report read from it."
        ),
        "_meta": {
            "set_dir": set_dir,
            "set_sha256": F.set_fingerprint(
                all_paths
            ),  # incl. the lateral — see main()
            "n_submitted": len(paths),
            "n_in_directory": len(all_paths),
            "n_fitted": len(frames),
            "reference_photo": ref["file"],
        },
        "frames": [
            {
                "file": f["file"],
                "yaw_deg": round(f["yaw"], 1),
                "ipd_px": round(f["ipd_px"], 1),
            }
            for f in frames
        ],
        "yaw_spread_deg": round(spread, 1),
        "reprojection_rms_px": round(float(result["rms_px"]), 3),
        "reprojection_history": [round(float(v), 3) for v in result["history"][-5:]],
        "prior_divergence_pct": round(float(drift), 2),
        "landmark_var": [round(float(v), 8) for v in result["landmark_var"]],
        # THE SAME VARIANCE, IN THE UNITS THE REPORT MEASURES IN.
        #
        # `landmark_var` is E[|error vector|^2] in shape units -- the convention
        # symmetry_3d_debiased already uses, where it is compared against a
        # squared 3D distance. Every metric downstream is computed on the 2D
        # render in pixels, so it needs a PER-AXIS SD in THOSE pixels:
        #
        #     per-axis variance = landmark_var / 3   (isotropy assumed)
        #     pixels            = sqrt(that) x render_scale
        #
        # Converted here rather than at the consumer, because a unit conversion
        # done at two call sites is a unit conversion done two slightly different
        # ways. `multiview_probe.render_scale` is the single definition of the
        # factor `frontal_render` applies.
        #
        # IN-PLANE, NOT ISOTROPIC -- this was wrong once and it mattered.
        # The first version spread the 3D trace evenly over three axes. Measured
        # on a 36-degree synthetic set the posterior splits x 5.1%, y 4.2%,
        # z 90.7%: depth is where a few-view fit is weak, by an order of
        # magnitude, and the frontal render never sees it. Isotropy therefore
        # overstated the in-plane SD by 2.67x -- bars so wide the numbers beside
        # them stopped meaning anything. `landmark_var_xy` is the real x,y
        # posterior.
        "landmark_sd_px": [round(float(v), 5) for v in sd_px],
        "_landmark_sd_px_note": (
            "per-landmark positional SD, per axis, in the pixel frame of "
            "landmarks_frontal.csv. Phase B propagates these through the metrics "
            "to give error bars from the disagreement between REAL photographs."
        ),
        # PHASE C. `valid` is the whole contract: the report must render a
        # refusal when it is False and must never fall back to a point estimate.
        "asymmetry_bracket": {
            k: (None if isinstance(v, float) and np.isnan(v) else v)
            for k, v in asym.items()
        },
        # SCALE-FREE ANGLES. `shippable` is the contract the report reads: an
        # angle whose measured swing on an unchanging face is a large fraction of
        # the scale it is judged on does not get printed as a number, however
        # much better it is than the instrument it replaces.
        "scale_free_angles": {
            "mandibular_contour_deg": angles["mandibular_contour_deg"],
            "nasolabial_deg": angles["nasolabial_deg"],
            "posterior_sd_deg": ang_sd,
            "shippable": {"mandibular_contour": True, "nasolabial": False},
            # RATIOS, gated 2026-08-12 on THREE sittings against the bank's own
            # p5-p95 span. shippable comes from face_angles.RATIO_SHIPPABLE so
            # the verdict lives in ONE place; a refused ratio still travels here
            # WITH its value, because the report renders a refusal rather than
            # omitting the row -- a number that vanishes looks like one that was
            # never measured.
            "ratios": angles.get("ratios", {}),
            "ratio_bank_percentile": ratio_pct,
            "ratio_shippable": dict(FANG.RATIO_SHIPPABLE),
            # SAGITTAL. Values and ranks travel; the ship/refuse decision does
            # NOT live here. `metric_tiers.py` is the single authority on what
            # each measurement has earned the right to say, so a verdict written
            # into this file too would be a second place to change it -- and the
            # two would drift silently, because nothing reads them together.
            "sagittal": sagittal,
            "sagittal_bank_percentile": sagittal_pct,
            # THE TWO-TIER TABLE THE REPORT RENDERS. None means the tiers could
            # not be computed, and the report must render a refusal for the whole
            # section rather than fall back to printing numbers whose basis it
            # cannot state -- the same contract asymmetry_bracket.valid carries.
            "measurements": measurements,
            "_sagittal_note": (
                "eight scale-free profile measurements taken on the fitted shape. "
                "Ranked against FLAME's identity population, and only where the "
                "ten real scanned humans corroborate that population for the "
                "metric. NOT a rank against a demographically matched human "
                "population — it carries none of the ethnic matching the frontal "
                "bank attempts."
            ),
            "_ratio_note": (
                "Judged on cross-sitting reproducibility against the range 90% of the "
                "243 reference faces occupy: facial thirds 3%, fwhr 13%, mouth:nose 22% "
                "(closest to the line, and anchored on expression-mobile mouth corners), "
                "intercanthal 28% REFUSED. The 26-second burst scores intercanthal at 16% "
                "and would have shipped it -- a within-sitting figure is a floor, not a "
                "verdict, and it has now understated three separate quantities."
            ),
            "_nasolabial_note": (
                "measured on the fitted mesh and NOT shipped as a number: it "
                "swings 10.8 deg (36% of its scale) on one unchanging face and "
                "its own posterior is ~21 deg, because the columella-to-subnasale "
                "vector is short enough that landmark error dominates its "
                "direction. Kept because it corroborates the silhouette figure's "
                "documented ~15-25 deg offset."
            ),
        },
        "intake": {
            "ready": verdict["ready"],
            "coverage": verdict["coverage"],
            "advice": verdict["advice"],
        },
    }
    # ⭐ CARRY THE DEGRADATION FORWARD. A report produced from a below-floor set
    # must be able to say so on every bracket, so the penalty travels in the
    # diagnostics rather than being printed once to a terminal nobody kept.
    if degraded is not None:
        diag["degraded"] = degraded
    with open(OUT_DIAG, "w") as fh:
        json.dump(diag, fh, indent=2)
    # The DRAWS themselves, not a summary. A per-landmark SD applied
    # independently afterwards would throw away the correlation the gauge
    # introduces, which is the whole reason a clustered set reads wider.
    np.savez_compressed(
        OUT_POSTERIOR, draws=draws, set_sha256=np.array(diag["_meta"]["set_sha256"])
    )

    print()
    print(f"  wrote {F.GEOMETRY_CSV} (478 rows, pose-free), fit_diagnostics.json")
    print(
        f"  wrote fit_posterior.npz — {N_POSTERIOR} draws, median SD "
        f"{np.median(sd_px):.2f}px = {100 * np.median(sd_px) / F.mesh_ipd(full):.2f}% of IPD"
    )
    print(f"  wrote frontal.png <- {ref['file']}, the appearance source")
    return 0


if __name__ == "__main__":
    sys.exit(main())
