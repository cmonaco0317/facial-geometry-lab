"""Literature-faithful averageness (closeness to the average face).

Averageness, as defined in the attractiveness literature (Langlois & Roggman
1990 and successors), is HOW CLOSE a face's full shape is to the average of a
matched population. This module computes exactly that, with one genuine,
leakage-removing improvement kept:

  PRIMARY (the headline). Full-shape generalized Procrustes alignment to the
  matched population mean, then the Euclidean Procrustes distance from each face
  to the mean. Smaller distance = more average. The percentile is the fraction
  of reference faces FARTHER from the mean than the subject is.

  Leave-one-out (the kept improvement). The old equal-weighted method scored
  each reference face against a mean it helped build (home-field advantage) while
  the subject was always held out — not apples-to-apples. Here every reference
  face i is scored against the mean of the OTHER n-1 faces, mirroring how the
  subject (a permanent hold-out) is scored against the full-n mean. On this data
  LOO barely moves the primary set (white_20_29: 72.2 -> 72.2) but matters on
  smaller/looser sets (white_10_19: 65.1 -> 69.8), so it is kept.

  Bootstrap CI. A 95% interval on the percentile by resampling the LOO distance
  pool (B=3000).

SECONDARY (disclosure only). The SAME leave-one-out Euclidean percentile but
computed on the SYMMETRIC component of each shape (left/right averaged via an
exact mirror map from MediaPipe's canonical face model). This "shape-typicality
with symmetry removed" is a DIFFERENT construct from averageness and lands much
lower (~42% here); it is reported only as `symmetric_typicality_pct` to show how
much of the averageness signal overlaps with symmetry, which Part C already
credits. It is NOT the headline.

WHAT WAS TRIED AND REVERTED. An earlier rewrite made the headline a
Mahalanobis-in-PCA, chi-square-calibrated score on the symmetric component only.
A diagnostic on the real data proved this was an artifact: it measured a
different construct (symmetry-stripped typicality, ~42%) rather than literature
averageness (~72%), it dragged the percentile down, and the Mahalanobis
percentile was unstable in the choice of k (number of PCs). It is removed from
the primary path. The headline is back to literature-faithful averageness.

HONESTY: shape-only, weak effect (r~.3-.5); report a band, not a point. The
pose-gated FairFace reference and the guardrail (subject extracted with the SAME
plain detector from frontal.png, never the refined landmarks_frontal.csv) are
unchanged.

Outputs: averageness_v2.json, averageness_v2.png
"""

import glob
import json
import os

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision

import analyze_advanced as AA  # per-metric pose tolerances live here
import averageness_band as AB  # posture gate, axis bands, the refusal rule

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "face_landmarker.task")
CACHE = os.path.join(HERE, "ref_landmarks2.npz")
REFDIR = os.path.join(HERE, "reference_faces2")
# Exact left<->right landmark permutation derived once from MediaPipe's
# canonical_face_model.obj (symmetric by construction). Mirroring a shape =
# apply this permutation to the rows AND negate the x coordinate. Verified an
# exact involution with zero 3D mirror residual; 28 midline points map to self.
MIRROR_PERM_FILE = os.path.join(HERE, "mirror_perm.npy")


# ⛔ DERIVED FROM THE CACHE, NOT HARDCODED. This was a fixed list of the three
# male banks, so the female and broader-age banks added on 2026-08-15 would have
# been fetched, cached, and then silently never scored against — the subject
# would keep being compared to young men while a matched bank sat unused on disk.
def _available_sets() -> list[str]:
    # ⛔ THE CACHE FIRST, THEN THE DIRECTORIES — and the order matters because
    # `build_cache` iterates this list to decide what to landmark. A version that
    # only read the cache was circular: with no cache it fell back to three
    # hardcoded names, so a rebuild would quietly landmark three of the eight
    # downloaded cohorts and purge the rest of the photographs afterwards.
    if os.path.exists(CACHE):
        return sorted(
            k
            for k in np.load(CACHE).files
            if not k.endswith("_raw") and not k.endswith("_race")
        )
    if os.path.isdir(REFDIR):
        found = sorted(
            os.path.basename(d.rstrip("/"))
            for d in glob.glob(os.path.join(REFDIR, "*"))
            if os.path.isdir(d)
        )
        if found:
            return found
    return ["white_10_19", "white_20_29", "broad_10_19"]


SETS = _available_sets()

BOOTSTRAP_B = 3000  # bootstrap resamples for the percentile CI

MIDLINE = [
    10,
    151,
    9,
    8,
    168,
    6,
    197,
    195,
    5,
    4,
    1,
    19,
    94,
    2,
    164,
    0,
    13,
    14,
    17,
    18,
    200,
    199,
    175,
    152,
]
REGION = {
    "brow/forehead": [10, 151, 9, 8, 107, 336, 105, 334, 66, 296, 55, 285],
    "eyes": [33, 263, 133, 362, 159, 386, 145, 374, 157, 384],
    "nose": [1, 4, 2, 6, 168, 129, 358, 98, 327, 5, 195],
    "mouth/lips": [0, 17, 13, 14, 61, 291, 40, 270, 84, 314],
    "cheeks": [234, 454, 116, 345, 50, 280, 101, 330, 205, 425],
    "jaw/chin": [152, 172, 397, 136, 365, 150, 379, 200, 199, 175],
}


# ---- landmarking (UNCHANGED) ----------------------------------------------
def make_landmarker(blendshapes: bool = False):
    """The BANK'S OWN plain detector. `blendshapes` is additive and off by default.

    ⛔ The landmark path must not change. This function is what SS5.1's guardrail
    rests on -- the subject is re-detected here with exactly the detector the 243
    reference faces were cached with, so that a refined subject is never scored
    against a raw bank. Turning blendshapes on adds an OUTPUT; it does not alter
    the landmarks, and `expression_of` exists so the caller reads that output
    instead of being tempted to swap in a different detector to get it.
    """
    opts = vision.FaceLandmarkerOptions(
        base_options=mpt.BaseOptions(model_asset_path=MODEL),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3,
        output_facial_transformation_matrixes=True,
        output_face_blendshapes=blendshapes,
    )
    return vision.FaceLandmarker.create_from_options(opts)


def pose_angles(matrix):
    R = np.array(matrix)[:3, :3]
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    pitch = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    yaw = np.degrees(np.arctan2(-R[2, 0], sy))
    roll = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    return pitch, yaw, roll


def landmarks_for(landmarker, path):
    img = cv2.imread(path)
    if img is None:
        return None, None
    h, w = img.shape[:2]
    res = landmarker.detect(
        mp.Image(
            image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        )
    )
    if not res.face_landmarks or not res.facial_transformation_matrixes:
        return None, None
    L = np.array(
        [[p.x * w, p.y * h] for p in res.face_landmarks[0][:468]], dtype=np.float64
    )
    return L, pose_angles(res.facial_transformation_matrixes[0])


def is_frontal(pitch, yaw, roll):
    return abs(yaw) <= 14 and abs(pitch) <= 16 and abs(roll) <= 25


def landmarks_and_expression(landmarker, path):
    """Landmarks, pose, and the blendshape scores, from ONE detection.

    Same detector and same landmarks as `landmarks_for`; the blendshapes come
    from the same pass so the expression reading and the shape reading can never
    describe two different detections of the same photograph.
    """
    img = cv2.imread(path)
    if img is None:
        return None, None, None
    h, w = img.shape[:2]
    res = landmarker.detect(
        mp.Image(
            image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        )
    )
    if not res.face_landmarks or not res.facial_transformation_matrixes:
        return None, None, None
    lms = np.array(
        [[p.x * w, p.y * h] for p in res.face_landmarks[0][:468]], dtype=np.float64
    )
    expr = (
        {c.category_name: float(c.score) for c in res.face_blendshapes[0]}
        if getattr(res, "face_blendshapes", None)
        else {}
    )
    return lms, pose_angles(res.facial_transformation_matrixes[0]), expr


def _manifest_labels(name: str) -> dict:
    """basename -> FairFace race index, from the set's manifest.

    Tolerates both manifest formats: format 1 is a bare list of entries,
    format 2 wraps them in an object with collection metadata.
    """
    path = os.path.join(REFDIR, f"{name}_manifest.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    entries = doc["entries"] if isinstance(doc, dict) else doc
    return {os.path.basename(e["file"]): int(e["race"]) for e in entries if "race" in e}


def build_cache():
    if os.path.exists(CACHE):
        d = np.load(CACHE)
        return {k: d[k] for k in d.files}
    lmk = make_landmarker()
    out = {}
    for name in SETS:
        files = sorted(glob.glob(os.path.join(REFDIR, name, "*.jpg")))
        labels = _manifest_labels(name)
        kept, raw, kept_race = [], 0, []
        for f in files:
            L, ang = landmarks_for(lmk, f)
            if L is None or L.shape != (468, 2):
                continue
            raw += 1
            if is_frontal(*ang):
                kept.append(L)
                # PROVENANCE, recorded at the only moment it is knowable.
                # The frontal filter drops faces, so the mapping from cached
                # landmark back to its manifest row is lost the instant this
                # loop ends -- and the photographs are deleted straight after,
                # so it cannot be recovered later. Without this the cache cannot
                # be partitioned by demographic, which is exactly what a
                # phenotype-fairness check needs. -1 means "no manifest row".
                kept_race.append(labels.get(os.path.basename(f), -1))
        out[name] = np.array(kept)
        out[name + "_raw"] = np.array([raw])
        out[name + "_race"] = np.array(kept_race, dtype=np.int64)
        print(
            f"  {name}: {len(kept)} frontal kept / {raw} detected / {len(files)} images",
            flush=True,
        )
    np.savez(CACHE, **out)
    purge_reference_images()
    return out


def purge_reference_images(refdir: str = None, dry_run: bool = False) -> dict:
    """Delete the downloaded FairFace photographs once landmarks are cached.

    THIS CLOSES AN ASSERTED-BUT-UNENFORCED PRIVACY GUARANTEE. Both README.md and
    `fetch_reference2.py`'s docstring have always said the reference images are
    deleted after landmarking, and until now nothing did it -- a project-wide
    grep for rmtree/os.remove/unlink returned nothing, so it was done by hand.
    Anyone following the documented reuse recipe re-downloaded hundreds of
    photographs of real people with nothing to clean them up.

    A promise about other people's biometric data that the code does not keep is
    a defect while the project is private and a compliance finding once money
    changes hands (BACKLOG Phase 7).

    Only `.jpg` files under the reference directory are removed. The manifests
    stay: they are the provenance record, and CC-BY-4.0 attribution depends on
    them. Landmarks in the cache are derived geometry, not photographs, and are
    kept deliberately -- they are what the pipeline needs and they are already
    excluded from git.
    """
    refdir = refdir or REFDIR
    removed, freed = 0, 0
    if os.path.isdir(refdir):
        for path in glob.glob(os.path.join(refdir, "*", "*.jpg")):
            try:
                freed += os.path.getsize(path)
                if not dry_run:
                    os.remove(path)
                removed += 1
            except OSError as exc:  # report, never silently leave a photo behind
                print(f"  WARNING could not delete {path}: {exc}", flush=True)
    verb = "would delete" if dry_run else "deleted"
    print(
        f"  privacy: {verb} {removed} reference photograph(s) "
        f"({freed / 1e6:.1f} MB); landmark geometry retained",
        flush=True,
    )
    return {"removed": removed, "bytes_freed": freed, "dry_run": dry_run}


# ---- Procrustes (rotation only, UNCHANGED) --------------------------------
def standardize(X):
    X = X - X.mean(0)
    n = np.linalg.norm(X)
    return X / n if n > 0 else X


def align(X, ref):
    M = X.T @ ref
    U, _, Vt = np.linalg.svd(M)
    d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1.0, d]) @ Vt
    return X @ R


def gpa(configs, iters=10):
    S = [standardize(c) for c in configs]
    mean = S[0].copy()
    for _ in range(iters):
        A = [align(c, mean) for c in S]
        nm = standardize(np.mean(A, axis=0))
        if np.linalg.norm(nm - mean) < 1e-9:
            mean, S = nm, A
            break
        mean, S = nm, A
    return mean, np.array(S)


# ---- symmetric / asymmetric decomposition (W3) ----------------------------
def load_mirror_perm():
    if not os.path.exists(MIRROR_PERM_FILE):
        raise RuntimeError(
            "averageness: missing mirror_perm.npy (exact L/R map from canonical "
            "face model). Regenerate it before scoring."
        )
    perm = np.load(MIRROR_PERM_FILE).astype(int)
    if perm.shape != (468,) or not np.array_equal(perm[perm], np.arange(468)):
        raise RuntimeError("averageness: mirror_perm.npy is not a valid involution")
    return perm


def mirror(X, perm):
    """Reflect a 2D shape: swap L/R rows, negate x about the centroid."""
    M = X[perm].copy()
    M[:, 0] = -(M[:, 0] - X[:, 0].mean()) + X[:, 0].mean()
    return M


def symmetric_part(X, perm):
    return 0.5 * (X + mirror(X, perm))


# ---- helpers ----------------------------------------------------------------
def aligned_sym_vec(shape, target_mean, perm):
    """Align a raw shape to target_mean, take symmetric part, flatten."""
    a = align(standardize(shape), target_mean)
    return symmetric_part(a, perm).reshape(-1)


# ---- per-set analysis ------------------------------------------------------
def loo_euclidean_distances(configs, perm=None):
    """Leave-one-out Euclidean Procrustes distance to the mean.

    For each face i, build the mean of the OTHER n-1 faces (GPA), align face i to
    that mean, and take the Euclidean distance. If `perm` is given, the distance
    is measured on the SYMMETRIC component of the shapes (left/right averaged);
    otherwise on the full shape.
    """
    n = len(configs)
    loo_d = np.empty(n)
    for i in range(n):
        others = configs[:i] + configs[i + 1 :]
        m_i, _ = gpa(others)
        if perm is None:
            a_i = align(standardize(configs[i]), m_i)
            loo_d[i] = float(np.linalg.norm(a_i - m_i))
        else:
            sm_i = standardize(symmetric_part(m_i, perm))
            v_i = aligned_sym_vec(configs[i], sm_i, perm)
            loo_d[i] = float(np.linalg.norm(v_i - sm_i.reshape(-1)))
    return loo_d


def analyze_set(subject, configs, perm):
    configs = list(configs)
    n = len(configs)

    # ---- PRIMARY: literature averageness on the FULL shape ----------------
    # Full-shape GPA -> matched mean. Subject aligned to that mean.
    mean_full, _ = gpa(configs)
    subject_aligned = align(standardize(subject), mean_full)
    subject_d = float(np.linalg.norm(subject_aligned - mean_full))

    # Leave-one-out Euclidean distance for every reference face (full shape).
    loo_d = loo_euclidean_distances(configs, perm=None)

    # Percentile: fraction of reference faces FARTHER from the mean than subject.
    percentile = float(100.0 * np.mean(loo_d > subject_d))
    z = float((subject_d - loo_d.mean()) / (loo_d.std() + 1e-12))

    # Bootstrap 95% CI by resampling the LOO distance pool.
    rng = np.random.default_rng(0)
    boots = np.empty(BOOTSTRAP_B)
    for b in range(BOOTSTRAP_B):
        samp = loo_d[rng.integers(0, n, n)]
        boots[b] = 100.0 * np.mean(samp > subject_d)
    ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]

    # ---- SECONDARY (disclosure only): same LOO Euclidean percentile but on
    # the SYMMETRIC component. A DIFFERENT construct ("shape-typicality with
    # symmetry removed"); shows the overlap with Part C, not the headline.
    sym_mean = standardize(symmetric_part(mean_full, perm))
    subject_sym_vec = aligned_sym_vec(subject, sym_mean, perm)
    subject_sym_d = float(np.linalg.norm(subject_sym_vec - sym_mean.reshape(-1)))
    loo_sym_d = loo_euclidean_distances(configs, perm=perm)
    sym_pct = float(100.0 * np.mean(loo_sym_d > subject_sym_d))
    # The symmetric figure shipped without an interval for as long as it was a
    # footnote. Measured 2026-08-04 it is the MORE reliable of the two -- ~10x
    # less sensitive to the reference bank's pose and ~6x less to the subject's
    # own head turn -- so it is being promoted to primary, and a primary figure
    # does not ship without its sampling error. Same bootstrap, same seed, same
    # resampling of the LOO pool as the full-shape interval, so the two remain
    # directly comparable.
    rng_sym = np.random.default_rng(0)
    boots_sym = np.empty(BOOTSTRAP_B)
    for b in range(BOOTSTRAP_B):
        samp_s = loo_sym_d[rng_sym.integers(0, n, n)]
        boots_sym[b] = 100.0 * np.mean(samp_s > subject_sym_d)
    sym_ci = [
        float(np.percentile(boots_sym, 2.5)),
        float(np.percentile(boots_sym, 97.5)),
    ]
    sym_z = float((subject_sym_d - loo_sym_d.mean()) / (loo_sym_d.std() + 1e-12))

    # ---- region relative deviation on the FULL shape (as before) ----------
    dev = np.linalg.norm(subject_aligned - mean_full, axis=1)
    ref_dev = np.array(
        [
            np.linalg.norm(align(standardize(c), mean_full) - mean_full, axis=1)
            for c in configs
        ]
    )
    samp = ref_dev.mean(0)
    region = {
        r: round(
            float(
                np.mean(dev[[i for i in idx if i < 468]])
                / (np.mean(samp[[i for i in idx if i < 468]]) + 1e-9)
            ),
            2,
        )
        for r, idx in REGION.items()
    }

    return {
        # --- PRIMARY, since 2026-08-04: the symmetric component ---------------
        # Promoted on measurement, not preference. Across a reference-gate sweep
        # the full-shape figure moved 32.4 percentile points where this moved
        # 3.4; across a 26-second burst of one unchanging face inside the pose
        # gate, 66.7 points where this moved 11.1. Mirror-averaging cancels the
        # antisymmetric part of a yaw distortion, which is the mechanism.
        "n": int(n),
        "symmetric_typicality_pct": round(sym_pct, 1),
        "symmetric_ci": [round(sym_ci[0], 1), round(sym_ci[1], 1)],
        "symmetric_z": round(sym_z, 2),
        # --- SECONDARY, disclosed as fragile: full-shape averageness ----------
        # Kept because it IS the literature construct and removing it would hide
        # the comparison rather than make it. Every renderer must carry its
        # measured sensitivity beside it -- see
        # reference_values repeatability.averageness_construct_swing_pp.
        "percentile": round(percentile, 1),
        "ci": [round(ci[0], 1), round(ci[1], 1)],
        "z_score": round(z, 2),
        "region_rel_deviation": dict(sorted(region.items(), key=lambda kv: -kv[1])),
        "method_detail": (
            "PRIMARY symmetric_typicality_pct: LOO Euclidean Procrustes closeness to "
            "the matched mean measured on the SYMMETRIC COMPONENT (left/right averaged), "
            "bootstrap CI. Promoted 2026-08-04 on measured stability: 3.4 vs 32.4 "
            "percentile points across a reference-pose-gate sweep, 11.1 vs 66.7 across "
            "a constant-face burst inside the pose gate. SECONDARY percentile: the same "
            "statistic on the FULL shape -- the literature-faithful averageness "
            "construct, retained for comparison and disclosed as the fragile one. The "
            "two are different constructs: the full-shape figure counts asymmetry, "
            "which on a single photograph is not separable from head turn."
        ),
        # --- viz handles (stripped before JSON) ---
        "_mean": mean_full,
        "_subject_aligned": subject_aligned,
        "_loo_d": loo_d,
        "_subject_d": subject_d,
    }


def subject_cohort() -> str:
    """The bank this subject is compared against, from subject.json.

    Read here rather than passed in, because the cohort decides which banks are
    even scored -- and a default that silently means "male" is the defect this
    whole family mechanism exists to remove.
    """
    import subject as _subject

    return _subject.load()["reference_set"]


def load_subject():
    # GUARDRAIL (unchanged): re-extract the SUBJECT with the SAME plain detector
    # as the reference. Do NOT read landmarks_frontal.csv (refined / sub-pixel),
    # which would create a refined-subject vs raw-reference mismatch.
    lmk = make_landmarker()
    try:
        L, _ = landmarks_for(lmk, os.path.join(HERE, "frontal.png"))
    finally:
        lmk.close()
    if L is None or L.shape != (468, 2):
        raise RuntimeError("averageness: failed to detect subject (plain detector)")
    return L


def load_subject_expression():
    """The subject's blendshapes, from the SAME detector and the SAME photograph.

    Separate from `load_subject` so that function's contract — and with it the
    §5.1 guardrail every caller depends on — is untouched. This exists because
    the mouth-posture gate needs an expression reading, and the one thing that
    must not happen is a second, different detector being introduced to get it.
    """
    lmk = make_landmarker(blendshapes=True)
    try:
        _lms, _angles, expr = landmarks_and_expression(
            lmk, os.path.join(HERE, "frontal.png")
        )
    finally:
        lmk.close()
    return expr or {}


def _race_banks() -> dict:
    """The 42 race x sex x age cohorts, from their own caches.

    ⛔ KEPT OUT OF ref_landmarks2.npz ON PURPOSE. catalogue_bank_probe pools every
    bank in that file, so adding these to it would silently redefine the 825-face
    population every published percentile, anchor-coverage figure and claim was
    measured against.
    """
    out = {}
    male = os.path.join(HERE, "fairness_landmarks.npz")
    if os.path.exists(male):
        z = np.load(male, allow_pickle=False)
        for k in z.files:
            out[f"m_{k}_20_29"] = z[k]
    race = os.path.join(HERE, "race_landmarks.npz")
    if os.path.exists(race):
        z = np.load(race, allow_pickle=False)
        for k in z.files:
            out[k] = z[k]
    return out


def main():
    perm = load_mirror_perm()
    subject = load_subject()
    cache = build_cache()
    cfgs = {name: cache[name] for name in SETS}
    # ⭐ THE RACE+SEX+AGE MATCHED BANKS, added alongside the legacy eight. They
    # live in their own caches so ref_landmarks2.npz — and every figure and claim
    # measured against its 825 faces — stays exactly as it was.
    cfgs.update(_race_banks())
    # WHICH BANKS THIS SUBJECT IS SCORED AGAINST, derived from their own cohort
    # rather than hardcoded. The hardcoded version listed the four male banks and
    # survived the rebuild that added the female ones, so a female subject's own
    # cohort was never scored -- see averageness_band.cohort_family.
    fam = AB.cohort_family(subject_cohort(), sorted(cfgs))
    try:
        import subject as _subj

        _rc, _why = _subj.race_cohort()
    except Exception:  # noqa: BLE001
        _rc, _why = None, "subject configuration unavailable"
    fam = AB.race_matched_family(fam, _rc, sorted(cfgs))
    if _rc and fam.get("race_matched"):
        print(f"  race-matched typicality bank: {_rc} (n={len(cfgs[_rc])})")
    else:
        print(f"  no race-matched typicality bank: {_why or 'none available'}")
    if fam["pooled"]:
        cfgs[fam["pooled"]] = np.concatenate(
            [cache[n] for n in fam["pooled_from"]], axis=0
        )

    # THE MOUTH-POSTURE GATE. Shape typicality was reading the subject's lips:
    # mouthPucker correlates -0.92 with the dominant shape axis and +0.91 with
    # the percentile itself across the reference subject's in-gate frontals, and
    # the readings clustered by sitting rather than scattering as noise would.
    # See averageness_posture_probe.py. Gating is exactly what the pipeline
    # already does for head pose, for the same reason.
    expr = load_subject_expression()
    posture_ok, posture_note = AB.posture_ok(expr)
    print(f"  mouth posture: {'PASS' if posture_ok else 'REFUSED'} — {posture_note}")

    order = fam["order"]
    res, viz = {}, {}
    for name in order:
        repro = AB.axis_repro_for(name)
        r = analyze_set(subject, cfgs[name], perm)
        ctx = AB.bank_context(cfgs[name], perm)
        rd = AB.read(subject, ctx, n_axes=len(repro))
        names = AB.name_axes(list(cfgs[name]), ctx, n_axes=len(repro))
        axes = []
        for a, (floor, swing, share) in enumerate(repro):
            ships, frac, why = AB.axis_ships(share, floor, swing)
            nm = names[a]
            z = float(rd["axes"][a])
            axes.append(
                {
                    "axis": a + 1,
                    # oriented so "+" means MORE of the named thing; an unnamed
                    # axis keeps the deterministic SVD sign
                    "z": round(-z if nm["flip_for_name"] else z, 2),
                    "band_sd": round(max(floor, swing), 2),
                    "var_share": share,
                    "band_frac_of_scale": round(frac, 3),
                    "ships": bool(ships),
                    "why": why,
                    "name": nm["name"],
                    "name_r": nm["r"],
                    "name_runner_up": nm["runner_up"],
                    "name_runner_up_r": nm["runner_up_r"],
                }
            )
        # WHAT RESOLVES IS THE PLACEMENT WORD, NOT THE NUMBER — and the
        # distinction is the whole of the 2026-08-15 change. Measured each run
        # rather than hardcoded: if a subject's percentile ever sat clear of every
        # placement cutoff by more than its own measured movement, this resolves
        # instead of refusing out of habit. On the reference subject, post-gate,
        # it does not — the gated swing is 7.4-13.9pp against cutoffs 1.5pp away.
        #
        # ⭐ THE NUMBER STILL SHIPS, AS AN INTERVAL. Until 2026-08-15 a
        # non-resolving cohort withheld the percentile entirely, because the
        # report had only two states: a figure with a verdict, or nothing. The
        # two-tier rule adopted on 2026-08-14 created the missing middle — print
        # the measurement and its measured bar, claim no placement — and
        # typicality is exactly what that middle was built for. Withholding a
        # number whose movement is KNOWN and printable was the binary regime
        # talking, not the evidence.
        # ⛔ A BANK WITH NO MEASURED MOVEMENT INHERITS THE WORST, and the report
        # says which it got. KeyError'd on every cohort added after 2026-08-12.
        swing_pp = AB.gated_swing_for(name)
        resolves = AB.percentile_resolves(swing_pp, r["symmetric_typicality_pct"])
        lo, hi = AB.typicality_interval(r["symmetric_typicality_pct"], swing_pp)
        r["axes"] = axes
        r["typicality_resolves"] = bool(resolves and posture_ok)
        r["gated_swing_pp"] = swing_pp
        r["gated_swing_is_measured"] = AB.swing_is_measured_for(name)
        # ⛔ THE INTERVAL THE REPORT MUST PRINT — the subject's own measured
        # movement, NOT `ci`/`symmetric_ci`, which are bootstrap intervals over
        # the reference pool and are narrower than this by construction.
        r["typicality_interval"] = [round(lo, 1), round(hi, 1)]
        r["typicality_interval_is_movement_not_sampling"] = True
        viz[name] = r
        res[name] = {k: v for k, v in r.items() if not k.startswith("_")}
        print(
            f"[{name}] n={r['n']}  pct={r['percentile']:.0f}% "
            f"(literature averageness, full-shape LOO; CI "
            f"{r['ci'][0]:.0f}-{r['ci'][1]:.0f}%)  "
            f"z={r['z_score']:+.2f}  "
            f"symmetric_typicality={r['symmetric_typicality_pct']:.0f}%",
            flush=True,
        )

    # Pose condition. Head yaw is measured once in stage 1 and carried in
    # metrics_frontal.json. This percentile turned out to be MORE pose-fragile than
    # symmetry (measured 2026-08-01: 4.2%-84.7% across five near-frontal frames of a
    # constant face), and this module's own is_frontal() filter -- which gates the
    # REFERENCE population at |yaw| <= 14 deg -- is far too loose to qualify the
    # subject: a 12.9 deg frame passes it and reads 71 points low.
    #
    # 🔴 THE POSE THIS GATE NEEDS IS THE PHOTOGRAPH'S, NOT THE GEOMETRY'S.
    # This module deliberately re-detects the subject from frontal.png rather than
    # reading the geometry file, precisely so subject and reference bank are the
    # same KIND of object -- both raw single-photo landmarks. That is what keeps
    # MULTIVIEW-WIRING §5.1 from biting: a pose-free fitted subject scored against
    # a raw bank would be measuring how much cleaner the subject's landmarks are.
    # But in the set path `pose.yaw_deg` is 0 by construction (pose was solved and
    # removed from the FIT), while this stage measured a casual photograph that is
    # not at 0. Reading the fit's yaw here would tell the gate the subject is
    # perfectly frontal when it is not -- the flattering direction, self-inflicted.
    # `pose.photo_pose` carries the photograph's own pose whenever they differ.
    try:
        with open(os.path.join(HERE, "metrics_frontal.json")) as f:
            _yaw = AA.F.photograph_pose(json.load(f).get("pose")).get("yaw_deg")
    except (OSError, ValueError):
        _yaw = None
    pose_cond = AA.pose_condition(_yaw, "averageness")
    print(f"  pose condition: {pose_cond['status'].upper()} — {pose_cond['note']}")

    raw = {name: int(cache[name + "_raw"][0]) for name in SETS}
    with open(os.path.join(HERE, "averageness_v2.json"), "w") as f:
        json.dump(
            {
                "method": "pose-QC frontal FairFace reference; rotation-only GPA; "
                "FULL-shape Euclidean Procrustes distance to the matched mean "
                "(literature-faithful averageness: closeness to the average face); "
                "leave-one-out so reference faces and the held-out subject are "
                "scored comparably; bootstrap 95% CI. Secondary "
                "symmetric_typicality_pct is the SAME LOO percentile on the "
                "symmetric component only (a different construct, symmetry stripped) "
                "reported to show overlap with Part C's symmetry score. Shape-only, "
                "weak effect (r~.3-.5), band not point. A Mahalanobis/symmetric-only "
                "variant was tried and reverted: it changed the construct (~42% vs "
                "~72%) and the Mahalanobis percentile was unstable in k. "
                "RESOLUTION MISMATCH — CHECKED, NOT ASSUMED: the subject is "
                "landmarked at 2062x2886 (face ~1406 px wide) while the FairFace "
                "references are 448 px crops (face ~153 px), a 9.2x linear "
                "difference, and low-resolution detections are noisier. Because "
                "scale is normalised that noise does not cancel, so the concern was "
                "that it inflates the reference distance pool and pushes the subject "
                "up. Measured directly by downsampling the subject to the reference "
                "face scale and re-scoring with this same code: the primary "
                "(white_20_29) percentile moved 0.0 pp, white_all 0.7 pp, "
                "white_10_19 4.7 pp. The headline number is unaffected and every "
                "shift is far inside the bootstrap CI, because a rank percentile "
                "over n=72 moves in ~1.4 pp steps. Caveat on the check itself: it "
                "equalises the two sides at LOW resolution; equalising at HIGH "
                "resolution is impossible because the FairFace source images were "
                "deleted after landmarking.",
                "pose_condition": pose_cond,
                "pose_fragility_measured": (
                    "MEASURED 2026-08-01, captures/2026-08-01_pose-burst. Across 5 "
                    "near-frontal frames of a CONSTANT face (26-second sitting) this "
                    "percentile ran 4.2% to 84.7% on the primary cohort — 80.5 points "
                    "of a 100-point scale, CV 107%, a LARGER collapse than bilateral "
                    "symmetry suffers on the same frames. Monotone in yaw: 84.7% at "
                    "0.4 deg, 84.7% at 1.2 deg, 62.5% at 4.5 deg, 13.9% at 12.9 deg, "
                    "~0% past 20 deg. is_frontal() (|yaw| <= 14) does NOT protect a "
                    "subject reading — the 12.9 deg frame passes it. The probe was "
                    "validated by reproducing all four shipped percentiles exactly "
                    "from frontal.png before any burst frame was scored. Limits: one "
                    "subject, n=5, and the burst is ~2.2x lower face resolution than "
                    "the baseline photograph, so it cannot be differenced against "
                    "these numbers directly."
                ),
                "raw_detected": raw,
                # WHICH BANKS THIS SUBJECT WAS ACTUALLY SCORED AGAINST, written
                # down rather than re-derived downstream. build_report used to
                # name `white_all` and `broad_10_19` as literals, which is a
                # second place the male banks were wired in by hand.
                "_family": {
                    **{k: v for k, v in fam.items()},
                    "labels": {n: AB.label_for(n) for n in fam["order"]},
                    "axis_repro_measured": {
                        n: AB.axis_repro_is_measured_for(n) for n in fam["order"]
                    },
                    "swing_measured": {
                        n: AB.swing_is_measured_for(n) for n in fam["order"]
                    },
                },
                **res,
            },
            f,
            indent=2,
        )

    # ---- figure --------------------------------------------------------
    fig = plt.figure(figsize=(17, 6))

    # Panel 1: subject vs full-shape matched-average shape.
    ax0 = fig.add_subplot(1, 3, 1)
    _P = fam["primary"]
    m, hs = viz[_P]["_mean"], viz[_P]["_subject_aligned"]
    for i in range(468):
        ax0.plot(
            [m[i, 0], hs[i, 0]], [-m[i, 1], -hs[i, 1]], c="#e0a0a0", lw=0.4, alpha=0.6
        )
    ax0.scatter(m[:, 0], -m[:, 1], s=6, c="#888", label="matched-average")
    ax0.scatter(hs[:, 0], -hs[:, 1], s=6, c="#1565c0", label="subject")
    ax0.set_aspect("equal")
    ax0.axis("off")
    ax0.set_title(
        "Subject vs matched-average shape\n(%s, pose-QC, n=%d)"
        % (AB.label_for(_P), viz[_P]["n"]),
        fontsize=11,
    )
    ax0.legend(loc="lower center", fontsize=9, ncol=2)

    # Panel 2: histogram of reference LOO distances + subject + percentile.
    ax1 = fig.add_subplot(1, 3, 2)
    r = viz[_P]
    d = r["_loo_d"]
    ax1.hist(
        d,
        bins=20,
        color="#bcd",
        edgecolor="#789",
        label="reference LOO distance",
    )
    ax1.axvline(r["_subject_d"], color="#c62828", lw=2.5, label="subject distance")
    ax1.set_title(
        f"Primary: {AB.label_for(_P)} (n={r['n']})\n"
        f"more average than {r['percentile']:.0f}% of references  "
        f"(95% CI {r['ci'][0]:.0f}-{r['ci'][1]:.0f}%)",
        fontsize=11,
    )
    ax1.set_xlabel("Euclidean Procrustes distance to mean  (smaller = more average)")
    ax1.set_ylabel("count")
    ax1.legend(fontsize=8)

    # Panel 3: robustness across the four reference sets (percentile + CI).
    ax2 = fig.add_subplot(1, 3, 3)
    labels = [AB.label_for(n) + ("\n(primary)" if n == _P else "") for n in order]
    yv = np.arange(len(order))[::-1]
    pe = [res[n]["percentile"] for n in order]
    lo = [res[n]["percentile"] - res[n]["ci"][0] for n in order]
    hi = [res[n]["ci"][1] - res[n]["percentile"] for n in order]
    ax2.errorbar(
        pe,
        yv,
        xerr=[lo, hi],
        fmt="o",
        color="#1565c0",
        capsize=4,
        ms=8,
        label="averageness percentile (bar = bootstrap 95% CI)",
    )
    ax2.set_yticks(yv)
    ax2.set_yticklabels(labels, fontsize=9)
    ax2.set_xlim(0, 100)
    ax2.axvline(50, color="#aaa", ls="--", lw=1)
    ax2.set_xlabel("averageness percentile (more average than X%)")
    ax2.set_title(
        "Robustness across reference choices\n(full-shape leave-one-out)",
        fontsize=11,
    )
    ax2.legend(fontsize=8, loc="lower left")
    for p, y in zip(pe, yv):
        ax2.annotate(
            f"{p:.0f}%",
            (p, y),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=9,
        )

    fig.suptitle(
        "Averageness — full-shape closeness to the matched average, "
        "leave-one-out, bootstrap CI (literature-faithful)",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(HERE, "averageness_v2.png"), dpi=130)
    print("wrote averageness_v2.json, averageness_v2.png")


if __name__ == "__main__":
    main()
