#!/usr/bin/env python3
"""Measure what each sexual-dimorphism ratio can actually DO, on real faces.

WHY THIS EXISTS
---------------
Every neutral band in `analyze_advanced.dimorphism` used to come from published
anthropometry whose landmark definitions are not this pipeline's. The module's
own comment said so -- "literature-informed midpoints for ORIENTATION, not exact
population norms for this pipeline's specific landmark set" -- but the report
still printed a masculine/feminine badge off them. Measured 2026-08-19 against
825 reference rows (762 distinct faces) of known sex, that turned out to matter enormously:

  * `Brow-to-eyelid gap / eye width` put its neutral band at 0.50-0.60 while the
    real population median under these landmarks is 0.291. 93% of all 825 people,
    men and women alike, were labelled "masculine" by it.
  * `Nose width / bizygomatic width` separated the sexes at AUC 0.445 -- i.e.
    BACKWARDS from the direction the literature asserts, because the denominator
    is itself male-larger and cancels the numerator's real dimorphism.
  * four of seven features could not tell a man from a woman at all.

⛔ THE FIX IS NOT TO MOVE THE BANDS ONTO THE POPULATION MEDIAN. That manufactures
a 50/50 verdict out of a feature carrying no signal -- a coin flip dressed as
anatomy. A band is only worth printing if the underlying ratio actually separates
the sexes, so this module measures that first and the band second.

METHOD, fixed before any result was looked at
---------------------------------------------
* Candidates are LITERATURE-MOTIVATED renormalisations of the same anatomy, not a
  blind search. The defect being repaired is a denominator that cancels the
  signal, so the candidates vary the denominator.
* SELECT on the 10-19 banks, VALIDATE on the disjoint 20-29 banks -- different
  people. A variant is adopted only if it survives on faces it was not chosen on:
  held-out AUC >= 0.60 at p < 0.05, and select-half AUC >= 0.58.
* Direction is ASSERTED from the literature and never flipped to fit. A variant
  that separates in the wrong direction is a refutation, not a discovery.
* Every adopted variant must also hold in all four RACE- AND AGE-MATCHED pairings
  separately, so a pooled result cannot be carried by cohort composition.
* The subject's own values are not read by this file and play no part in
  selection. Nothing here can be tuned to flatter one face.

⚠️ LIMIT, and it is the important one: these are 2D FRONTAL landmarks from
photographs admitted at |yaw| <= 14 deg. A ratio that fails here has failed FOR
THIS INSTRUMENT; it is not a claim about the anatomy. The brow ridge is the clear
case -- it is a DEPTH feature, and no frontal renormalisation recovered it.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import fa_common as F  # noqa: E402

CACHE = os.path.join(HERE, "ref_landmarks2.npz")
SEL_M = ["white_10_19", "broad_10_19"]
SEL_F = ["female_10_19", "female_broad_10_19"]
VAL_M = ["white_20_29", "broad_20_29"]
VAL_F = ["female_20_29", "female_broad_20_29"]
PAIRINGS = [
    ("white 20-29", "white_20_29", "female_20_29"),
    ("all-race 20-29", "broad_20_29", "female_broad_20_29"),
    ("white 10-19", "white_10_19", "female_10_19"),
    ("all-race 10-19", "broad_10_19", "female_broad_10_19"),
]
# A lean is asserted only where the value is this many times more common in one
# sex than the other. 1.5 keeps roughly half the population neutral at the
# separations actually measured here, and gives the report a sentence that means
# something ("1.5x more common among men") instead of a bare badge.
LEAN_ODDS = 1.5


def measures(p):
    """Every base length the candidate ratios are built from."""
    d = {}
    ecL = p[F.LEFT_EYE_RING].mean(axis=0)
    ecR = p[F.RIGHT_EYE_RING].mean(axis=0)
    # Eye-ring centroids rather than iris: the reference bank is a 468-point mesh
    # with no iris landmarks, and the ring centroid reproduces the detected iris
    # IPD to 1.4% (see the `improvement-measured-on-one-ruler` claim).
    d["ipd"] = float(np.hypot(*(ecL - ecR)))
    d["bizygo"] = F.hdist(p, F.ZYGION_L, F.ZYGION_R)
    browc = (p[F.BROW_L][1] + p[F.BROW_R][1]) / 2
    d["face_h"] = abs(p[F.TRICHION][1] - p[F.MENTON][1])
    d["upper_h"] = abs(browc - p[F.TRICHION][1])
    d["mid_h"] = abs(p[F.SUBNASALE][1] - browc)
    d["lower_h"] = abs(p[F.MENTON][1] - p[F.SUBNASALE][1])
    # ⚠️ The CLASSICAL fWHR denominator is brow -> UPPER LIP, not brow ->
    # hairline. They are different quantities and the first version of this
    # harness conflated them, which would have had the module claim it had
    # tested "the published fWHR construct" when it had tested something else.
    d["fwhr_h"] = abs(p[F.LABRALE_SUP][1] - browc)
    d["nose_w"] = F.hdist(p, F.ALA_L, F.ALA_R)
    d["mouth_w"] = F.hdist(p, F.MOUTH_L, F.MOUTH_R)
    d["lip_h"] = F.vdist(p, F.LABRALE_SUP, F.LABRALE_INF)
    d["jaw_w"] = F.hdist(p, F.JAW_L, F.JAW_R)
    d["eye_w"] = (
        F.hdist(p, F.L_CANTHUS_LAT, F.L_CANTHUS_MED)
        + F.hdist(p, F.R_CANTHUS_MED, F.R_CANTHUS_LAT)
    ) / 2
    d["eye_h"] = (
        abs(p[F.L_EYE_TOP][1] - p[F.L_EYE_BOT][1])
        + abs(p[F.R_EYE_TOP][1] - p[F.R_EYE_BOT][1])
    ) / 2
    lid = (p[F.L_EYE_TOP][1] + p[F.R_EYE_TOP][1]) / 2
    d["brow_med"] = abs(lid - browc)
    return d


# The shipped ratio for each feature. `sex_high` is the literature's assertion
# about which sex reads HIGHER, never a fit to these data.
FEATURES = [
    ("face_width", "Face width / eye spacing", lambda d: d["bizygo"] / d["ipd"], "M"),
    ("jaw_width", "Jaw width / eye spacing", lambda d: d["jaw_w"] / d["ipd"], "M"),
    ("eye_size", "Eye width / face width", lambda d: d["eye_w"] / d["bizygo"], "F"),
    ("lower_third", "Lower-face height / eye spacing", lambda d: d["lower_h"] / d["ipd"], "M"),
    ("nose_width", "Nose width / mouth width", lambda d: d["nose_w"] / d["mouth_w"], "M"),
    ("lip_height", "Lip height / lower-face height", lambda d: d["lip_h"] / d["lower_h"], "F"),
    ("brow_gap", "Brow-to-eyelid gap / eye width", lambda d: d["brow_med"] / d["eye_w"], "F"),
]


def _auc(m, f, hi):
    """AUC oriented so >0.5 ALWAYS means 'matches the asserted direction'."""
    u = stats.mannwhitneyu(f, m, alternative="two-sided")
    a = u.statistic / (len(m) * len(f))
    return (a if hi == "F" else 1 - a), float(u.pvalue)


def build():
    z = np.load(CACHE, allow_pickle=False)
    banks = {b: [measures(p) for p in z[b]] for b in SEL_M + SEL_F + VAL_M + VAL_F}

    def pull(names, fn):
        return np.array([fn(d) for b in names for d in banks[b]])

    out = {}
    for fid, label, fn, hi in FEATURES:
        M = pull(SEL_M + VAL_M, fn)
        Fe = pull(SEL_F + VAL_F, fn)
        sel_auc, _ = _auc(pull(SEL_M, fn), pull(SEL_F, fn), hi)
        val_auc, val_p = _auc(pull(VAL_M, fn), pull(VAL_F, fn), hi)
        per = {}
        for nm, mb, fb in PAIRINGS:
            per[nm] = round(_auc(pull([mb], fn), pull([fb], fn), hi)[0], 3)
        separates = (
            val_auc >= 0.60
            and val_p < 0.05
            and sel_auc >= 0.58
            and all(v > 0.5 for v in per.values())
        )
        muM, muF = float(M.mean()), float(Fe.mean())
        sd = float(
            np.sqrt(
                ((len(M) - 1) * M.var(ddof=1) + (len(Fe) - 1) * Fe.var(ddof=1))
                / (len(M) + len(Fe) - 2)
            )
        )
        rec = {
            "label": label,
            "sex_high": hi,
            "separates": bool(separates),
            "auc_select": round(sel_auc, 3),
            "auc_holdout": round(val_auc, 3),
            "p_holdout": val_p,
            "auc_by_pairing": per,
            "n_male": int(len(M)),
            "n_female": int(len(Fe)),
            "male_mean": round(muM, 4),
            "female_mean": round(muF, 4),
            "pooled_sd": round(sd, 4),
            "pop_median": round(float(np.median(np.concatenate([M, Fe]))), 4),
            "pop_p5": round(float(np.percentile(np.concatenate([M, Fe]), 5)), 4),
            "pop_p95": round(float(np.percentile(np.concatenate([M, Fe]), 95)), 4),
        }
        # Band edges from the measured distributions, under a pooled-SD linear
        # discriminant: assert a lean only where one sex is LEAN_ODDS times more
        # likely. Monotone by construction, so no value can flip back at an
        # extreme the way unequal-variance densities do.
        delta = muF - muM
        if separates and abs(delta) > 1e-9:
            off = float(np.log(LEAN_ODDS) * sd * sd / delta)
            mid = (muM + muF) / 2
            lo, hi_e = sorted([mid - off, mid + off])
            rec["band"] = [round(lo, 4), round(hi_e, 4)]
            # Does the stated 1.5x actually hold in the data, or only in the model?
            fem_side = (Fe > hi_e) if delta > 0 else (Fe < lo)
            mal_side = (M > hi_e) if delta > 0 else (M < lo)
            rec["empirical_odds_at_feminine_edge"] = round(
                float(fem_side.mean() / max(mal_side.mean(), 1e-9)), 2
            )
            othF = (Fe < lo) if delta > 0 else (Fe > hi_e)
            othM = (M < lo) if delta > 0 else (M > hi_e)
            rec["empirical_odds_at_masculine_edge"] = round(
                float(othM.mean() / max(othF.mean(), 1e-9)), 2
            )
            rec["pct_neutral"] = round(
                100
                * float(
                    np.mean(
                        (np.concatenate([M, Fe]) >= lo)
                        & (np.concatenate([M, Fe]) <= hi_e)
                    )
                ),
                1,
            )
        else:
            rec["band"] = None
        rec["_sorted_population"] = np.sort(np.concatenate([M, Fe]))
        out[fid] = rec

    # Which features move together? The report counts features, so a cluster
    # counted once per member would be one signal counted several times.
    ids = [f[0] for f in FEATURES]
    X = np.column_stack([pull(SEL_M + SEL_F + VAL_M + VAL_F, fn) for _, _, fn, _ in FEATURES])
    C = np.corrcoef(X.T)
    return out, ids, C


if __name__ == "__main__":
    recs, ids, C = build()
    print(f"{'feature':34s} {'dir':>4s} {'select':>7s} {'holdout':>8s} {'p':>9s} {'band':>17s}  separates?")
    print("-" * 104)
    for fid in ids:
        r = recs[fid]
        b = "—" if not r["band"] else f"{r['band'][0]:.3f}-{r['band'][1]:.3f}"
        print(
            f"{r['label']:34s} {r['sex_high']:>4s} {r['auc_select']:7.3f} "
            f"{r['auc_holdout']:8.3f} {r['p_holdout']:9.1e} {b:>17s}  {r['separates']}"
        )
    print("\ncorrelation between features (825 faces):")
    print("      " + " ".join(f"{i[:9]:>10s}" for i in ids))
    for i, fid in enumerate(ids):
        print(f"{fid[:9]:>9s} " + " ".join(f"{C[i,j]:10.2f}" for j in range(len(ids))))


# ----------------------------------------------------------------------------
# EMIT the shipped reference module.
#
# Production must not read `ref_landmarks2.npz` at run time: it is gitignored,
# it is rebuilt by a fetch that deletes its own source photographs, and a
# measured constant that can silently change under the report is the thing this
# project's whole registry exists to prevent. So the measurement is frozen into
# literals with its provenance attached, exactly as reference_values.py does.
# ----------------------------------------------------------------------------

# Literature note + citation per feature. The ANATOMY claim is the literature's;
# the "MEASURED HERE" sentence is this project's and is what the band rests on.
NOTES = {
    "face_width": (
        "greater facial width relative to eye spacing reads masculine. NOTE this is "
        "NOT the published fWHR construct (bizygomatic width / brow-to-upper-lip "
        "height): that exact ratio was measured here at AUC 0.531 (p=0.32, held-out) and did not "
        "clear the bar, which is consistent with the contested replication record "
        "for fWHR sex differences. Re-normalising the same width against "
        "inter-ocular distance -- a dimension that grows far less between the sexes "
        "than face width does -- recovers the separation.",
        "Bizygomatic breadth robustly larger in males across populations (Farkas "
        "anthropometry and later multi-population reviews); fWHR sex-difference "
        "claims are contested (Kramer 2017 and subsequent failures to replicate).",
    ),
    "jaw_width": (
        "a wider jaw relative to eye spacing reads masculine; mandibular breadth is "
        "one of the more robustly dimorphic facial dimensions. Measured against eye "
        "spacing rather than bizygomatic width, because the bizygomatic denominator "
        "is itself male-larger and cancels most of the signal (that ratio measures "
        "AUC 0.586 here against 0.666 for this one).",
        "Mandibular/bigonial breadth larger in males (Farkas 2005 anthropometry; "
        "Velemínská 2015 3D facial morphology).",
    ),
    "eye_size": (
        "relatively larger eyes read feminine. Retained and PRINTED, but it carries "
        "no lean: measured held-out AUC 0.599 against a bar of 0.60 fixed before the "
        "measurement was made. It misses by 0.001. ⛔ The bar was NOT lowered to "
        "admit it -- a threshold moved after seeing where a candidate landed is "
        "fitted to that candidate, and this project has retired claims for less.",
        "Females show greater vertical eye-fissure height and larger relative eye "
        "dimensions (Velemínská 2015 3D facial morphology).",
    ),
    "lower_third": (
        "a taller lower face relative to eye spacing reads masculine; lower-face "
        "height is dimorphic and is the part of facial height that grows most with "
        "male pubertal development.",
        "Lower facial height greater in males; mandibular growth is the dominant "
        "male-specific adolescent change (Farkas anthropometry; Enlow craniofacial "
        "growth).",
    ),
    "nose_width": (
        "a wider nose relative to the mouth reads masculine. ⭐ THE PREVIOUS "
        "NORMALISATION WAS MEASURABLY BACKWARDS: nose width / bizygomatic width "
        "separates the sexes at AUC 0.445 here -- the WRONG side of a coin flip -- "
        "because the bizygomatic denominator is itself male-larger and more than "
        "cancels the numerator. Absolute nasal breadth is robustly dimorphic; the "
        "ratio that was shipped was not.",
        "Alar width ~44.4mm male vs ~39.3mm female (p<0.001); all linear nasal "
        "dimensions larger in males (Ozdemir 2015; reviews after Farkas 2005).",
    ),
    "lip_height": (
        "greater vermilion height relative to lower-face height reads feminine. "
        "Normalised against lower-face height rather than mouth width because mouth "
        "width is itself dimorphic in the same direction and dilutes the signal "
        "(that ratio measures AUC 0.546 here against 0.638 for this one). Lip "
        "fullness also thins markedly with age, which this cannot separate.",
        "Increased vermilion height associated with femininity (Penna 2024 "
        "ideal-lips anthropometry; Velemínská 2015 3D analysis).",
    ),
    "brow_gap": (
        "a larger visible gap between brow and upper lid is CLAIMED to read "
        "feminine. ⛔ IT DOES NOT DO SO HERE, and the value is printed without a "
        "lean for that reason. Measured on 825 reference rows (762 distinct faces) of known sex this ratio "
        "separates at AUC 0.467 -- a coin flip, and if anything on the wrong side -- "
        "and it reads below 0.5 in ALL FOUR race- and age-matched pairings "
        "separately. Seventeen renormalisations across two rounds were tried "
        "(brow head, mid-brow, lateral brow, arch, brow span, against eye width, "
        "IPD, face height and lower-face height) and none cleared the bar. The "
        "mechanism is not mysterious: the sex-linked part of this anatomy is "
        "supraorbital RIDGE PROJECTION, which is a DEPTH measurement, and these are "
        "frontal landmarks. The profile path does see the glabella, but its own "
        "cross-session check failed (metrics_profile._meta.cross_session_warning), "
        "so it cannot carry a verdict either. ⚠️ The old band was 0.50-0.60 against "
        "a real population median of 0.291, so it labelled 93% of all 825 people -- "
        "men and women alike -- 'masculine'.",
        "Brow ridge robustly more prominent in males (Godinho/Spikins 2018); 3D "
        "analysis finds brow top-edge height ~equal between sexes, female brows "
        "thinner, so the apparent gap is largely a grooming artefact (Gao 2023).",
    ),
}


def _py(v):
    """numpy scalars must not reach the emitted literals -- the module would
    then need numpy imported to be READ, and a generated constants file that
    cannot be opened without its generator is not a frozen constant."""
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, (list, tuple)):
        return [_py(x) for x in v]
    if isinstance(v, dict):
        return {k: _py(x) for k, x in v.items()}
    return v


def emit(path=None):
    MOVEMENT = movement()
    recs, ids, C = build()
    z = np.load(CACHE, allow_pickle=False)
    n_tot = sum(len(z[b]) for b in SEL_M + SEL_F + VAL_M + VAL_F)

    # Which features move together, so a count of features is not one signal
    # counted several times. Single-link at |r| >= 0.50.
    parent = {i: i for i in ids}

    def find(a):
        while parent[a] != a:
            a = parent[a]
        return a

    for i, a in enumerate(ids):
        for j, b in enumerate(ids):
            if j > i and abs(C[i, j]) >= 0.50:
                parent[find(b)] = find(a)
    clusters = {}
    for a in ids:
        clusters.setdefault(find(a), []).append(a)

    L = []
    L.append('"""Measured sexual-dimorphism reference — GENERATED, do not hand-edit.')
    L.append("")
    L.append("Regenerate with:  ./venv/bin/python dimorphism_bank_probe.py --emit")
    L.append("")
    L.append(f"Every number below is measured on {n_tot} real reference faces of known sex")
    L.append("(MediaPipe 468-point landmarks, frontal-filtered at |yaw| <= 14 deg), split")
    L.append("SELECT = the 10-19 banks / HELD-OUT = the disjoint 20-29 banks. A feature")
    L.append("carries a lean ONLY if it cleared, on faces it was not selected on:")
    L.append("held-out AUC >= 0.60 at p < 0.05, select AUC >= 0.58, and AUC > 0.5 in all")
    L.append("four race- and age-matched pairings separately.")
    L.append("")
    L.append("⚠️ ONE FIELD IS NOT MEASURED ON THOSE FACES: 'movement' is the full observed")
    L.append(f"range of the ratio across the {MOVEMENT['_n_sittings']} GATE-PASSING SITTINGS OF ONE PERSON named in")
    L.append("MOVEMENT_SITTINGS below. It is a FLOOR -- one face, one camera -- and it can")
    L.append("only GROW as sittings are added, which is the point of adding them.")
    L.append("")
    L.append("⛔ These bands are NOT literature anchors and must not be edited toward one.")
    L.append("⛔ Nor are they the population median: a band centred on the median would")
    L.append("   manufacture a 50/50 verdict from a feature carrying no signal. They are")
    L.append("   likelihood-ratio edges — the value beyond which one sex is")
    L.append(f"   {LEAN_ODDS}x more likely — computed from the measured distributions.")
    L.append('"""')
    L.append("")
    L.append(f"LEAN_ODDS = {LEAN_ODDS}")
    L.append(f"N_REFERENCE_FACES = {n_tot}")
    L.append("")
    # ⭐ THE SITTING COUNT IS EMITTED, NOT LEFT TO A COMMENT. Every 'movement'
    # below is the full observed range across these sittings, and a reader
    # holding this file had no way to see how many that was -- the number lived
    # only in a hand-written comment in the probe and in prose in three other
    # files, which is exactly how "n=2" survived into documents after a third
    # sitting existed. catalogue_reference.py already showed its n implicitly
    # through 'sitting_values'; this makes it explicit on both sides.
    L.append(f"N_MOVEMENT_SITTINGS = {MOVEMENT['_n_sittings']}")
    L.append(f"MOVEMENT_SITTINGS = {MOVEMENT['_sittings']!r}")
    L.append("")
    L.append("FEATURES = {")
    for fid in ids:
        r = recs[fid]
        why, src = NOTES[fid]
        pcts = [round(float(v), 4) for v in np.percentile(r["_sorted_population"], np.arange(0, 101))]
        L.append(f"    {fid!r}: {{")
        for k in ("label", "sex_high", "separates", "band", "auc_select", "auc_holdout"):
            L.append(f"        {k!r}: {_py(r[k])!r},")
        L.append(f"        'p_holdout': {r['p_holdout']:.3e},")
        L.append("        'auc_by_pairing': {" + ", ".join(f"{k!r}: {float(v)!r}" for k, v in r['auc_by_pairing'].items()) + "},")
        for k in ("n_male", "n_female", "male_mean", "female_mean", "pooled_sd", "pop_median"):
            L.append(f"        {k!r}: {_py(r[k])!r},")
        for k in ("pct_neutral", "empirical_odds_at_feminine_edge", "empirical_odds_at_masculine_edge"):
            if k in r:
                L.append(f"        {k!r}: {_py(r[k])!r},")
        L.append(f"        'movement': {_py(MOVEMENT[fid])!r},")
        L.append(f"        'why': {why!r},")
        L.append(f"        'source': {src!r},")
        L.append(f"        'percentiles': {pcts!r},")
        L.append("    },")
    L.append("}")
    L.append("")
    L.append("# Features whose values move together, so the tally counts SIGNALS not rows.")
    L.append("# Single-link clustering at |r| >= 0.50 on the same reference faces.")
    L.append("CLUSTERS = " + repr([sorted(v) for v in clusters.values()]))
    L.append("CORRELATION = {")
    for i, a in enumerate(ids):
        L.append(f"    {a!r}: {{" + ", ".join(f"{ids[j]!r}: {round(float(C[i,j]),2)!r}" for j in range(len(ids))) + "},")
    L.append("}")
    txt = "\n".join(L) + "\n"
    path = path or os.path.join(HERE, "dimorphism_reference.py")
    with open(path, "w") as fh:
        fh.write(txt)
    print(f"wrote {path}  ({len(txt)} bytes, {len(ids)} features, {n_tot} reference faces)")


# Cross-sitting movement of each shipped ratio, measured on ONE UNCHANGED FACE
# across the three GATE-PASSING sittings (84.5, 84.8 and 81.8 deg of effective
# spread). ⚠️ n=3 SINCE 2026-08-20, n=1 PERSON. This is still a FLOOR on the true
# movement -- one face and one camera -- but it is no longer a two-point range.
# The third of them was shot on a different day in a NEW ROOM with a fresh setup,
# which is the session variation the margin exists to capture, and folding it in
# GREW five of the seven ratios (brow gap 19x, lower third 1.9x, jaw 1.4x, face
# width 1.4x, nose width and lip height barely). The 38.3-deg sitting is excluded deliberately: it
# sits below the 52-deg MEASURED-REAL intake floor, and pooling it is what made
# the previous margins refuse on good photographs.
# ⭐ COMPUTED FROM THE SITTINGS, NOT TYPED. These were hardcoded literals from
# the first two gate-passing sittings, which made adding a third a CODE EDIT --
# and a margin nobody can regenerate is exactly what this project's registry
# exists to prevent. Adding a sitting is now a data change: drop its
# landmarks_frontal.csv path into SITTING_CSVS and re-emit.
#
# ⚠️ THE MARGIN IS THE FULL OBSERVED RANGE ACROSS SITTINGS, so it can only GROW
# as sittings are added. That is the correct direction and the point of adding
# them: a feature that stops resolving once a third real sitting is included was
# never resolving, it was being judged on too little evidence.
SITTING_CSVS = [
    os.path.join(HERE, "captures", "2026-08-12_second-set", "_results", "landmarks_frontal.csv"),
    os.path.join(HERE, "captures", "2026-08-12_third-set", "_results", "landmarks_frontal.csv"),
    os.path.join(HERE, "captures", "2026-08-19_fourth-sitting", "_results", "landmarks_frontal.csv"),
]


def _load_csv(path):
    import csv as _csv

    rows = list(_csv.DictReader(open(path)))
    a = np.zeros((len(rows), 2))
    for r in rows:
        a[int(r["index"])] = [float(r["x"]), float(r["y"])]
    return a


def measure_movement():
    """Full observed range of each shipped ratio across the gate-passing sittings."""
    present = [p for p in SITTING_CSVS if os.path.exists(p)]
    if len(present) < 2:
        raise SystemExit(
            f"movement needs at least 2 gate-passing sittings, found {len(present)}"
        )
    bases = [measures(_load_csv(p)) for p in present]
    out = {}
    for fid, label, fn, hi in FEATURES:
        vals = [float(fn(b)) for b in bases]
        out[fid] = round(max(vals) - min(vals), 5)
    out["_n_sittings"] = len(present)
    out["_sittings"] = [os.path.basename(os.path.dirname(os.path.dirname(p))) for p in present]
    return out


# ⛔ NOT MEASURED AT IMPORT. Until 2026-09-21 this read `MOVEMENT = measure_movement()`,
# which made importing this module -- and through analyze_advanced ->
# analyze_averageness2 -> assess_set, the product's whole analysis chain --
# Private development note omitted.
# are his own face), so the packaged macOS app raised SystemExit inside its
# request thread on the first real upload: no response, no session cleanup,
# the browser saw the connection drop. The product path only needs measures(),
# FEATURES and probability(); the movement table is for --emit alone, so it is
# measured when emit() asks for it and not before.
_MOVEMENT = None


def movement():
    """The cross-sitting movement table, measured on first use (emit-time only)."""
    global _MOVEMENT
    if _MOVEMENT is None:
        _MOVEMENT = measure_movement()
    return _MOVEMENT


# ⛔ GUARDED ON __main__, NOT ON sys.argv ALONE. Until 2026-08-23 this was just
# `if "--emit" in sys.argv`, which meant ANY other script invoked with --emit
# silently regenerated this module's frozen reference the moment it imported
# this one for `measures()` or `_auc()`. race_bank_probe.py --emit did exactly
# that; it happened to be byte-identical so nothing broke, but a frozen file
# rewriting itself as a side effect of an unrelated command is how a reference
# quietly stops matching the figures published against it.
if __name__ == "__main__" and "--emit" in sys.argv:
    emit()
