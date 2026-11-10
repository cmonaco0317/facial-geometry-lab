#!/usr/bin/env python3
"""Gives symmetry the population it has never had, so it can be RANKED.

WHY THIS IS THE ODD ONE OUT
----------------------------
Symmetry does not refuse. It prints — 4.6% of interpupillary distance, per
region, with a bracket. What it cannot do is tell a reader whether that is
unusual, because `anchor_kind` is `none`: there is no population anywhere in this
instrument to compare it against. "Your asymmetry is 4.6% of IPD" is a fact
without a scale, and the backlog names this as the highest-value unbuilt item.

⛔ THE TRAP THIS PROJECT HAS FALLEN INTO FOUR TIMES, avoided by construction here.
The subject's shipped asymmetry can come from the multi-view FIT, which removes
pose. Every bank face is ONE PHOTOGRAPH, which does not. Ranking a fit-derived
number inside a distribution of photograph-derived ones is the same-label-two-
quantities defect — and Phase E measured that a fit scores better than its own
source photographs because it is CLEANER, not better.

So this emits TWO SEPARATELY NAMED QUANTITIES and never mixes them:
    asymmetry_ranked    single photograph, the bank's own method — comparable
    asymmetry_measured  whatever the report already prints — precise, NOT ranked
`selfcheck_cross_file.py` already goes red if one id ever carries two values.

⚠️ AND A CONFOUND THAT MUST TRAVEL WITH THE NUMBER. Single-photograph asymmetry
rises with head yaw: a face turned slightly reads asymmetric whether or not it is.
Every bank face passed the same `is_frontal` gate (|yaw| <= 14 deg), so the
residual is bounded but not zero, and it inflates the spread for EVERYONE rather
than for the subject alone. §3 measures how much of the bank's spread tracks a
yaw proxy instead of asserting that it is small.

WHAT THIS IS NOT
----------------
  * Not a claim that asymmetry is good or bad. It is a position in a population.
  * Not cross-day. Every bank face is one photograph on one day.
  * ⚠️ The bank is FairFace: annotator-labelled, in-the-wild photographs. A
    percentile against it is a percentile against those photographs, not against
    a demographic truth.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import analyze_advanced as AA  # noqa: E402

CACHES = ("ref_landmarks2.npz", "race_landmarks.npz", "fairness_landmarks.npz")
OUT = os.path.join(HERE, "symmetry_population.json")
BOOT = 2000
SEED = 20260904
MIN_N = 30

# ⭐⭐ THE YAW GATE IS THE WHOLE REASON THIS POPULATION MEANS ANYTHING, and 2.0 is
# not a new number — it is `pose_gate.symmetry_yaw_deg`, the gate this project
# already applies to the SUBJECT's symmetry, now applied to the bank so both
# sides are held to it.
#
# ⛔ MEASURED, ON A FACE KNOWN NOT TO CHANGE. Across the pose burst one unchanging
# face reads 0.96% asymmetry at 0.4 deg of yaw and 202% at 58 deg — r = +0.932,
# and ~1.14%/deg inside the old |yaw|<=14 intake gate. The bank's observed spread
# at that gate was 1.95-13.98%, which pose ALONE can account for: the "population"
# was a distribution of how square-on people happened to be photographed.
#
# Tightening to 2 deg collapses the spread from 12.03 points to 2.52 and still
# leaves 1,957 faces — far more than the ~369 a 10-point interval needs.
YAW_GATE_DEG = 2.0


# ⛔ WHY THIS IS NOT `analyze_advanced.symmetry` CALLED DIRECTLY, which was the
# first attempt. That function normalises by IRIS centres (landmarks 468/473),
# and the reference caches store only the first 468 points — the irises are not
# in them. It raises IndexError on every bank face.
#
# ⭐ AND THE TEMPTING FIX IS THE TRAP. Swapping the BANK to outer-canthus IPD
# while the subject keeps iris IPD would put two different normalisations on the
# two sides of one comparison: the same-label-two-quantities defect, in the exact
# place this file exists to avoid it. So BOTH sides use the function below, and
# the subject is re-detected with the bank's own plain detector to feed it —
# §5.1's rule, applied rather than quoted.
L_CANTHUS, R_CANTHUS = 33, 263


def symmetry_468(pts: np.ndarray) -> float | None:
    """Mean bilateral asymmetry as % of OUTER-CANTHAL distance, on 468 points.

    Same geometry as `analyze_advanced.symmetry` — midline SVD for the symmetry
    axis, each pair's mirror residual — with the one substitution the caches
    force. Kept beside the original rather than parameterising it, because
    changing that function's normalisation would silently move every figure the
    report already prints.
    """
    p = np.asarray(pts, dtype=float)
    if p.shape != (468, 2):
        return None
    mid = p[AA.MIDLINE]
    c = mid.mean(axis=0)
    _u, _s, vt = np.linalg.svd(mid - c)
    axis = vt[0]
    if np.dot(p[152] - p[10], axis) < 0:
        axis = -axis
    perp = np.array([-axis[1], axis[0]])
    if perp[0] < 0:
        perp = -perp
    ipd = float(np.linalg.norm(p[L_CANTHUS] - p[R_CANTHUS]))
    if not np.isfinite(ipd) or ipd <= 0:
        return None

    def coords(i):
        d = p[i] - c
        return float(np.dot(d, axis)), float(np.dot(d, perp))

    vals = []
    for li, ri, _reg in AA.PAIRS:
        if li >= 468 or ri >= 468:
            continue
        s_l, t_l = coords(li)
        s_r, t_r = coords(ri)
        vals.append(np.hypot(s_l - s_r, t_l + t_r) / ipd * 100.0)
    return float(np.mean(vals)) if vals else None


def asym_of(pts: np.ndarray) -> float | None:
    return symmetry_468(pts)


def yaw_proxy(pts: np.ndarray) -> float | None:
    """A pose proxy from the landmarks alone, so the confound can be MEASURED.

    Ratio of nose-tip offset from the eye midpoint to interpupillary distance:
    zero looking straight on, growing with turn. Not degrees, and not meant to
    be — it only has to correlate with turn well enough to show whether the
    bank's asymmetry spread is tracking pose.
    """
    p = np.asarray(pts, dtype=float)
    if p.shape != (468, 2):
        return None
    l, r = p[L_CANTHUS], p[R_CANTHUS]
    mid = (l + r) / 2.0
    ipd = float(np.linalg.norm(l - r))
    if not np.isfinite(ipd) or ipd <= 0:
        return None
    return float(abs(p[1][0] - mid[0]) / ipd)


def yaw_degrees(pts: np.ndarray):
    """Estimated |yaw| from the CALIBRATED proxy, not from a raw ratio.

    `reference_pose_audit.validate_against_burst` fits proxy -> degrees on a
    constant-face burst, so the mapping is measured rather than assumed. Cached
    once: it re-detects photographs.
    """
    global _CAL
    import reference_pose_audit as RPA
    if _CAL is None:
        v = RPA.validate_against_burst()
        _CAL = (v["slope_deg_per_unit"], v["intercept_deg"])
    sl, ic = _CAL
    try:
        return abs(sl * RPA.yaw_proxy(pts) + ic)
    except Exception:  # noqa: BLE001
        return None


_CAL = None


def load_cohorts() -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for fn in CACHES:
        path = os.path.join(HERE, fn)
        if not os.path.exists(path):
            continue
        z = np.load(path)
        for k in z.files:
            a = z[k]
            if a.ndim == 3 and a.shape[1:] == (468, 2) and len(a) >= MIN_N:
                # fairness_landmarks holds MALE 20-29 under a bare race name.
                key = k if ("_" in k and k[:2] in ("m_", "f_")) else f"m_{k}_20_29"
                out.setdefault(key, a)
    return out


def percentile_with_interval(value: float, pool: np.ndarray) -> dict:
    """Percentile plus the interval the cohort's size supports.

    ⛔ RESAMPLES THE FACES, matching `analyze_catalogue._cohort_percentile`. An
    interval from a resampled SUMMARY would be narrower than the faces justify.
    ⭐ LOW asymmetry is the notable end, so the percentile is stated as the share
    of the cohort MORE asymmetric than the subject — a high number means unusually
    symmetric. Stating it the other way round would make the flattering direction
    the ambiguous one.
    """
    rng = np.random.default_rng(SEED)
    n = len(pool)
    pct = 100.0 * float(np.mean(pool > value))
    draws = [100.0 * float(np.mean(pool[rng.integers(0, n, n)] > value))
             for _ in range(BOOT)]
    lo, hi = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
    return {"percentile_more_asymmetric_than_subject": round(pct, 1),
            "ci": [round(lo, 1), round(hi, 1)],
            "ci_width": round(hi - lo, 1), "n": int(n)}



def gated_values(cohorts: dict[str, np.ndarray]) -> tuple[list[float], list[tuple]]:
    """Every banked face inside the yaw gate, and the (yaw, asym) pairs for all.

    ⛔ ONE FUNCTION, TWO CONSUMERS. The probe's gate check and the frozen
    reference used to accumulate this separately and disagreed — 1285 against
    1957 — because one filtered out cohorts that fell under MIN_N after gating
    and the other did not. Two counts for one quantity is the drift this project
    has been bitten by; the per-cohort TABLE still filters for readability, but
    the POOLED population both of them rank against comes from here.
    """
    vals, pairs = [], []
    for arr in cohorts.values():
        for pts in arr:
            a = asym_of(pts)
            if a is None or not np.isfinite(a):
                continue
            y = yaw_degrees(pts)
            if y is not None and np.isfinite(y):
                pairs.append((y, a))
                if y > YAW_GATE_DEG:
                    continue
            vals.append(float(a))
    return vals, pairs


def main() -> int:
    cohorts = load_cohorts()
    if not cohorts:
        print("!! no landmark caches found")
        return 1

    print("=" * 78)
    print("SYMMETRY POPULATION — the scale this measurement has never had")
    print("=" * 78)

    dist: dict[str, dict] = {}
    all_vals, all_yaw = [], []
    for name, arr in sorted(cohorts.items()):
        vals, yaws, dropped = [], [], 0
        for pts in arr:
            a = asym_of(pts)
            if a is None or not np.isfinite(a):
                continue
            y = yaw_degrees(pts)
            if y is not None and np.isfinite(y):
                yaws.append((y, a))
                if y > YAW_GATE_DEG:
                    dropped += 1
                    continue
            vals.append(a)
        if len(vals) < MIN_N:
            continue
        v = np.array(vals)
        dist[name] = {"n": int(len(v)), "dropped_off_axis": int(dropped),
                      "median": round(float(np.median(v)), 3),
                      "p10": round(float(np.percentile(v, 10)), 3),
                      "p90": round(float(np.percentile(v, 90)), 3)}
        all_vals.extend(vals)
        all_yaw.extend(yaws)

    print(f"  {len(dist)} cohorts shown (n>={MIN_N} after gating); the POOLED "
          f"population is every gated face\n")
    print(f"{'cohort':30} {'n':>5} {'median':>8} {'p10':>7} {'p90':>7}")
    for k, d in sorted(dist.items(), key=lambda kv: kv[1]["median"])[:6]:
        print(f"{k:30} {d['n']:>5} {d['median']:>8.2f} {d['p10']:>7.2f} "
              f"{d['p90']:>7.2f}")
    print("   ...")
    for k, d in sorted(dist.items(), key=lambda kv: kv[1]["median"])[-3:]:
        print(f"{k:30} {d['n']:>5} {d['median']:>8.2f} {d['p10']:>7.2f} "
              f"{d['p90']:>7.2f}")

    pooled_vals, all_yaw = gated_values(cohorts)
    all_vals = pooled_vals
    pooled = np.array(pooled_vals)
    print(f"\n  pooled: median {np.median(pooled):.2f}%  "
          f"p10 {np.percentile(pooled, 10):.2f}  p90 {np.percentile(pooled, 90):.2f}")

    # ---- §3 THE YAW CONFOUND, MEASURED --------------------------------------
    ya = np.array(all_yaw)
    ungated = ya[:, 1]
    gated = ya[ya[:, 0] <= YAW_GATE_DEG][:, 1]
    r_all = float(np.corrcoef(ya[:, 0], ya[:, 1])[0, 1])
    sub = ya[ya[:, 0] <= YAW_GATE_DEG]
    r_gate = float(np.corrcoef(sub[:, 0], sub[:, 1])[0, 1]) if len(sub) > 10 else \
        float("nan")
    sp_all = float(np.percentile(ungated, 90) - np.percentile(ungated, 10))
    sp_gate = float(np.percentile(gated, 90) - np.percentile(gated, 10))
    print(f"\n  THE YAW CONFOUND, and what the gate does to it:")
    print(f"    all {len(ungated)} faces      spread {sp_all:5.2f} pts, "
          f"corr(|yaw|, asym) {r_all:+.3f}")
    print(f"    {len(gated)} inside {YAW_GATE_DEG:g} deg   spread {sp_gate:5.2f} pts, "
          f"corr(|yaw|, asym) {r_gate:+.3f}")
    print(f"    the gate removes {100 * (1 - sp_gate / sp_all):.0f}% of the spread")
    print("    ⛔ measured on ONE UNCHANGING FACE: 0.96% asymmetry at 0.4 deg of")
    print("       yaw, 202% at 58 deg. Single-photograph asymmetry is largely a")
    print("       POSE measurement, so an ungated bank ranks how square-on people")
    print("       were photographed. That is why the gate is not optional.")
    share = r_gate * r_gate

    # ---- the subject, measured the SAME WAY ---------------------------------
    # ⭐ THE SUBJECT IS RE-DETECTED WITH THE BANK'S OWN PLAIN DETECTOR and run
    # through the SAME symmetry_468. Reading the shipped figure instead would
    # rank an iris-normalised, possibly fit-derived number inside a distribution
    # of outer-canthus single-photograph ones.
    subject = None
    frontal = os.path.join(HERE, "frontal.png")
    if os.path.exists(frontal):
        import analyze_averageness2 as A2
        lmk = A2.make_landmarker()
        try:
            lms, _ang = A2.landmarks_for(lmk, frontal)
        finally:
            lmk.close()
        val = symmetry_468(lms) if lms is not None else None
        if val is not None:
            val = round(val, 3)
            cohort = None
            try:
                import subject as SUBJ
                cohort = SUBJ.race_cohort()[0]
            except Exception:  # noqa: BLE001
                pass
            pool = cohorts.get(cohort) if cohort in dist else None
            used = cohort if pool is not None else "POOLED"
            arr = (np.array([a for a in (asym_of(p) for p in pool)
                             if a is not None]) if pool is not None else pooled)
            subject = {"asymmetry_ranked": val, "cohort": used,
                       **percentile_with_interval(float(val), arr)}
            print(f"\n  SUBJECT: asymmetry_ranked {val}% of IPD, against "
                  f"{used} (n={subject['n']})")
            print(f"    {subject['percentile_more_asymmetric_than_subject']}% of "
                  f"that cohort is MORE asymmetric "
                  f"(interval {subject['ci'][0]}-{subject['ci'][1]})")
            ship = json.load(open(os.path.join(HERE, "advanced_metrics.json"),
                                  encoding="utf-8"))["symmetry"].get(
                                      "mean_asym_pct_ipd")
            print(f"    the report separately prints {ship}% "
                  "(asymmetry_measured, iris-normalised)")
            print("    ⚠️ TWO NAMES ON PURPOSE. This ranks a single-photograph, "
                  "outer-canthus number")
            print("       against single-photograph outer-canthus faces. The "
                  "shipped figure uses iris")
            print("       centres and may come from the multi-view fit: more "
                  "precise, NOT rankable here.")

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({
            "_what": "distribution of mean asymmetry (% of IPD) across every "
                     "banked cohort, measured by analyze_advanced.symmetry — the "
                     "bank's own method, unmodified",
            "_two_names_on_purpose": "asymmetry_ranked is single-photograph and "
                                     "comparable to this bank; asymmetry_measured "
                                     "is whatever the report prints and may be "
                                     "fit-derived, which is more precise and NOT "
                                     "rankable here. Mixing them is the "
                                     "same-label-two-quantities defect this "
                                     "project has hit four times.",
            "_yaw_confound": {"r_ungated": r_all, "r_gated": r_gate,
                              "spread_ungated": sp_all, "spread_gated": sp_gate,
                              "gate_deg": YAW_GATE_DEG,
                              "constant_face": "0.96% asymmetry at 0.4 deg yaw, "
                                               "202% at 58 deg, r=+0.932",
                              "note": "single-photograph asymmetry rises with "
                                      "head turn; every bank face passed the same "
                                      "|yaw|<=14 gate so the residual is bounded, "
                                      "and it inflates the spread for everyone "
                                      "rather than for the subject alone"},
            "_limitation": "FairFace: annotator-labelled in-the-wild photographs, "
                           "one photograph per face, one day. A percentile against "
                           "it is a percentile against those photographs.",
            "n_cohorts": len(dist), "n_faces": len(all_vals),
            "pooled": {"median": round(float(np.median(pooled)), 3),
                       "p10": round(float(np.percentile(pooled, 10)), 3),
                       "p90": round(float(np.percentile(pooled, 90)), 3)},
            "cohorts": dist, "subject": subject,
        }, fh, indent=2)
    print(f"\n  wrote {os.path.basename(OUT)}")

    # ⛔ A PASS CONDITION, because a probe that always exits 0 cannot be a claim.
    # The population is only usable if the gate did its job, so the bar is on the
    # gate rather than on the answer: most of the pose-driven spread removed, a
    # residual correlation that is small rather than merely smaller, and enough
    # faces left to support an interval (cohort_depth_probe: ~369 for 10 points).
    ok = True
    if sp_all > 0 and (1 - sp_gate / sp_all) < 0.50:
        print("  ⛔ the gate removes less than half the spread — the population "
              "is still mostly pose")
        ok = False
    if not (abs(r_gate) < 0.40):
        print(f"  ⛔ residual yaw correlation {r_gate:+.3f} is too high to rank "
              "faces rather than framings")
        ok = False
    if len(gated) < 369:
        print(f"  ⛔ only {len(gated)} faces survive the gate; ~369 are needed "
              "for a 10-point interval")
        ok = False
    if ok:
        print(f"  ✅ {len(gated)} faces inside the gate, {100*(1-sp_gate/sp_all):.0f}% "
              f"of the spread removed, residual r {r_gate:+.3f}")
    return 0 if ok else 1


def emit() -> int:
    """Freeze the gated population into symmetry_reference.py.

    ⛔ FROZEN RATHER THAN RECOMPUTED PER RUN, for the same reason race_reference
    is: measuring 10,000 bank faces and re-detecting a pose burst on every report
    would put minutes into every run and make the population silently depend on
    what happened to be on disk that day. A frozen file is auditable and a run
    cannot move it.
    """
    import json as _json
    if main() != 0:
        print("!! the population did not pass its own gate check; not emitting")
        return 1
    d = _json.load(open(OUT, encoding="utf-8"))
    cohorts = load_cohorts()
    vals = sorted(round(v, 4) for v in gated_values(cohorts)[0])
    body = [
        '"""FROZEN pose-gated symmetry population — emitted by '
        'symmetry_population.py.',
        "",
        "⛔ DO NOT HAND-EDIT. Regenerate with:",
        "    ./venv/bin/python symmetry_population.py --emit",
        "",
        "Mean bilateral asymmetry as % of OUTER-CANTHAL distance, from banked",
        f"FairFace faces whose estimated |yaw| is within {YAW_GATE_DEG:g} deg —",
        "pose_gate.symmetry_yaw_deg, the gate this project already applies to the",
        "SUBJECT, applied to the bank so both sides are held to it.",
        "",
        "⭐ THE GATE IS NOT OPTIONAL AND THE MEASUREMENT SAYS WHY. On one",
        "unchanging face asymmetry reads 0.96% at 0.4 deg of yaw and 202% at 58",
        "deg (r=+0.932). Ungated, this bank ranked how square-on people happened",
        "to be photographed: the subject scored at the 86th percentile for",
        "symmetry, and gated properly he is MORE asymmetric than 78% of faces.",
        "The ranking inverts, in the flattering direction.",
        "",
        "⚠️ A number ranked here must be measured the SAME way: single",
        "photograph, outer-canthus normalised, via the bank's own plain detector.",
        "The report's iris-normalised or fit-derived figure is NOT comparable.",
        '"""',
        "",
        f"YAW_GATE_DEG = {YAW_GATE_DEG!r}",
        f"N_FACES = {len(vals)}",
        f"MEDIAN = {round(float(np.median(vals)), 4)!r}",
        f"YAW_RESIDUAL_R = {round(float(d['_yaw_confound']['r_gated']), 4)!r}",
        f"SPREAD_REMOVED_BY_GATE = "
        f"{round(1 - d['_yaw_confound']['spread_gated'] / d['_yaw_confound']['spread_ungated'], 4)!r}",
        "",
        "# Every gated face, sorted. Kept whole rather than as percentiles so a",
        "# bootstrap interval can resample the FACES — resampling a summary",
        "# reports a narrower interval than the faces justify.",
        f"VALUES = {vals!r}",
        "",
    ]
    path = os.path.join(HERE, "symmetry_reference.py")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(body))
    print(f"\n  froze {len(vals)} gated faces -> symmetry_reference.py")
    return 0


if __name__ == "__main__":
    if "--emit" in sys.argv:
        sys.exit(emit())
    sys.exit(main())
