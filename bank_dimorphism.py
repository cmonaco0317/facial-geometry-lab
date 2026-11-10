#!/usr/bin/env python3
"""§6.2 — dimorphism AUC computed WITHIN a cohort, with an interval, per feature.

WHY POOLED AUC IS THE WRONG NUMBER, AND HAS BEEN THE SHIPPED ONE
-----------------------------------------------------------------
Today's figures (0.62-0.67 on held-out faces, one metric CONTRADICTED at 0.46)
are pooled across cohorts.  On a sex-linked measure, pooling INFLATES AUC by
whatever the cohorts differ from each other by: if faces labelled one way are on
average wider than faces labelled another, and the two groups are not sampled
identically by sex, a width feature scores for a reason that has nothing to do
with sex.  The number is then a fact about the sampling, printed as a fact about
the face.

⭐ WHAT THIS BUYS THAT ARGUMENT CANNOT.  The old per-race banks held 56-130
faces each, which is why 16 of 35 race x feature combinations came back
'underpowered' -- a fact about the instrument, not the person.  This runs
against 25,534 faces in 56 cohorts, median 423 per cohort.

THE THRESHOLD WAS FIXED BEFORE THE DATA EXISTED
------------------------------------------------
`OBJECTIVE-ACCURACY.md` §3.1, written 2026-09-04 before any of this was
downloaded: **a feature is established for a cohort only if the LOWER BOUND of
its 95% CI exceeds 0.55.**  Not the point estimate.  The three verdicts are:

    established   CI low  > 0.55     the asserted direction is real here
    contradicted  CI high < 0.50     it is real and BACKWARDS here
    underpowered  otherwise          this bank cannot tell, which is a fact
                                     about the bank and NOT about the person

⛔ 'underpowered' is not a polite 'contradicted'.  Collapsing them would tell
someone their face contradicts a pattern when the truth is nobody measured
enough faces like theirs.

THE CI IS BOOTSTRAPPED, NOT ANALYTIC.  DeLong's variance is a normal
approximation that is optimistic in the tails, and the tail is exactly where a
0.55 bound lives.

    ./venv/bin/python bank_dimorphism.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy import stats
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import dimorphism_bank_probe as DBP  # noqa: E402
import dimorphism_probability as DP  # noqa: E402

BANK = os.path.join(HERE, "bank")
OUT = os.path.join(BANK, "dimorphism.json")

ESTABLISHED_CI_LOW = 0.55   # §3.1, pre-registered 2026-09-04
CONTRADICTED_CI_HIGH = 0.50
N_BOOT = 2000


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(a random `pos` scores above a random `neg`), ties counted as half.

    ⚠️ `rankdata` rather than a hand-rolled tie loop: the hand-rolled version was
    correct and ~400x too slow to bootstrap 196 cohort-features, which is the
    difference between a measurement that gets made and one that gets skipped.
    """
    n1, n2 = len(pos), len(neg)
    if n1 == 0 or n2 == 0:
        return float("nan")
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n2))


def boot_ci(pos, neg, rng, n=N_BOOT):
    """Percentile bootstrap, resampling BOTH classes -- the AUC depends on both."""
    a = np.empty(n)
    for i in range(n):
        p = pos[rng.integers(0, len(pos), len(pos))]
        q = neg[rng.integers(0, len(neg), len(neg))]
        a[i] = auc(p, q)
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def main() -> int:
    z = np.load(os.path.join(BANK, "derived.npz"))
    prov = json.load(open(os.path.join(BANK, "provenance.json")))
    rng = np.random.default_rng(20260907)

    # A dimorphism cohort is a race x age band that has BOTH sexes.
    keys = {k.split("|")[0] for k in z.files}
    cohorts = sorted({c[2:] for c in keys if c.startswith("m_")}
                     & {c[2:] for c in keys if c.startswith("f_")})

    res = {
        "_what": "within-cohort dimorphism AUC with a bootstrapped 95% CI, "
                 "per feature per cohort.",
        "_threshold": {"established_ci_low_above": ESTABLISHED_CI_LOW,
                       "contradicted_ci_high_below": CONTRADICTED_CI_HIGH,
                       "_preregistered": "OBJECTIVE-ACCURACY.md §3.1, 2026-09-04, "
                                         "before any dataset was downloaded"},
        "bank_instrument_digest": prov["instrument"]["digest"],
        "n_bootstrap": N_BOOT,
        "cohorts": {},
    }
    tally = {"established": 0, "contradicted": 0, "underpowered": 0}
    pooled_rows = {}
    for fid, label, _fn, hi in DBP.FEATURES:
        pooled_rows[fid] = {"m": [], "f": [], "label": label, "hi": hi}

    for cohort in cohorts:
        row = {}
        for fid, label, _fn, hi in DBP.FEATURES:
            M = z[f"m_{cohort}|dim__{fid}"]
            F = z[f"f_{cohort}|dim__{fid}"]
            M, F = M[np.isfinite(M)], F[np.isfinite(F)]
            if len(M) < 20 or len(F) < 20:
                row[fid] = {"verdict": "underpowered", "why": "fewer than 20 per sex",
                            "n_male": int(len(M)), "n_female": int(len(F))}
                tally["underpowered"] += 1
                continue
            # orient so >0.5 always means "matches the asserted direction"
            pos, neg = (F, M) if hi == "F" else (M, F)
            a = auc(pos, neg)
            lo, hiCI = boot_ci(pos, neg, rng)
            if lo > ESTABLISHED_CI_LOW:
                v = "established"
            elif hiCI < CONTRADICTED_CI_HIGH:
                v = "contradicted"
            else:
                v = "underpowered"
            tally[v] += 1
            # `p`, `cohort_band` and `pooled_sd` exist so this record is a
            # DROP-IN for `race_reference.DIMORPHISM_BY_RACE`. Consumers read
            # `cohort_band` to re-band a lean against the cohort's own spread
            # rather than a global constant (analyze_advanced.py:771), so a
            # record missing it would silently stop re-banding while still
            # reading as "established".
            _u = stats.mannwhitneyu(pos, neg, alternative="two-sided")
            _sd = float(np.sqrt(((len(M) - 1) * M.var(ddof=1)
                                 + (len(F) - 1) * F.var(ddof=1))
                                / max(1, len(M) + len(F) - 2)))
            # ⛔ `cohort_band` IS A LIKELIHOOD-RATIO NEUTRAL BAND, NOT A SPREAD.
            # It is the region in which the male/female likelihood ratio stays
            # below DP.LEAN_ODDS -- "the evidence favours neither sex by more
            # than the odds threshold" -- and `analyze_advanced` re-bands every
            # lean against it. On 2026-09-07 this shim first served the pooled
            # p5-p95 instead, which is a DIFFERENT QUANTITY under the same name:
            # bands came out a median 1.47x wider, 32 established cohort-features
            # silently became underpowered, and the regression harness caught it
            # as moved dimorphism verdicts that looked like a deeper bank.
            # ⭐ THE SAME DEFECT CLASS THIS SESSION SHIPPED A GUARD FOR, one level
            # down: that guard watches KEYS, and this was a MEANING. Formula
            # copied from race_bank_probe.py so the two cannot drift again.
            muM, muF = float(M.mean()), float(F.mean())
            delta = muF - muM
            cband = None
            if abs(delta) > 1e-9 and _sd > 0:
                off = float(np.log(DP.LEAN_ODDS) * _sd * _sd / delta)
                mid = (muM + muF) / 2
                b0, b1 = sorted([mid - off, mid + off])
                cband = [round(b0, 4), round(b1, 4)]
            row[fid] = {"auc": round(a, 4), "ci": [round(lo, 4), round(hiCI, 4)],
                        "p": round(float(_u.pvalue), 6),
                        "verdict": v, "n_male": int(len(M)), "n_female": int(len(F)),
                        "cohort_band": cband,
                        "pooled_sd": round(_sd, 5),
                        "label": label, "higher_in": hi}
            pooled_rows[fid]["m"].append(M)
            pooled_rows[fid]["f"].append(F)
        res["cohorts"][cohort] = row

    # ⭐ THE COMPARISON THAT IS THE POINT: the same features, pooled, on exactly
    # the same faces. Any gap is inflation that pooling introduced.
    res["pooled_for_contrast"] = {}
    for fid, d in pooled_rows.items():
        if not d["m"]:
            continue
        M = np.concatenate(d["m"])
        F = np.concatenate(d["f"])
        pos, neg = (F, M) if d["hi"] == "F" else (M, F)
        a = auc(pos, neg)
        lo, hiCI = boot_ci(pos, neg, rng, n=500)
        per = [res["cohorts"][c][fid]["auc"] for c in cohorts
               if "auc" in res["cohorts"][c].get(fid, {})]
        res["pooled_for_contrast"][fid] = {
            "pooled_auc": round(a, 4), "pooled_ci": [round(lo, 4), round(hiCI, 4)],
            "median_within_cohort_auc": round(float(np.median(per)), 4),
            "inflation_from_pooling": round(a - float(np.median(per)), 4),
            "n_cohorts": len(per),
            "established_in": sum(1 for c in cohorts
                                  if res["cohorts"][c][fid]["verdict"] == "established"),
        }
    res["tally"] = tally
    res["n_cohorts"] = len(cohorts)
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)

    print(f"cohorts: {len(cohorts)}   verdicts: {tally}")
    print(f"\n{'feature':14s}{'pooled':>9s}{'within-med':>12s}{'inflation':>11s}"
          f"{'established':>13s}")
    for fid, v in res["pooled_for_contrast"].items():
        print(f"  {fid:12s}{v['pooled_auc']:9.3f}{v['median_within_cohort_auc']:12.3f}"
              f"{v['inflation_from_pooling']:+11.3f}"
              f"{v['established_in']:>9d}/{v['n_cohorts']}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
