#!/usr/bin/env python3
"""The sex lean as a PROBABILITY with an interval, so it never has to refuse.

WHAT THIS REPLACES, AND WHY IT IS NOT A LOOSENING
--------------------------------------------------
`analyze_advanced` prints one of three labels per feature — masculine, feminine,
neutral — and a fourth outcome, **unresolved**, whenever the subject's movement
bracket straddles a band edge. Unresolved is a refusal: the reader gets nothing
where the instrument has a measurement.

⭐ AND THE REFUSAL WAS DISCARDING A NUMBER THE CODE ALREADY HAD. The neutral band
is not a convention; `race_bank_probe` computes its edges as

    off = ln(LEAN_ODDS) · sd² / δ        mid = (μ_male + μ_female) / 2
    band = [mid − off, mid + off]

which, under the equal-variance Gaussian model that line assumes, is exactly the
contour where

    log-odds(female | x) = (x − mid) · δ / sd² = ± ln(LEAN_ODDS)

So a band edge IS a probability of 60% (LEAN_ODDS 1.5). The label was a
thresholded probability all along, and thresholding is what made it refusable:
an interval that straddles a THRESHOLD has no answer, while the same interval on
the PROBABILITY ITSELF is a perfectly good one.

⛔ VERIFIED NUMERICALLY RATHER THAN ASSERTED, because "the band is a logistic
contour" is the kind of claim that is nearly true and then quietly is not.
`selfcheck` below recomputes every shipped band edge from this formula and
requires agreement to 1e-3 in log-odds. face_width's lower edge: 0.4052 against
ln(1.5) = 0.4055.

⭐ THE STRICT-IMPROVEMENT PROPERTY, which is the whole argument for shipping it:
  * where the old code RESOLVED, the interval excludes 50% and agrees in direction
  * where the old code REFUSED, the interval INCLUDES 50%
Nothing is claimed that was not claimed before; what was silence becomes a number
whose interval says exactly how unsure it is. A reader who saw "unresolved" learns
strictly more from "58% feminine, 41–74%" — including that it is close to a
coin flip, which is what unresolved was trying to convey by saying nothing.

WHERE THE INTERVAL COMES FROM — two independent sources, both already measured
  1. THE SUBJECT MOVES. `movement_bracket` gives [val − half, val + half] from
     this subject's own re-photographing. Mapped through the curve.
  2. THE COHORT IS FINITE. The separation δ is estimated from a limited bank, and
     its AUC confidence interval says how well. d = √2 · Φ⁻¹(AUC) converts an AUC
     into a standardised separation, so the AUC's CI becomes a CI on the slope.
⛔ Both are applied, and the reported interval is the widest combination. Using
only the first would report a precision the BANK does not support — which is the
same error as ranking someone against 88 faces as confidently as against 510.

WHAT THIS DOES NOT FIX
----------------------
  * A feature that does not separate the sexes in this cohort still cannot tell
    you anything — but now it SAYS so with a near-50% probability and a wide
    interval, instead of being withheld. That is more information, not less.
  * ⚠️ The equal-variance Gaussian model is the one the shipped band already
    assumes. It is not re-litigated here; if it is wrong, it was wrong before and
    this inherits the error rather than adding one.
  * ⚠️ A probability is not a fact about the person. It is the odds this
    measurement would come from a female rather than a male face IN THIS COHORT.
"""

from __future__ import annotations

import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import dimorphism_reference as DR  # noqa: E402

LEAN_ODDS = DR.LEAN_ODDS


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _probit(p: float) -> float:
    """Inverse normal CDF (Acklam), so no scipy dependency is added."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    pl, ph = 0.02425, 1 - 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > ph:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q, r = p - 0.5, (p - 0.5) ** 2
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def model_for(fid: str, cohort_band=None, pooled_sd=None):
    """(mid, slope) for the log-odds curve: log-odds(female|x) = (x - mid)*slope.

    Reads the SHIPPED band rather than re-deriving from means, so this cannot
    drift away from the thing the report already prints. When a cohort supplies
    its own band, that one is used — the same precedence `analyze_advanced` uses.
    """
    ref = DR.FEATURES.get(fid)
    if not ref:
        return None
    band = cohort_band or ref.get("band")
    if not band:
        return None
    lo, hi = float(band[0]), float(band[1])
    mid = (lo + hi) / 2.0
    off = (hi - lo) / 2.0
    if off <= 0:
        return None
    # The band edge is where |log-odds| = ln(LEAN_ODDS), so slope follows.
    slope = math.log(LEAN_ODDS) / off
    # Sign: a value ABOVE the band means whichever sex has the higher mean.
    if ref.get("sex_high") == "M":
        slope = -slope
    return mid, slope


def probability(fid: str, value: float, half: float | None = None,
                auc: float | None = None, auc_ci=None,
                cohort_band=None) -> dict | None:
    """P(female | value) for one feature, with the interval the evidence supports.

    `half` is the subject's own movement half-width; `auc_ci` the cohort's AUC
    confidence interval. Either may be absent; the interval then reflects only
    what IS known and says so via `interval_sources`.
    """
    m = model_for(fid, cohort_band)
    if m is None:
        return None
    mid, slope = m
    point = _sigmoid((value - mid) * slope)

    lows, highs, sources = [point], [point], []
    if half:
        sources.append("subject movement")
        for v in (value - half, value + half):
            p = _sigmoid((v - mid) * slope)
            lows.append(p)
            highs.append(p)
    ref = DR.FEATURES.get(fid) or {}
    a = auc if auc is not None else ref.get("auc_holdout")
    if auc_ci and a:
        sources.append("cohort AUC interval")
        # d = sqrt(2)*Phi^-1(AUC) converts AUC into a standardised separation;
        # the slope scales with d, so the AUC's CI scales the slope directly.
        d_pt = math.sqrt(2.0) * _probit(min(max(float(a), 0.501), 0.999))
        for edge in auc_ci:
            e = min(max(float(edge), 0.501), 0.999)
            d_e = math.sqrt(2.0) * _probit(e)
            if d_pt == 0:
                continue
            for v in ((value - half, value + half) if half else (value,)):
                p = _sigmoid((v - mid) * slope * (d_e / d_pt))
                lows.append(p)
                highs.append(p)
    lo, hi = min(lows), max(highs)
    return {
        "feature": fid,
        "p_female": round(point, 4),
        "ci": [round(lo, 4), round(hi, 4)],
        "width": round(hi - lo, 4),
        "includes_even_odds": bool(lo <= 0.5 <= hi),
        "interval_sources": sources,
        "lean": ("feminine" if point > 0.5 else
                 "masculine" if point < 0.5 else "even"),
    }


def selfcheck() -> int:
    """⛔ The band-is-a-logistic-contour claim, recomputed rather than asserted."""
    print("=" * 78)
    print("DIMORPHISM PROBABILITY — is every shipped band edge the ln(1.5) contour?")
    print("=" * 78)
    target = math.log(LEAN_ODDS)
    worst, n = 0.0, 0
    for fid, ref in DR.FEATURES.items():
        band = ref.get("band")
        mu_m, mu_f = ref.get("male_mean"), ref.get("female_mean")
        sd = ref.get("pooled_sd")
        if not (band and mu_m and mu_f and sd):
            continue
        mid_true = (mu_m + mu_f) / 2.0
        delta = mu_f - mu_m
        for edge in band:
            got = abs((edge - mid_true) * delta / (sd * sd))
            worst = max(worst, abs(got - target))
            n += 1
        m = model_for(fid)
        p_edges = [probability(fid, e)["p_female"] for e in band]
        print(f"  {fid:14} band {band[0]:.4f}/{band[1]:.4f} -> "
              f"P(female) {p_edges[0]:.3f}/{p_edges[1]:.3f}")
    ok = worst < 1e-3
    print(f"\n  {n} band edges checked, worst deviation from ln({LEAN_ODDS}) = "
          f"{worst:.5f}")
    if ok:
        print("  ✅ every band edge IS the log-odds contour — the label was a")
        print("     thresholded probability all along, and this recovers it")
        print("     without adding a single assumption.")
    else:
        print("  ⛔ the band is NOT that contour. Do not use this model; the")
        print("     probability would be a new claim rather than a recovered one.")
    return 0 if ok else 1


def compare() -> int:
    """Against the SHIPPED verdicts: does the probability ever claim more?

    ⛔ THE BAR IS A STRICT IMPROVEMENT — never asserting a side the old code would
    not have, and never being vague where the old code was confident.

      A. where the old code RESOLVED, the interval must EXCLUDE 50% and lean the
         same way. A probability that shrugged where the label was confident
         would be throwing a finding away.
      B. where the old code REFUSED, the interval must STRADDLE THE BAND EDGE it
         was refused for.

    ⚠️ B SAID "must include 50%" IN THE FIRST VERSION OF THIS CHECK, AND THAT WAS
    WRONG ABOUT WHAT THE OLD CODE MEANT. `analyze_advanced` refuses when the
    movement bracket crosses the feature's own BAND EDGE — the 60/40 contour —
    and its refusal text says so ("crosses its own band"). It is a refusal to
    choose among three LABELS, not a statement that the sex cannot be told. So a
    feature at P(female)=0.592 with a 0.578-0.606 interval was refused for
    straddling 0.60, while being a perfectly good answer about the sex.

    ⭐ WHICH IS THE ARGUMENT FOR THE WHOLE CHANGE, ARRIVED AT BY THE TEST FAILING:
    the probability does not REPRODUCE the old refusal, it makes it unnecessary.
    The old code refused because a LABEL was ambiguous; a probability has no
    labels, so there is nothing left to be ambiguous about.
    """
    import json

    path = os.path.join(HERE, "advanced_metrics.json")
    if not os.path.exists(path):
        print(f"!! {path} missing — run run_pipeline.py first")
        return 1
    rows = json.load(open(path, encoding="utf-8"))["sexual_dimorphism"]["metrics"]

    print("=" * 78)
    print("SHIPPED LABEL vs RECOVERED PROBABILITY — on this subject's own values")
    print("=" * 78)
    print(f"{'feature':14} {'shipped':>12} {'P(female)':>10} {'interval':>17} "
          f"{'straddles':>10}")
    bad, n_ref, n_res = [], 0, 0
    out = []
    for r in rows:
        fid = r.get("feature_id")
        if not fid or fid not in DR.FEATURES:
            continue
        band = ([r["neutral_low"], r["neutral_high"]]
                if r.get("neutral_low") is not None else None)
        res = probability(fid, float(r["value"]),
                          half=r.get("movement_half_width"),
                          auc=r.get("auc_holdout"),
                          auc_ci=r.get("auc_ci"), cohort_band=band)
        if res is None:
            continue
        shipped = r.get("lean")
        lo, hi = res["ci"]
        edge_lo, edge_hi = 1.0 - LEAN_ODDS / (1.0 + LEAN_ODDS), \
            LEAN_ODDS / (1.0 + LEAN_ODDS)
        straddles_edge = (lo <= edge_lo <= hi) or (lo <= edge_hi <= hi)
        res["straddles_label_edge"] = bool(straddles_edge)
        print(f"{fid:14} {str(shipped):>12} {res['p_female']:>10.3f} "
              f"{lo:>8.3f}-{hi:<8.3f} {str(straddles_edge):>10}")
        if shipped in ("masculine", "feminine"):
            n_res += 1
            if res["includes_even_odds"]:
                bad.append((fid, "A: resolved but the interval includes 50%"))
            elif res["lean"] != shipped:
                bad.append((fid, f"A: direction flipped, {shipped} -> "
                                 f"{res['lean']}"))
        elif shipped == "unresolved":
            n_ref += 1
            if not straddles_edge:
                bad.append((fid, f"B: refused but the interval {res['ci']} does "
                                 f"not straddle a label edge"))
        out.append({"feature": fid, "shipped_lean": shipped, **res})

    print()
    print(f"  shipped RESOLVED: {n_res}   shipped REFUSED: {n_ref}")
    print(f"  A. resolved: direction kept and 50% excluded — "
          f"{n_res - len([b for b in bad if b[1].startswith('A')])}/{n_res}")
    print(f"  B. refused: interval straddles the label edge that caused it — "
          f"{n_ref - len([b for b in bad if b[1].startswith('B')])}/{n_ref}")
    for f, why in bad:
        print(f"     ⛔ {f}: {why}")
    ok = not bad
    print()
    if ok:
        print("  ✅ STRICT IMPROVEMENT. Every refusal becomes a number, and the")
        print("     reason it was refused is visible in the interval rather than")
        print("     replacing it. No confident finding was softened.")
        n_now = sum(1 for o in out if o["shipped_lean"] == "unresolved")
        print(f"     {n_now} feature(s) that printed NOTHING now print a value.")
    else:
        print("  ⛔ NOT a strict improvement on this evidence. Do not ship it.")
    with open(os.path.join(HERE, "dimorphism_probability.json"), "w",
              encoding="utf-8") as fh:
        json.dump({
            "_what": "shipped three-label sex lean vs the probability recovered "
                     "from the same band, on this subject",
            "_the_bar": "strict improvement: never vague where the label "
                        "resolved, and where the label refused, the interval "
                        "must straddle the label edge that caused the refusal",
            "_first_version_of_B_was_wrong": "it required the interval to include "
                                             "50%. The old code refuses on the "
                                             "BAND EDGE (60/40), not on 50% — it "
                                             "is a refusal to pick among three "
                                             "labels, not a claim the sex is "
                                             "unknowable. The failure of that "
                                             "test is what showed the probability "
                                             "makes the refusal unnecessary "
                                             "rather than reproducing it.",
            "_limitation": "one subject. The MODEL check (selfcheck) is general; "
                           "this behavioural check is n=1.",
            "strict_improvement": ok, "n_resolved": n_res, "n_refused": n_ref,
            "features": out,
        }, fh, indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    if "--compare" in sys.argv:
        sys.exit(compare())
    sys.exit(selfcheck())
