#!/usr/bin/env python3
"""Public implementation; private development notes omitted."""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

GATE_JSON = os.path.join(HERE, "metric_gate.json")
PHOTO_JSON = os.path.join(HERE, "metric_photographic_probe.json")
OUT = os.path.join(HERE, "metric_tiers.json")

# The capture configuration the product actually has. `metric_ship_probe` also
# measures a "+pitch" configuration, which is what a sentence in the capture brief
# would buy; reading it here would report accuracy from photographs nobody has
# taken.
CONFIG = "today (9 views, yaw only)"

TIER1_MAX_FRAC = 0.25  # = metric_error_bank.SHIP_MAX_FRAC_OF_SCALE, unchanged
TIER2_MAX_FRAC = 0.50  # Private development note omitted.
MIN_HOLDOUT_COVERAGE = 0.80  # = metric_error_bank.MIN_COVERAGE

# Angles read against a clinical description rather than a population percentile.
# The bands are angle_reproducibility_probe's, unchanged.
CLINICAL_BANDS = {
    "mandibular_mean": (
        "the 100-140&deg; band jaw angles are described on",
        100.0,
        140.0,
    ),
    "nasolabial": (
        "the 90-120&deg; band nasolabial angles are described on",
        90.0,
        120.0,
    ),
}

# Display metadata, in ONE table. Every metric the gate knows about must appear
# here or `build_table` refuses to emit it -- a measurement with no plain-English
# name would reach the report as a variable name, and a reader cannot audit a row
# they cannot read.
#   name, what it divides, unit, "higher means", where it is placed
LABELS: dict[str, dict] = {
    # ---- frontal proportions, placed against the 243-face reference bank ----
    "facial_thirds_ratio": {
        "name": "Facial thirds",
        "desc": "middle third &divide; lower third",
        "fmt": "{:.3f}",
        "placed_in": "bank",
    },
    "fwhr": {
        "name": "Facial width-to-height",
        "desc": "cheekbone width &divide; brow-to-lip height",
        "fmt": "{:.3f}",
        "placed_in": "bank",
    },
    "mouth_nose_width_ratio": {
        "name": "Mouth-to-nose width",
        "desc": "mouth width &divide; nose width",
        "fmt": "{:.3f}",
        "placed_in": "bank",
    },
    "intercanthal_ratio": {
        "name": "Eye spacing",
        "desc": "inner-corner &divide; outer-corner width",
        "fmt": "{:.3f}",
        "placed_in": "bank",
    },
    "jaw_taper_ratio": {
        "name": "Jaw taper",
        "desc": "jaw width &divide; cheekbone width",
        "fmt": "{:.3f}",
        "placed_in": "bank",
    },
    "lower_face_height_ratio": {
        "name": "Lower-face height",
        "desc": "nose-base-to-chin &divide; brow-to-chin",
        "fmt": "{:.3f}",
        "placed_in": "bank",
    },
    "nasal_width_ratio": {
        "name": "Nose width",
        "desc": "nose width &divide; cheekbone width",
        "fmt": "{:.3f}",
        "placed_in": "bank",
    },
    "eye_aspect_ratio": {
        "name": "Eye opening",
        "desc": "eye height &divide; eye width",
        "fmt": "{:.3f}",
        "placed_in": "bank",
    },
    "canthal_tilt_deg": {
        "name": "Canthal tilt",
        "desc": "outer eye corner vs inner",
        "fmt": "{:.1f}&deg;",
        "placed_in": "bank",
    },
    "chin_height_ratio": {
        "name": "Chin height",
        "desc": "lip-to-chin &divide; nose-base-to-chin",
        "fmt": "{:.3f}",
        "placed_in": "bank",
    },
    "lip_face_width_ratio": {
        "name": "Mouth width",
        "desc": "mouth width &divide; cheekbone width",
        "fmt": "{:.3f}",
        "placed_in": "bank",
    },
    "lip_height_ratio": {
        "name": "Lip fullness",
        "desc": "lip height &divide; mouth width",
        "fmt": "{:.3f}",
        "placed_in": "bank",
    },
    # ---- sagittal, placed against the FLAME population the scans corroborate --
    "facial_convexity": {
        "name": "Facial convexity",
        "desc": "brow &rarr; nose-base &rarr; chin, seen from the side",
        "fmt": "{:.1f}&deg;",
        "placed_in": "sagittal",
    },
    "nasofrontal_angle": {
        "name": "Nasofrontal angle",
        "desc": "brow &rarr; nose-bridge &rarr; nose-tip",
        "fmt": "{:.1f}&deg;",
        "placed_in": "sagittal",
    },
    "nasomental_angle": {
        "name": "Nasomental angle",
        "desc": "nose-bridge &rarr; nose-tip &rarr; chin",
        "fmt": "{:.1f}&deg;",
        "placed_in": "sagittal",
    },
    "nasal_projection_ratio": {
        "name": "Nasal projection",
        "desc": "how far the nose tip stands out, against nose height",
        "fmt": "{:.3f}",
        "placed_in": "sagittal",
    },
    "chin_projection_ratio": {
        "name": "Chin projection",
        "desc": "chin forwardness against face width",
        "fmt": "{:.3f}",
        "placed_in": "sagittal",
    },
    "midface_depth_ratio": {
        "name": "Midface depth",
        "desc": "nose tip vs cheekbone, against face width",
        "fmt": "{:.3f}",
        "placed_in": "sagittal",
    },
    "eline_upper_lip": {
        "name": "Upper lip to E-line",
        "desc": "Ricketts' nose-tip-to-chin line, % of face width",
        "fmt": "{:+.2f}%",
        "placed_in": "sagittal",
    },
    "eline_lower_lip": {
        "name": "Lower lip to E-line",
        "desc": "Ricketts' nose-tip-to-chin line, % of face width",
        "fmt": "{:+.2f}%",
        "placed_in": "sagittal",
    },
    # ---- angles read against a clinical band -------------------------------
    "mandibular_mean": {
        "name": "Jaw angle",
        "desc": "soft-tissue mandibular contour, both sides averaged",
        "fmt": "{:.1f}&deg;",
        "placed_in": "clinical",
    },
    "nasolabial": {
        "name": "Nasolabial angle",
        "desc": "columella &rarr; nose-base &rarr; upper lip",
        "fmt": "{:.1f}&deg;",
        "placed_in": "clinical",
    },
}


def tier_for(
    geometric_error: float | None,
    photographic_bar: float | None,
    scale: float,
    holdout_coverage: float | None,
) -> dict:
    """The tier one metric has earned, from its two measured errors.

    ⛔ MISSING IS REFUSED, NEVER PASSED. A metric with no photographic term has
    never been read off a real photograph, and one with no geometric term has
    never been checked against a truth. Treating an absent measurement as a
    passing one is how an untested path ships; this project has already had a
    defect live undetected in the one arm no conclusion depended on.
    """
    if not scale or scale <= 0 or scale != scale:
        return _refuse("no judged scale — nothing to compare an error against")
    if geometric_error is None or geometric_error != geometric_error:
        return _refuse("never checked against ground truth on a real scanned face")
    if photographic_bar is None or photographic_bar != photographic_bar:
        return _refuse("never read from a real photograph of an unchanging face")
    if holdout_coverage is None or holdout_coverage < MIN_HOLDOUT_COVERAGE:
        cov = "unknown" if holdout_coverage is None else f"{holdout_coverage:.0%}"
        return _refuse(
            f"the bar covers only {cov} of the subjects it was not fitted on — "
            "it describes the faces that built it"
        )

    bar = max(float(geometric_error), float(photographic_bar))
    frac = bar / float(scale)
    binding = "photographs" if photographic_bar >= geometric_error else "geometry"
    if frac < TIER1_MAX_FRAC:
        tier, why = 1, (
            f"error is {frac:.0%} of the judged scale, so a placement holds"
        )
    elif frac < TIER2_MAX_FRAC:
        tier, why = 2, (
            f"error is {frac:.0%} of the judged scale — the number is real, "
            "a placement is not"
        )
    else:
        tier, why = 0, (
            f"error is {frac:.0%} of the judged scale, so the reading is not "
            "distinguishable from much of the population"
        )
    return {
        "tier": tier,
        "bar": bar,
        "frac_of_scale": frac,
        "binding": binding,
        "geometric_error": float(geometric_error),
        "photographic_bar": float(photographic_bar),
        "scale": float(scale),
        "holdout_coverage": float(holdout_coverage),
        "why": why,
    }


def _refuse(why: str) -> dict:
    return {
        "tier": 0,
        "bar": None,
        "frac_of_scale": None,
        "binding": None,
        "geometric_error": None,
        "photographic_bar": None,
        "scale": None,
        "holdout_coverage": None,
        "why": why,
    }


def build_table(gate: dict, photo: dict, config: str = CONFIG) -> dict:
    """Every metric the gate knows about, with its tier and the numbers behind it."""
    rows = {}
    metrics = gate["configurations"][config]["metrics"]
    for key, g in metrics.items():
        if key not in LABELS:
            raise KeyError(
                f"{key} has no entry in metric_tiers.LABELS — a measurement with "
                "no plain-English name cannot be printed, and silently dropping "
                "it would shrink the report without anyone deciding to"
            )
        band = CLINICAL_BANDS.get(key)
        scale = (band[2] - band[1]) if band else g["scale"]
        row = tier_for(
            g.get("geometric_error"),
            (photo["metrics"].get(key) or {}).get("bar"),
            scale,
            g.get("holdout_coverage"),
        )
        row.update(LABELS[key])
        row["band"] = [band[1], band[2]] if band else None
        row["judged_on"] = (
            band[0]
            if band
            else "the spread of real face geometry (FLAME identities, "
            "corroborated on ten real scanned humans)"
        )
        rows[key] = row
    return rows


def load(config: str = CONFIG) -> dict:
    """The tier table, computed from the two probes' outputs."""
    for path, script in (
        (GATE_JSON, "metric_gate.py"),
        (PHOTO_JSON, "metric_photographic_probe.py"),
    ):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{os.path.basename(path)} absent — run {script}")
    gate = json.load(open(GATE_JSON, encoding="utf-8"))
    photo = json.load(open(PHOTO_JSON, encoding="utf-8"))
    return build_table(gate, photo, config)


# The thirds a placement is described on. Deliberately the report's existing
# cutoffs (`build_report.SYMTYP_CUTOFFS`), not new ones.
THIRD_CUTOFFS = (33.0, 67.0)


def _third(pct: float) -> str:
    if pct < THIRD_CUTOFFS[0]:
        return "bottom third"
    if pct > THIRD_CUTOFFS[1]:
        return "top third"
    return "middle third"


def placement(pct: float | None, pct_lo: float | None, pct_hi: float | None) -> dict:
    """Which third of the population, and a refusal when the bar straddles a line.

    ⛔ THE PHILTRUM RULE, THIRD APPLICATION. A measurement whose interval crosses
    the boundary it is being placed against does not get the placement word, only
    the number: the philtrum shipped "long" on a 16.2mm reading whose bracket ran
    14.6-17.5mm across a 15mm cutoff, and averageness lost its placement the same
    way. A bar that is small enough to earn tier 1 in general can still be too
    large for one particular subject sitting on a line, and the tier alone cannot
    know that -- it is a property of where THIS face landed.
    """
    if pct is None:
        return {"third": None, "why": "no population to place this reading in"}
    if pct_lo is None or pct_hi is None:
        return {"third": None, "why": "the placement interval could not be computed"}
    lo, hi = _third(min(pct_lo, pct_hi)), _third(max(pct_lo, pct_hi))
    if lo != hi:
        return {
            "third": None,
            "why": (
                f"this reading's own interval runs from the {lo} to the {hi}, so "
                "it crosses the line the placement would be made on"
            ),
        }
    return {"third": lo, "why": f"the whole interval falls in the {lo}"}


def measurement_rows(
    table: dict,
    values: dict[str, float],
    rank: "callable",
) -> dict:
    """Subject values joined to their tiers, with placements where tier 1 allows.

    `rank(metric, value)` returns a percentile in that metric's own population, or
    None. It is passed in rather than imported so this module never opens the
    reference banks -- one of them needs the licensed scans to build.
    """
    import math

    out = {}
    for key, t in table.items():
        row = dict(t)
        v = values.get(key)
        row["value"] = None if v is None or not math.isfinite(float(v)) else float(v)
        if row["value"] is None:
            # ⛔ NOT SILENTLY OMITTED. A metric the gate cleared but this fit did
            # not produce is a different failure from one that was refused, and a
            # row that vanishes reads as a measurement nobody ever tried.
            row["tier"] = 0
            row["why"] = "the fit did not produce this measurement"
            out[key] = row
            continue
        # ⛔ A FIGURE THAT MUST NOT BE PRINTED IS NOT WRITTEN DOWN. This project
        # has caught a refused figure reappearing one paragraph later three
        # times, and the reason is always that the number stayed in the artifact
        # and something downstream found it. So a tier-2 row carries NO
        # percentile at all, and a refused row carries no value: not printed
        # because there is nothing to print, rather than because a template
        # remembered to leave it out.
        if row["tier"] == 1 and row["bar"] is not None:
            pct = rank(key, row["value"])
            lo = rank(key, row["value"] - row["bar"])
            hi = rank(key, row["value"] + row["bar"])
            row["percentile"], row["percentile_lo"], row["percentile_hi"] = pct, lo, hi
            row["placement"] = placement(pct, lo, hi)
        elif row["tier"] == 2:
            row["percentile"] = row["percentile_lo"] = row["percentile_hi"] = None
            row["placement"] = {
                "third": None,
                "why": "tier 2 prints the measurement and makes no placement claim",
            }
        else:
            row["value"] = None  # refused: the bar and the reason print, not a number
            row["percentile"] = row["percentile_lo"] = row["percentile_hi"] = None
            row["placement"] = {"third": None, "why": row["why"]}
        out[key] = row
    return out


def hedge(key: str, table: dict | None = None) -> str:
    """The sentence a module must attach when it names a tier-2 measurement.

    ⭐ THIS EXISTS BECAUSE A TWO-TIER TABLE ALONE WOULD BE COSMETIC. A tier-2 row
    says "no placement claim" while another section three pages earlier calls the
    same number "mildly wide-set" or "≈ average" hands the reader the placement
    anyway, and hands it to them FIRST. This project has caught a refused figure
    reappearing one paragraph later three times; the general form of the fix is
    that a claim word about a gated measurement has to come from the gate.

    Returns "" for tier 1 (the placement is earned) and for anything the table
    does not know about, so a caller can concatenate it unconditionally.
    """
    return hedges([key], table)


def hedges(keys: list[str], table: dict | None = None) -> str:
    """One hedge sentence covering several tier-2 measurements at once.

    Called per metric it produced the same sentence twice in a row wherever a
    structure has two gated measurements (the canthi have both eye spacing and
    canthal tilt), which reads as a template misfiring rather than as a caveat.
    """
    try:
        tbl = table if table is not None else load()
    except Exception:  # noqa: BLE001 - a missing probe must not break a stage
        return ""
    rows = [
        tbl[k] for k in keys if tbl.get(k) and tbl[k]["tier"] == 2 and tbl[k]["bar"]
    ]
    if not rows:
        return ""
    parts = []
    for r in rows:
        fmt = (r.get("fmt") or "{:.3g}").replace(":+", ":").replace("&deg;", "°")
        parts.append(
            f"{r['name'].lower()} ±{fmt.format(r['bar'])} "
            f"({r['frac_of_scale']:.0%} of the range real faces occupy)"
        )
    return (
        " Measured error bars: "
        + "; ".join(parts)
        + ". No placement against other faces is claimed for "
        + ("either" if len(parts) == 2 else "these" if len(parts) > 2 else "it")
        + " — see the measurements section."
    )


def refuses_placement(key: str, table: dict | None = None) -> bool:
    """True when this metric may be printed but not placed."""
    try:
        t = (table if table is not None else load()).get(key)
    except Exception:  # noqa: BLE001
        return False
    return bool(t and t["tier"] != 1)


def band_position(value: float, band_lo: float, band_hi: float) -> float:
    """Where a reading sits inside a clinical band, 0-100. Clamped, not wrapped."""
    span = band_hi - band_lo
    return float(min(100.0, max(0.0, 100.0 * (value - band_lo) / span)))


def counts(table: dict) -> dict[str, int]:
    return {
        "tier1": sum(1 for r in table.values() if r["tier"] == 1),
        "tier2": sum(1 for r in table.values() if r["tier"] == 2),
        "refused": sum(1 for r in table.values() if r["tier"] == 0),
    }


def main() -> int:
    table = load()
    n = counts(table)
    print("=" * 78)
    print("THE TIERS — what each measurement has earned the right to say")
    print("=" * 78)
    print(
        f"  tier 1 (< {TIER1_MAX_FRAC:.0%} of scale, placed)     : {n['tier1']}\n"
        f"  tier 2 ({TIER1_MAX_FRAC:.0%}-{TIER2_MAX_FRAC:.0%}, number only)   : {n['tier2']}\n"
        f"  refused (>= {TIER2_MAX_FRAC:.0%} or unmeasured) : {n['refused']}\n"
    )
    print(
        f"  {'metric':<26}{'bar':>10}{'scale':>10}{'% scale':>9}" f"{'binds':>13}  tier"
    )
    order = {1: 0, 2: 1, 0: 2}
    for k, r in sorted(
        table.items(),
        key=lambda kv: (order[kv[1]["tier"]], kv[1]["frac_of_scale"] or 9),
    ):
        if r["tier"] == 0 and r["bar"] is None:
            print(f"  {k:<26}{'—':>10}{'—':>10}{'—':>9}{'—':>13}  REFUSED")
            continue
        print(
            f"  {k:<26}{r['bar']:10.4f}{r['scale']:10.4f}"
            f"{r['frac_of_scale'] * 100:8.1f}%{r['binding']:>13}"
            f"  {['REFUSED', 'TIER 1', 'TIER 2'][r['tier']]}"
        )

    payload = {
        "_what": "what each measurement has earned the right to say: tier 1 "
        "carries a placement verdict, tier 2 prints a number and its measured "
        "bar with no placement, and a refusal prints neither",
        "_bar": "the WORSE of the geometric error across ten real scanned humans "
        "and the photographic movement across every accepted fit of one "
        "unchanging face. Both are the 95th percentile of an absolute deviation, "
        "so the max() compares like with like.",
        "_no_posterior": "the fit's propagated posterior is deliberately absent — "
        "it draws landmark errors independently where they are correlated, and "
        "was measured overstating some metrics 50x",
        "_thresholds": {
            "tier1_max_frac_of_scale": TIER1_MAX_FRAC,
            "tier2_max_frac_of_scale": TIER2_MAX_FRAC,
            "min_holdout_coverage": MIN_HOLDOUT_COVERAGE,
        },
        "_config": CONFIG,
        "counts": n,
        "metrics": table,
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n  wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
