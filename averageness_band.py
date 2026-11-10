#!/usr/bin/env python3
"""Shape typicality: the posture gate, the refusal, and the axes that survive.

WHAT WENT WRONG WITH THE FIGURE THIS REPLACES
----------------------------------------------
Shape typicality moved **18.0-27.8 percentile points** on one unchanging face
photographed again inside the pose gate. It was diagnosed as detector jitter
amplified by the percentile transform. The amplification is real -- about 19
percentile points per SD of the underlying distance, because the subject sits
where the reference bank is densest -- but the jitter was not the source.

`averageness_posture_probe.py` measured the source: **mouth posture**.
`mouthPucker` correlates **-0.92 with the dominant shape axis** and **+0.91 with
the percentile itself** across the reference subject's in-gate frontals. The
readings cluster by sitting, which additive landmark error cannot produce.

⭐ AND WHY NO CORRECTION RESCUES IT. Across the BANK, that axis is real anatomy
(lower-face vertical proportion, r = -0.86 with the midface/lower-face ratio).
WITHIN one person it is posture. Different directions in shape space, and a bank
holding one photograph per person cannot teach the postural one. Four repairs
were tried and all four are recorded in the probe rather than forgotten.

WHAT THIS MODULE PROVIDES
-------------------------
  1. **A capture gate** on mouth posture, parallel to the pose gate the pipeline
     already enforces. The reference subject's sittings separate 2.3x wide with
     nothing in the gap, so the threshold does no delicate work -- the standard
     the 26-degree intake threshold was held to.
  2. **The refusal.** Even gated, the percentile swings 7.4pp against placement
     cutoffs 1.5pp away, so it does not resolve and does not ship.
  3. **The axes that do survive** -- signed projections onto the bank's own shape
     axes, each with an error bar, shipped only when they clear the project's
     <25%-of-judged-scale rule.

⭐ THE STRUCTURAL REASON THE AXES WORK WHERE THE PERCENTILE DOES NOT: typicality
is a DISTANCE, i.e. a MAGNITUDE, so noise can only ADD to it -- the same hazard
that biased the asymmetry figure, flattered the multi-view fit in SS5.1, and
forced the acne grade into a bracket. A signed projection lets noise CANCEL.

⭐ SS5.1 IS UNTOUCHED BY CONSTRUCTION. The subject is still read from ONE raw
photograph by the BANK'S OWN plain detector. Nothing here averages, refines, or
multi-view-fits it, so no processing asymmetry is introduced between subject and
bank. The gate SELECTS a photograph; it does not clean one.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Measured constants. Each is derived by a script in this repo, and
# `selfcheck_averageness_axes.py` re-runs those scripts and asserts these still
# match -- a constant copied by hand from a probe is a constant that drifts.
# ---------------------------------------------------------------------------

# `noise_correlation_probe.py`: 1.5% of IPD, 13% of the variance correlated over
# 0.18x IPD, measured from upright-vs-mirrored detection of real photographs.
NOISE_FRAC = 0.015
CORR_SHARE = 0.13
CORR_LEN_IPD = 0.18

# `averageness_posture_probe.py`. The reference subject's sittings sit at
# 0.12-0.30 and 0.68-0.81 with NOTHING between, so this threshold is the midpoint
# of an empty region rather than a cut through a cluster.
# ⭐ It is also EXOGENOUS: "do not purse your lips" is a capture-quality
# instruction that stands on its own, so it is not a threshold fitted to an
# answer. And it moves in the UNFLATTERING direction -- the relaxed-mouth
# photographs read LESS typical -- so it cannot be selecting the pleasing half.
MOUTH_PUCKER_GATE = 0.40

# ⚠️ OUTER-CANTHAL, NOT IRIS, AND THE DISTINCTION COST A RUN. The session probe
# reports IPD from the IRIS centres; this module measures landmarks 33/263, the
# outer eye corners, which run 1.402x larger on the same face. A 200px iris
# threshold applied to an outer-canthal measurement excludes nothing. This
# project has already inflated one result 52x and another 2.6x by mixing those
# two rulers, so the ruler is in the NAME.
MIN_IPD_PX_OUTER = 280.0  # = 200px iris-equivalent

K_DRAWS = 200
BAND_Q = (2.5, 97.5)

# The judged scale for the <25% rule is the bank's 4-SD range, stated rather than
# left implicit: a threshold whose denominator is invisible can be moved without
# anyone noticing.
SHIP_MAX_BAND_FRAC = 0.25
SCALE_SD_RANGE = 4.0

IPD_PAIR = (33, 263)
MOUTH_CORNERS = (61, 291)


def mouth_posture_proxy(lms: np.ndarray) -> float:
    """Mouth width over IPD — the landmark-only stand-in for `mouthPucker`.

    r = -0.94 against the blendshape across the reference subject's frontals.
    ⛔ NOT what the gate uses. The gate reads the blendshape directly, because
    the blendshape separates the sittings 2.3x wide while this proxy's values sit
    0.541-0.586 — inside half a bank SD of each other, so any threshold on it
    would be knife-edge. This exists because the REFERENCE BANK has no
    blendshapes (source photographs deleted, landmarks only), so it is the only
    way to ask whether the bank's own mouth-width spread is anatomy or posture.
    It is anatomy: r = +0.21 with the dominant axis across the bank, against
    +0.89 within one person. That contrast is the whole diagnosis.
    """
    return float(
        np.linalg.norm(lms[MOUTH_CORNERS[0]] - lms[MOUTH_CORNERS[1]])
        / np.linalg.norm(lms[IPD_PAIR[0]] - lms[IPD_PAIR[1]])
    )


def posture_ok(expression: dict) -> tuple[bool, str]:
    """Does this photograph's mouth posture clear the gate?

    Returns (ok, reason). An absent blendshape is NOT a pass: a detection that
    could not report expression cannot be certified against a gate defined on
    expression, and this project's standing rule is that a failed read must not
    look like a clean result.
    """
    if "mouthPucker" not in expression:
        return False, "no expression reading, so the mouth posture is unverified"
    v = float(expression["mouthPucker"])
    if v > MOUTH_PUCKER_GATE:
        return False, (
            f"lips pursed (mouthPucker {v:.2f} over {MOUTH_PUCKER_GATE:.2f}) — "
            "re-shoot with a relaxed mouth, lips together and not pushed forward"
        )
    return True, f"mouth relaxed (mouthPucker {v:.2f})"


class NoiseSampler:
    """Draws detector-shaped noise fields for one photograph's landmark layout.

    The kernel depends only on where the landmarks sit, so it is factorised once
    per photograph and reused across draws; otherwise the Cholesky dominates.
    """

    def __init__(self, pts: np.ndarray) -> None:
        self.ipd = float(np.linalg.norm(pts[IPD_PAIR[0]] - pts[IPD_PAIR[1]]))
        d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1) / self.ipd**2
        k = np.exp(-0.5 * d2 / (CORR_LEN_IPD**2)) + 1e-9 * np.eye(len(pts))
        self.chol = np.linalg.cholesky(k)
        self.n = len(pts)

    def draw(self, mult: float, rng: np.random.Generator) -> np.ndarray:
        sigma = mult * NOISE_FRAC * self.ipd
        smooth = self.chol @ rng.normal(size=(self.n, 2))
        white = rng.normal(size=(self.n, 2))
        return sigma * (
            np.sqrt(CORR_SHARE) * smooth + np.sqrt(1.0 - CORR_SHARE) * white
        )


def bank_context(configs, perm) -> dict:
    """Everything about one reference bank that does not depend on the subject.

    The axes are the PCA of the bank in the SAME symmetric, aligned, flattened
    space the typicality distance is measured in, so an axis is a decomposition
    of that distance rather than a second unrelated construct.
    """
    import analyze_averageness2 as A2

    configs = list(configs)
    mean_full, _ = A2.gpa(configs)
    sym_mean = A2.standardize(A2.symmetric_part(mean_full, perm))
    loo_sym = A2.loo_euclidean_distances(configs, perm=perm)

    bank = np.stack([A2.aligned_sym_vec(c, sym_mean, perm) for c in configs])
    mu = bank.mean(0)
    _u, s, vt = np.linalg.svd(bank - mu, full_matrices=False)

    # SIGN IS ARBITRARY OUT OF THE SVD AND MUST NOT BE. "+0.9 SD" and "-0.9 SD"
    # are opposite statements about a face; a sign that flips between runs would
    # invert every one of them while the report still looked plausible. This
    # project has shipped two inversions already — a render that was upside down
    # for two sessions because a symmetric face looks normal flipped, and canthal
    # tilt scored against a mirrored population. Deterministic rule: the
    # largest-magnitude loading is positive.
    for a in range(len(vt)):
        if vt[a][np.argmax(np.abs(vt[a]))] < 0:
            vt[a] = -vt[a]

    scores = (bank - mu) @ vt.T
    return {
        "sym_mean": sym_mean,
        "loo_sym": loo_sym,
        "loo_mu": float(loo_sym.mean()),
        "loo_sd": float(loo_sym.std()),
        "pcs": vt,
        "mu": mu,
        "axis_sd": scores.std(0),
        "axis_var_share": s**2 / (s**2).sum(),
        "bank_scores": scores,
        "n": len(configs),
        "perm": perm,
    }


def read(lms: np.ndarray, ctx: dict, n_axes: int = 6) -> dict:
    """One reading: the typicality percentile, its z, and the signed axis scores."""
    import analyze_averageness2 as A2

    v = A2.aligned_sym_vec(lms, ctx["sym_mean"], ctx["perm"])
    d = float(np.linalg.norm(v - ctx["sym_mean"].reshape(-1)))
    return {
        "pct": 100.0 * float(np.mean(ctx["loo_sym"] > d)),
        "z": float((d - ctx["loo_mu"]) / (ctx["loo_sd"] + 1e-12)),
        "axes": (v - ctx["mu"])
        @ ctx["pcs"][:n_axes].T
        / (ctx["axis_sd"][:n_axes] + 1e-12),
        "distance": d,
    }


def band(
    lms: np.ndarray,
    ctx: dict,
    mult: float,
    rng: np.random.Generator,
    k: int = K_DRAWS,
    n_axes: int = 6,
    sampler: NoiseSampler | None = None,
) -> dict:
    """Monte Carlo error bars for one photograph, at `mult` x the measured noise.

    ⚠️ THIS IS A FLOOR, NOT THE ERROR. It propagates the noise that WAS measured,
    and the posture probe showed measured noise is not the whole story — it
    predicts 9.3-12.6pp where 18.0-27.8pp is observed. Anything shipped on this
    must also clear the observed re-shoot movement, which is why
    `axis_ships` takes both and uses the worse.
    """
    sampler = sampler or NoiseSampler(lms)
    pcts, zs = np.empty(k), np.empty(k)
    axes = np.empty((k, n_axes))
    for i in range(k):
        r = read(lms + sampler.draw(mult, rng), ctx, n_axes)
        pcts[i], zs[i], axes[i] = r["pct"], r["z"], r["axes"]
    lo, hi = BAND_Q
    return {
        "pct_lo": float(np.percentile(pcts, lo)),
        "pct_hi": float(np.percentile(pcts, hi)),
        "z_lo": float(np.percentile(zs, lo)),
        "z_hi": float(np.percentile(zs, hi)),
        "axis_lo": np.percentile(axes, lo, axis=0),
        "axis_hi": np.percentile(axes, hi, axis=0),
    }


# MEASURED per axis by `averageness_posture_probe.py`, on the reference subject's
# GATED frontals. Two numbers per axis: the Monte Carlo floor (what the measured
# detector noise predicts) and the movement actually observed across re-shoots
# that passed the posture gate.
#
# ⚠️ n = 1 SUBJECT, 4 PHOTOGRAPHS, 2 SITTINGS. These are one person's re-shoot
# behaviour standing in for everyone's, and a range over four draws is biased
# LOW. That is exactly why an axis must ALSO clear the floor and the residual
# test below — the observed swing on its own could not fail.
#
# They are constants rather than a JSON read so this works on a fresh clone;
# `selfcheck_averageness_axes.py` re-runs the probe and asserts they still match.
AXIS_REPRO = {
    #  bank:          [(mc_floor_sd, gated_reshoot_sd, var_share), ...] per axis
    "white_10_19": [
        (0.21, 0.34, 0.411),
        (0.27, 0.62, 0.239),
        (0.27, 0.21, 0.187),
        (0.47, 0.18, 0.055),
        (0.59, 0.56, 0.029),
        (0.72, 0.19, 0.017),
    ],
    "white_20_29": [
        (0.21, 0.34, 0.429),
        (0.28, 0.30, 0.244),
        (0.32, 0.66, 0.139),
        (0.44, 0.25, 0.066),
        (0.52, 0.40, 0.036),
        (0.86, 1.13, 0.017),
    ],
    "white_all": [
        (0.21, 0.35, 0.412),
        (0.29, 0.43, 0.231),
        (0.27, 0.56, 0.172),
        (0.44, 0.15, 0.064),
        (0.53, 0.45, 0.034),
        (0.80, 0.19, 0.017),
    ],
    "broad_10_19": [
        (0.23, 0.38, 0.385),
        (0.28, 0.65, 0.226),
        (0.32, 0.14, 0.172),
        (0.49, 0.27, 0.068),
        (0.47, 0.52, 0.053),
        (0.88, 0.49, 0.021),
    ],
}

# An axis explaining 2% of a face's shape variance is not a description of that
# face, it is a description of the residual. Reporting one would be mining detail
# the benchmark asks for out of noise the instrument cannot support.
# MEASURED, `averageness_posture_probe.py`: the typicality percentile's swing
# across the reference subject's GATED re-shoots. The gate cuts it from
# 23.6-30.2pp to these, which is a large repair and still not enough — the
# placement cutoffs sit 1.5pp from the subject's reading.
GATED_TYPICALITY_SWING_PP = {
    "white_10_19": 9.5,
    "white_20_29": 13.9,
    "white_all": 13.3,
    "broad_10_19": 7.4,
}


def gated_swing_for(cohort: str) -> float:
    """The measured movement for a bank, or the WORST measured one if it has none.

    ⛔ AN UNMEASURED COHORT INHERITS THE WORST FIGURE, NEVER A BETTER ONE. The
    banks above were measured on one subject's gated re-shoots; the female and
    broader-age banks added on 2026-08-15 have no such measurement, because
    nobody has re-photographed a subject against them. Defaulting to the mean, or
    to the nearest-looking bank, would hand a reader a tighter interval than
    anything anyone has established for them.

    ⚠️ AND THE INHERITED FIGURE IS STILL FROM ONE PERSON. It is a floor carried
    across, not a property of the new bank. That travels into the report wherever
    the interval prints.
    """
    return GATED_TYPICALITY_SWING_PP.get(
        cohort, max(GATED_TYPICALITY_SWING_PP.values())
    )


def swing_is_measured_for(cohort: str) -> bool:
    """Whether this bank's movement was measured or inherited. The report says which."""
    return cohort in GATED_TYPICALITY_SWING_PP


def axis_repro_for(cohort: str) -> list[tuple[float, float, float]]:
    """Per-axis (floor, swing, var_share) for a bank, measured or inherited.

    Same discipline as `gated_swing_for`, for the same reason: the four banks in
    AXIS_REPRO were characterised on one subject's re-shoots and the four added on
    2026-08-15 were not. An unmeasured bank inherits the WORST measured floor and
    the WORST measured swing axis by axis, and the SMALLEST variance share -- both
    directions chosen so an unmeasured bank can only ship FEWER axes than a
    measured one, never more.

    ⛔ Falling back to the nearest-looking bank would be the tempting version and
    is wrong: "the female bank probably behaves like the white male one" is
    precisely the assumption the all-male bank defect was made of.
    """
    if cohort in AXIS_REPRO:
        return AXIS_REPRO[cohort]
    n = min(len(v) for v in AXIS_REPRO.values())
    return [
        (
            max(v[a][0] for v in AXIS_REPRO.values()),
            max(v[a][1] for v in AXIS_REPRO.values()),
            min(v[a][2] for v in AXIS_REPRO.values()),
        )
        for a in range(n)
    ]


def axis_repro_is_measured_for(cohort: str) -> bool:
    """Whether this bank's per-axis reproducibility was measured or inherited."""
    return cohort in AXIS_REPRO


# ---------------------------------------------------------------------------
# WHICH BANKS A SUBJECT IS COMPARED AGAINST
# ---------------------------------------------------------------------------
# ⭐ THIS EXISTS BECAUSE THE LIST WAS HARDCODED TO THE MALE BANKS AND THAT SURVIVED
# THE REBUILD THAT ADDED THE FEMALE ONES. Eight banks were downloaded, landmarked
# and cached on 2026-08-15, every label was rewritten to name a sex -- and
# `analyze_averageness2.main` still scored exactly four, all male. A female
# subject therefore produced an averageness file with no entry for their own
# cohort, and the run died in `social_perception` on a KeyError (found 2026-08-16
# by running one). Loud, at least: nothing ranked a woman against men. But the
# banks "loading" was never the same thing as the pipeline USING them, and only a
# real run could tell the two apart.
#
# The family is derived from the subject's own cohort name, so adding a bank
# cannot leave a subject scored against somebody else's.
# The one place a bank's name becomes English. It was written out separately in
# the report and again in the figure, which is how "white_20_29" could mean "White
# young-adult male" in one and "White 20-29" in the other while neither said male.
COHORT_LABEL = {
    "white_10_19": "White male 10–19",
    "white_20_29": "White young-adult male",
    "white_all": "White male, all ages",
    "broad_10_19": "all-race male 10–19",
    "broad_20_29": "all-race young-adult male",
    "female_10_19": "White female 10–19",
    "female_20_29": "White young-adult female",
    "female_all": "White female, all ages",
    "female_broad_10_19": "all-race female 10–19",
    "female_broad_20_29": "all-race young-adult female",
}


# ⭐ RACE-MATCHED BANKS, generated rather than typed so a new cohort cannot ship
# with a missing or stale label. `label_for` falls back to the raw bank name for
# anything unknown, which is safe but unreadable, and unreadable is how
# "white_20_29" got read as neutral for years.
_RACE_PRETTY = {
    "east_asian": "East Asian", "indian": "Indian", "black": "Black",
    "white": "White", "middle_eastern": "Middle Eastern",
    "latino_hispanic": "Latino/Hispanic", "southeast_asian": "Southeast Asian",
}
for _r, _pretty in _RACE_PRETTY.items():
    for _b in ("20_29", "30_39", "40_49"):
        _age = _b.replace("_", "–")
        COHORT_LABEL[f"m_{_r}_{_b}"] = f"{_pretty} male {_age}"
        COHORT_LABEL[f"f_{_r}_{_b}"] = f"{_pretty} female {_age}"


def race_matched_family(fam: dict, race_cohort: str | None, available) -> dict:
    """Put the subject's race+sex+age bank first, keeping the legacy ones beside.

    ⛔ THE LEGACY BANKS ARE NOT DROPPED. They are what every published figure,
    guard and paragraph in this project was written against, and a comparison
    that silently disappears is how a reader loses the ability to see that the
    population moved the answer. Same resolution as the all-race pairing on
    2026-08-19: the better-matched bank LEADS, the older one stays visible.

    ⚠️ The race banks carry no MEASURED axis reproducibility, so `axis_repro_for`
    hands them the worst measured floor and swing and the smallest variance
    share. They can therefore only ever ship FEWER axes than a characterised
    bank, never more — which is the correct direction for a bank nobody has
    re-photographed anybody against.
    """
    if not race_cohort or race_cohort not in set(available):
        return fam
    out = dict(fam)
    out["race_matched"] = race_cohort
    out["primary"] = race_cohort
    order = [race_cohort] + [c for c in fam["order"] if c != race_cohort]
    out["order"] = order
    return out


def label_for(cohort: str) -> str:
    """English for a bank name. An unknown bank keeps its raw name rather than
    being described as something it may not be."""
    return COHORT_LABEL.get(cohort, cohort)


MATCHED_RACE = {
    "male": ("white_10_19", "white_20_29"),
    "female": ("female_10_19", "female_20_29"),
}
ALL_RACE = {
    "male": ("broad_10_19", "broad_20_29"),
    "female": ("female_broad_10_19", "female_broad_20_29"),
}
POOLED_NAME = {"male": "white_all", "female": "female_all"}

# The all-race comparison bank is AGE-MATCHED to the subject. `white_20_29` also
# carries a HISTORICAL pairing with `broad_10_19` -- the bank every shipped
# figure, guard and paragraph was written against before `broad_20_29` existed
# (2026-08-15 rebuild).
#
# Private development note omitted.
# age-matched bank leads, because comparing a 20-something against a 10-19 pool
# is the one place the age-matching rule was knowingly broken; the historical
# pairing stays beside it, labelled, so no published figure silently disappears
# and a reader can see how much the comparison POPULATION moves the answer --
# which on this subject is 24.1% -> 27.8% symmetric and 47.2% -> 63.3%
# whole-shape. ⚠️ Both banks have measured axis reproducibility, so showing both
# costs no refusal.
LEGACY_ALL_RACE = {"white_20_29": "broad_10_19"}


def cohort_family(primary: str, available: list[str]) -> dict:
    """The banks a report compares one subject against, from their own cohort.

    Returns primary / pooled / other_age / all_race names plus `order`, the
    sequence the report prints. Anything missing from `available` is dropped
    rather than substituted -- a comparison the cache cannot support should
    disappear, not quietly become a different comparison.
    """
    sex = "female" if primary.startswith("female") else "male"
    young, adult = MATCHED_RACE[sex]
    band_20 = primary.endswith("20_29")
    other_age = young if band_20 else adult
    all_race = ALL_RACE[sex][1 if band_20 else 0]
    # The historical pairing, kept ALONGSIDE the age-matched one rather than
    # instead of it. None for every cohort that never had one.
    all_race_legacy = LEGACY_ALL_RACE.get(primary)
    if all_race_legacy == all_race:
        all_race_legacy = None
    pooled = POOLED_NAME[sex]

    have = set(available)
    fam = {
        "sex": sex,
        "primary": primary,
        "other_age": other_age if other_age in have else None,
        "pooled": pooled if {young, adult} <= have else None,
        "pooled_from": [young, adult] if {young, adult} <= have else [],
        "all_race": all_race if all_race in have else None,
        "all_race_legacy": (
            all_race_legacy if (all_race_legacy and all_race_legacy in have) else None
        ),
    }
    order = [
        fam["other_age"],
        primary,
        fam["pooled"],
        fam["all_race"],
        fam["all_race_legacy"],
    ]
    seen: set[str] = set()
    fam["order"] = [
        n for n in order if n and not (n in seen or seen.add(n))  # keeps first
    ]
    return fam


MIN_VAR_SHARE = 0.05

# If an axis moves across gated re-shoots by much more than its own propagated
# error, then something the instrument does NOT model is still moving it — which
# is precisely the state shape typicality was in before this work. Shipping such
# an axis would repeat the defect one level down.
MAX_RESIDUAL_RATIO = 2.0


def axis_ships(
    var_share: float, mc_floor_sd: float, reshoot_swing_sd: float
) -> tuple[bool, float, str]:
    """Does one axis earn a place in the report? Returns (ships, band frac, why).

    Three independent tests, and an axis must pass all three:
      * it describes enough of the face to be worth a sentence;
      * its error bar clears the project's <25%-of-judged-scale rule, taking the
        WORSE of the propagated floor and the observed gated re-shoot movement;
      * nothing unexplained is moving it — the observed movement is consistent
        with the error the instrument can account for.
    """
    worst = max(float(mc_floor_sd), float(reshoot_swing_sd))
    frac = worst / SCALE_SD_RANGE
    if var_share < MIN_VAR_SHARE:
        return False, frac, f"explains only {var_share:.1%} of shape variance"
    if frac >= SHIP_MAX_BAND_FRAC:
        return False, frac, f"error bar is {frac:.0%} of the judged scale"
    ratio = reshoot_swing_sd / max(mc_floor_sd, 1e-9)
    if ratio > MAX_RESIDUAL_RATIO:
        return (
            False,
            frac,
            (
                f"moves {ratio:.1f}x more across re-shoots than its own error "
                "explains, so something unmodelled is still driving it"
            ),
        )
    return True, frac, "reproduces within its measured error"


# CONCEPTUALLY DISTINCT descriptors, one per idea. Redundant ones would break
# the naming rule for the wrong reason: an axis whose top two correlates are two
# spellings of the same anatomy is not ambiguous, but a rule comparing raw
# correlations cannot tell that from a genuine tie. So the tie-break is only ever
# run between descriptors that mean different things.
DESCRIPTORS = {
    "lower-face height": lambda L, s: abs(L[2, 1] - L[152, 1]) / s["height"],
    "face width against height": lambda L, s: s["width"] / s["height"],
    "eye spacing": lambda L, s: s["ipd"] / s["width"],
    "jaw width": lambda L, s: float(np.linalg.norm(L[172] - L[397])) / s["width"],
    "mouth width": lambda L, s: float(np.linalg.norm(L[61] - L[291])) / s["width"],
}

NAME_MIN_R = 0.70
NAME_MIN_MARGIN = 0.20  # the runner-up must be this much lower, in |r|


def _descriptor_values(config: np.ndarray) -> dict:
    import analyze_averageness2 as A2

    L = A2.standardize(config)
    s = {
        "width": float(np.ptp(L[:, 0])),
        "height": float(np.ptp(L[:, 1])),
        "ipd": float(np.linalg.norm(L[IPD_PAIR[0]] - L[IPD_PAIR[1]])),
    }
    return {k: float(f(L, s)) for k, f in DESCRIPTORS.items()}


def name_axes(configs, ctx: dict, n_axes: int = 6) -> list[dict]:
    """Give an axis a plain-English name only if the bank says it has earned one.

    ⛔ A NAME IS A CLAIM ABOUT ANATOMY AND GETS MEASURED LIKE ONE. "Face width" is
    a statement a reader will believe; a principal component is whatever direction
    happened to carry the most variance, and it owes nobody an interpretation.
    An axis is named only when it correlates at |r| >= 0.70 with one descriptor
    measured independently across the bank AND beats every other descriptor by a
    clear margin. Otherwise it ships unnamed.

    ⚠️ The hazard is live in this session's own history: "inner lip contour" was
    asserted from index memory and only survived because it was checked against
    FLAME's masks afterwards. It happened to be right.
    """
    vals = {k: [] for k in DESCRIPTORS}
    for c in configs:
        dv = _descriptor_values(c)
        for k in DESCRIPTORS:
            vals[k].append(dv[k])
    out = []
    for a in range(n_axes):
        scores = ctx["bank_scores"][:, a]
        rs = sorted(
            ((abs(float(np.corrcoef(scores, vals[k])[0, 1])), k) for k in DESCRIPTORS),
            reverse=True,
        )
        best_r, best_k = rs[0]
        second_r = rs[1][0] if len(rs) > 1 else 0.0
        named = best_r >= NAME_MIN_R and (best_r - second_r) >= NAME_MIN_MARGIN
        sign = float(np.corrcoef(scores, vals[best_k])[0, 1])
        out.append(
            {
                "axis": a + 1,
                "name": best_k if named else None,
                "r": round(best_r, 2),
                "runner_up": rs[1][1] if len(rs) > 1 else None,
                "runner_up_r": round(second_r, 2),
                # A named axis points the way its descriptor does, so "+0.9 SD"
                # means more of the named thing and not less. Without this the
                # deterministic SVD sign is arbitrary with respect to the WORDS.
                "flip_for_name": bool(named and sign < 0),
            }
        )
    return out


def typicality_interval(pct: float, swing_pp: float) -> tuple[float, float]:
    """The honest interval around a typicality percentile: the measured movement.

    ⛔ APPLIED IN EACH DIRECTION, NOT HALVED, and the two are not interchangeable.
    `swing_pp` is the RANGE over four gated photographs of one subject, and a
    range over four draws systematically understates the spread it estimates.
    Halving it let the primary cohort clear its nearest cutoff by 0.35 percentile
    points on 2026-08-12 — false precision of exactly the kind refused elsewhere
    here. Treating the observed range as a FLOOR applied each way is the
    conservative reading the evidence supports.

    ⛔ THIS IS NOT THE BOOTSTRAP CI, AND SUBSTITUTING ONE FOR THE OTHER IS THE
    ERROR THIS PROJECT HAS CAUGHT THREE TIMES. The bootstrap interval describes
    sampling error in the REFERENCE POOL — how well 72 faces estimate a
    population — and is NARROWER than how far this subject's own reading moves
    between photographs. Printing it beside the number would answer a question
    nobody asked with a tighter figure than the truth.
    """
    return (max(0.0, pct - swing_pp), min(100.0, pct + swing_pp))


def percentile_resolves(swing_pp: float, pct: float, cutoffs=(33.0, 67.0)) -> bool:
    """Does the typicality percentile survive its own measured movement?

    ⚠️ THE SWING IS APPLIED IN EACH DIRECTION, NOT HALVED, AND THAT IS THE WHOLE
    DIFFERENCE BETWEEN THIS RESOLVING AND REFUSING. `swing_pp` is the RANGE over
    four gated photographs of one subject, and a range over four draws
    systematically understates the spread it estimates. Halving it and centring
    on the reading would have let the primary cohort clear its nearest cutoff by
    0.35 percentile points — a margin far inside the uncertainty of the estimate
    producing it, which is false precision of exactly the kind this project
    refuses elsewhere. Treating the observed range as a FLOOR applied each way is
    the conservative reading the evidence supports.

    Kept as a function rather than a hardcoded False so the refusal stays a
    measurement: a subject sitting clear of every cutoff by more than this would
    resolve, and on the broad cohort one does.
    """
    lo, hi = typicality_interval(pct, swing_pp)
    return not any(lo < c <= hi for c in cutoffs)
