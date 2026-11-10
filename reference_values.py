"""The single authoritative catalogue of every reference value this pipeline
compares a face against.

WHY THIS FILE EXISTS
--------------------
Until now the pipeline's "ideals" were scattered across seven modules as bare
literals, and three of them (the profile angles) existed ONLY inside matplotlib
annotation strings, so they were shown to the reader but never written to any
JSON. A reader could not answer the two questions that decide whether a number
means anything: *where did this ideal come from* and *what kind of claim is it*.

This module answers both, per row, and refuses to answer anything else.

THE TIER SEPARATION IS THE POINT
--------------------------------
A flat table of "ideals" is dangerous precisely because it flattens. Printing
phi = 1.618 on the row above a measured population norm implies the two are the
same kind of claim. They are not. `evidence_tier` is therefore a first-class
column, never a footnote, and any renderer built on this file MUST show it
beside the value — not in a legend, not in a preamble, not in a tooltip.

SOURCING RULE (the hard one)
----------------------------
Every `source` string here is traceable to something already recorded in this
codebase — chiefly the per-feature citations in `analyze_advanced.py`, the
Credits section of `README.md`, and the construct definitions in
`analyze_averageness2.py`. Nothing was looked up, remembered, or inferred to
fill a gap. Where this project records no citation, the row says exactly that
via the `UNSOURCED` marker, and `is_sourced` is False. The unsourced count is
large. That is the finding, not a defect in the catalogue.

NO AGGREGATE — FOUNDING REFUSAL
-------------------------------
This module exposes no count of "ideals met", no score, no roll-up, and no
convenience helper that could become one. Accessors return records or subsets
of records. A face is not summarised here, in this file or anywhere downstream.

Importable and side-effect free: no file I/O, no printing, no network at import.
Run `python reference_values.py` for a self-inspection dump.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Iterator, Mapping, Optional, Sequence, Tuple, Union

# ---------------------------------------------------------------------------
# Evidence tiers — ordered weakest-claim-first, so a renderer that sorts by
# tier surfaces the artistic conventions where they belong: at the bottom.
# ---------------------------------------------------------------------------
TIER_ARTISTIC = "artistic convention (not validated)"
TIER_CLINICAL = "clinical or orthodontic norm (not a beauty claim)"
TIER_ANTHROPOMETRIC = "anthropometric population norm"
TIER_ATTRACTIVENESS = "attractiveness-literature supported (weak effect)"
TIER_AUTHOR = "author-chosen (this pipeline)"
# Two refinements the code justifies:
TIER_MEASUREMENT = "measurement constant (not an ideal)"
TIER_NONE = "no reference value applies (deliberate)"

TIER_ORDER: Tuple[str, ...] = (
    TIER_ATTRACTIVENESS,
    TIER_ANTHROPOMETRIC,
    TIER_CLINICAL,
    TIER_MEASUREMENT,
    TIER_AUTHOR,
    TIER_ARTISTIC,
    TIER_NONE,
)

ANCHOR_KINDS: Tuple[str, ...] = ("point", "range", "threshold_ladder", "none")

# The marker. Any `source` beginning with this prefix counts as unsourced, no
# matter what detail follows it. Renderers must show it inline, not in a
# footnote — an unsourced ideal that looks sourced is the failure mode this
# whole file exists to prevent.
UNSOURCED = "UNSOURCED — asserted by this pipeline, no citation recorded"

# A third, distinct state. A row that claims NO reference value needs no
# citation, and folding it into the sourced count would inflate that count with
# rows that cite nothing — the exact favourable drift this project's standing
# self-audit exists to catch. Kept separate so "cited" means cited.
NO_SOURCE_REQUIRED = "N/A — no reference value is claimed, so no source is required"

ReferenceScalar = Union[float, int, str]
ReferenceValue = Union[ReferenceScalar, Tuple[ReferenceScalar, ...], None]


@dataclass(frozen=True)
class Reference:
    """One reference value the pipeline compares a face against.

    `value` is the reference proper: a point, a (lo, hi) range, an ordered tuple
    of ladder cutoffs, or None when no reference exists. `bands` describes any
    classification cutoffs layered on top of it in words, because those cutoffs
    are almost always a DIFFERENT (usually author-chosen) kind of claim than the
    value they judge, and collapsing the two would re-create exactly the tier
    flattening this module refuses.
    """

    key: str
    metric: str
    value: ReferenceValue
    units: str
    anchor_kind: str
    tradition: str
    evidence_tier: str
    source: str
    what_deviation_means: str
    module: str
    code_location: str
    bands: Tuple[str, ...] = ()
    emitted_to_json: bool = True
    note: str = ""

    @property
    def provenance(self) -> str:
        """One of 'cited' | 'unsourced' | 'not applicable'."""
        if self.source.startswith(NO_SOURCE_REQUIRED):
            return "not applicable"
        if self.source.startswith(UNSOURCED):
            return "unsourced"
        return "cited"

    @property
    def is_sourced(self) -> bool:
        """True only when this project records an actual citation for the value."""
        return self.provenance == "cited"

    @property
    def has_reference(self) -> bool:
        return self.anchor_kind != "none"

    def value_text(self) -> str:
        """Human-readable rendering of `value`, honest about the None case."""
        if self.value is None:
            return "no reference value"
        if isinstance(self.value, tuple):
            if self.anchor_kind == "range" and len(self.value) == 2:
                return f"{self.value[0]}–{self.value[1]} {self.units}".strip()
            return f"{', '.join(str(v) for v in self.value)} {self.units}".strip()
        return f"{self.value} {self.units}".strip()


def _r(**kwargs) -> Reference:
    rec = Reference(**kwargs)
    if rec.anchor_kind not in ANCHOR_KINDS:
        raise ValueError(f"{rec.key}: bad anchor_kind {rec.anchor_kind!r}")
    if rec.evidence_tier not in TIER_ORDER:
        raise ValueError(f"{rec.key}: bad evidence_tier {rec.evidence_tier!r}")
    if rec.anchor_kind == "none" and rec.value is not None:
        raise ValueError(f"{rec.key}: anchor_kind 'none' must carry value None")
    return rec


# ---------------------------------------------------------------------------
# Shared prose, so the same caveat reads identically wherever it applies.
# ---------------------------------------------------------------------------
_CANON_MEANS_NOTHING = (
    'Nothing about attractiveness. README.md\'s "Limitations" states plainly '
    "that the classical canons and the golden ratio are NOT validated "
    "attractiveness predictors — they are historical artistic conventions. A "
    "face that departs from this number has not failed anything; the number "
    "simply describes a drawing convention, not a population or a preference."
)

_PHI_PROVENANCE = (
    "The golden-ratio-face claim largely traces to Marquardt's mask and has not "
    "held up. That is this project's own assessment, not a citation: NO literature "
    "reference for φ is recorded anywhere in this codebase. These are the weakest "
    "reference values here, and a deviation from any of them means nothing."
)

_ANNOTATION_ONLY_DEFECT = (
    "DEFECT — this reference exists only inside the matplotlib/OpenCV annotation "
    "text drawn onto profile_annotated.png (analyze_profile.py:365-370). It is "
    "never written to metrics_profile.json, so no downstream module, JSON "
    "consumer, or audit can see the value the reader was just shown. The reader "
    "is given a comparison the machine-readable output cannot reproduce."
)

_ADVANCED_ANCHOR_CAVEAT = (
    "These are POPULATION MIDPOINTS from the anthropometry / sexual-dimorphism "
    "literature, NOT validated decision thresholds. Male and female "
    "distributions overlap heavily on every one of these features, so landing "
    "on one side of a midpoint is a weak population tendency, not a "
    "classification. Several source measurements also come from calipers/3D/CT "
    "whose landmark definitions differ from this pipeline's 2D FaceMesh points "
    "(analyze_advanced.py:206-220)."
)


# ===========================================================================
# THE CATALOGUE
# ===========================================================================
CATALOGUE: Tuple[Reference, ...] = (
    # -----------------------------------------------------------------------
    # analyze_frontal.py — the neoclassical canons
    # -----------------------------------------------------------------------
    _r(
        key="thirds.equal_share",
        metric="Facial thirds — each third's share of trichion→menton height",
        value=33.333,
        units="% of face height",
        anchor_kind="point",
        tradition="neoclassical / classical artistic canon (equal facial thirds)",
        evidence_tier=TIER_ARTISTIC,
        source=UNSOURCED,
        what_deviation_means=_CANON_MEANS_NOTHING,
        module="analyze_frontal.py",
        code_location="analyze_frontal.py:49",
        bands=("green ≤5 pp deviation", "yellow ≤15 pp", "red >15 pp"),
        note=(
            "The colour bands are a separate, author-chosen claim — see "
            "grade_band.thirds. Note also that the lower third's share is "
            "re-used as a sexual-dimorphism input (dimorphism.lower_third_share) "
            "where it DOES carry a sourced anthropometric anchor; the same "
            "measurement therefore appears under two different tiers, correctly."
        ),
    ),
    _r(
        key="grade_band.thirds",
        metric="Colour grade cutoffs applied to facial-thirds deviation",
        value=(5.0, 15.0),
        units="percentage points (green max, yellow max)",
        anchor_kind="threshold_ladder",
        tradition="this pipeline's display grading",
        evidence_tier=TIER_AUTHOR,
        source=UNSOURCED,
        what_deviation_means=(
            "Only that the deviation crossed a cutoff someone in this project "
            "picked. The cutoffs judge an artistic convention, so a 'red' third "
            "is a red flag about nothing."
        ),
        module="analyze_frontal.py",
        code_location="analyze_frontal.py:56 (fa_common.grade, fa_common.py:221)",
    ),
    _r(
        key="fifths.equal_share",
        metric="Facial fifths — each vertical fifth's share of bizygomatic width",
        value=20.0,
        units="% of face width",
        anchor_kind="point",
        tradition="neoclassical / classical artistic canon (equal facial fifths)",
        evidence_tier=TIER_ARTISTIC,
        source=UNSOURCED,
        what_deviation_means=_CANON_MEANS_NOTHING,
        module="analyze_frontal.py",
        code_location="analyze_frontal.py:88",
        bands=("green ≤2 pp deviation", "yellow ≤4 pp", "red >4 pp"),
        note=(
            "The intercanthal fifth is re-used as the balance point of "
            "harmony.horizontal_balance, where the same 20% convention is "
            "restated as 'eye gap should equal one eye width'."
        ),
    ),
    _r(
        key="grade_band.fifths",
        metric="Colour grade cutoffs applied to facial-fifths deviation",
        value=(2.0, 4.0),
        units="percentage points (green max, yellow max)",
        anchor_kind="threshold_ladder",
        tradition="this pipeline's display grading",
        evidence_tier=TIER_AUTHOR,
        source=UNSOURCED,
        what_deviation_means=(
            "Only that the deviation crossed a project-chosen cutoff. These "
            "bands are 2.5× tighter than the thirds bands with no recorded "
            "reason, so 'red' is not comparable across the two canons."
        ),
        module="analyze_frontal.py",
        code_location="analyze_frontal.py:95 (fa_common.grade, fa_common.py:221)",
    ),
    _r(
        key="phi.face_height_over_width",
        metric="Face height / face width",
        value=1.618,
        units="ratio",
        anchor_kind="point",
        tradition="golden ratio (φ) applied to the face",
        evidence_tier=TIER_ARTISTIC,
        source=UNSOURCED,
        what_deviation_means=(
            _CANON_MEANS_NOTHING + " " + _PHI_PROVENANCE + " This is the weakest "
            "kind of row in the entire pipeline and should be read as a curiosity."
        ),
        module="analyze_frontal.py",
        code_location="analyze_frontal.py:132",
        bands=("green ≤5% relative deviation", "yellow ≤12%", "red >12%"),
    ),
    _r(
        key="phi.nose_width_over_eye_width",
        metric="Nose width / average eye width",
        value=1.0,
        units="ratio",
        anchor_kind="point",
        tradition=(
            "neoclassical canon (nasal width equals eye width) — carried in the "
            "φ block despite not being a φ value"
        ),
        evidence_tier=TIER_ARTISTIC,
        source=UNSOURCED,
        what_deviation_means=(
            _CANON_MEANS_NOTHING + " Note this row is 1.0, not φ: it sits under a "
            "heading called 'phi' but asserts a different (equal-widths) "
            "convention. The heading is misleading; the value is unsourced either "
            "way."
        ),
        module="analyze_frontal.py",
        code_location="analyze_frontal.py:133",
        bands=("green ≤5% relative deviation", "yellow ≤12%", "red >12%"),
        note=(
            "Re-used as one of the two centres of harmony.nose_mouth_eye, where "
            "the same 1.0 appears as a balance point."
        ),
    ),
    _r(
        key="phi.mouth_width_over_nose_width",
        metric="Mouth width / nose width",
        value=1.618,
        units="ratio",
        anchor_kind="point",
        tradition="golden ratio (φ) applied to the face",
        evidence_tier=TIER_ARTISTIC,
        source=UNSOURCED,
        what_deviation_means=_CANON_MEANS_NOTHING + " " + _PHI_PROVENANCE,
        module="analyze_frontal.py",
        code_location="analyze_frontal.py:134",
        bands=("green ≤5% relative deviation", "yellow ≤12%", "red >12%"),
        note=(
            "Re-used as the second centre of harmony.nose_mouth_eye "
            "(analyze_harmony.py:179), so this φ appears in two modules."
        ),
    ),
    _r(
        key="phi.lower_face_philtrum_over_lip_chin",
        metric="Lower face — philtrum (subnasale→labrale sup) / lip-chin (labrale inf→menton)",
        value=None,
        units="ratio",
        anchor_kind="none",
        tradition="none — deliberately withheld",
        evidence_tier=TIER_NONE,
        source=(
            NO_SOURCE_REQUIRED
            + ". The refusal itself is documented in analyze_frontal.py:136-155 "
            'and README.md\'s "What is actually unusual here".'
        ),
        what_deviation_means=(
            "There is nothing to deviate FROM. These segments are not the "
            "classical lower-third division (which runs subnasale→stomion→menton "
            "and follows a neoclassical 1:2 convention, not φ), so no documented "
            "ideal applies to the quantity as computed. This row is reported "
            "descriptively with dev_pct=None and grade='n/a'. Earlier versions "
            "graded it against 1.618, which guaranteed a large 'deviation' on "
            "every possible human face because the philtrum is always far shorter "
            "than the lip-to-chin span. THIS ROW IS A FIRST-CLASS STATE, NOT A GAP "
            "TO FILL — do not supply an ideal here."
        ),
        module="analyze_frontal.py",
        code_location="analyze_frontal.py:145-155",
        note=(
            "The same measurement is still given a descriptive centre of 0.5 in "
            "analyze_harmony.py (harmony.lower_face_chin). That is a knowing "
            "inconsistency documented at analyze_harmony.py:228-243: no canon in "
            "Part A, a borrowed 1:2 convention used only as a two-directional "
            "descriptive centre in the harmony module."
        ),
    ),
    _r(
        key="grade_band.phi",
        metric="Colour grade cutoffs applied to φ-ratio relative deviation",
        value=(5.0, 12.0),
        units="% relative deviation (green max, yellow max)",
        anchor_kind="threshold_ladder",
        tradition="this pipeline's display grading",
        evidence_tier=TIER_AUTHOR,
        source=UNSOURCED,
        what_deviation_means=(
            "Only that the deviation crossed a project-chosen cutoff around an "
            "unsourced artistic convention. Two layers of unsourced judgement."
        ),
        module="analyze_frontal.py",
        code_location="analyze_frontal.py:126 (fa_common.grade, fa_common.py:221)",
    ),
    _r(
        key="grade_band.frontal_symmetry_pairs",
        metric="Colour grade cutoffs applied to per-pair frontal asymmetry",
        value=(3.0, 7.0),
        units="% asymmetry (green max, yellow max)",
        anchor_kind="threshold_ladder",
        tradition="this pipeline's display grading",
        evidence_tier=TIER_AUTHOR,
        source=UNSOURCED,
        what_deviation_means=(
            "Only that a pair crossed a project-chosen cutoff. This is the "
            "first-pass symmetry measured off a fixed vertical midline; "
            "build_report.py:398-399 records that this panel is superseded by the "
            "roll-corrected Part C symmetry."
        ),
        module="analyze_frontal.py",
        code_location="analyze_frontal.py:186",
    ),
    _r(
        key="pose_gate.symmetry_yaw_deg",
        metric="Head yaw beyond which a bilateral-symmetry reading is pose, not face",
        value=(2.0, 6.0),
        units="degrees |yaw| (measured max, pose-inflated max)",
        anchor_kind="threshold_ladder",
        tradition="measured in this project, 2026-08-01 pose burst",
        evidence_tier=TIER_MEASUREMENT,
        source=(
            "MEASURED HERE — captures/2026-08-01_pose-burst, 10 frames of one "
            "subject over 26 seconds. Because the sitting is 26 seconds long the "
            "face is constant, so every difference between frames is measurement "
            "error by construction. n=5 inside |yaw|<=20 deg, n=9 usable overall. "
            "One subject: directional, not a population constant."
        ),
        what_deviation_means=(
            "That the symmetry number is reporting head position rather than the "
            "face. Measured: across 5 near-frontal frames of a CONSTANT face, "
            "analyze_advanced's mean_asym_pct_ipd spanned 1.18%–27.29% of IPD, a "
            "23x range — so at least 96% of the largest reading was capture "
            "condition. It rises monotonically with |yaw| (r=0.995 inside the "
            "gate; R^2=0.993 against the (1-cos yaw) foreshortening form over "
            "0.5–69.5 deg). At |yaw| 4 deg the reading was already 6.1x the "
            "lowest-yaw frame — six times outside the ±10% error bar the pipeline "
            "publishes for it. NOTE the gate does not correct anything: fitting "
            "the pose term two defensible ways (linear in yaw vs 1-cos yaw) put "
            "the pose-free asymmetry at 0.78 ± 1.09 and 4.50 ± 2.56 %IPD from the "
            "same five points. A 6x model-dependent disagreement means the "
            "pose-free value is NOT identifiable from this data, so subtracting a "
            "fitted correction would manufacture precision. Gate and disclose."
        ),
        module="analyze_advanced.py",
        code_location="analyze_advanced.py (symmetry pose_condition)",
        bands=(
            "|yaw| <= 2 deg — 'measured': pose contribution is comparable to the "
            "detector floor.",
            "2 < |yaw| <= 6 deg — 'pose_inflated': the reading is usable only as an "
            "upper bound; the pose term is already the same order as the face term.",
            "|yaw| > 6 deg — 'refused': the number is dominated by head position "
            "and is not a statement about the face.",
        ),
        note=(
            "The cutoffs are drawn from the measured inflation ladder rather than "
            "from a model, because the model choice is exactly what is unresolved. "
            "The subject's own June baseline sits at |yaw| 1.5 deg and so still "
            "grades 'measured' — this gate does not retract the shipped figure, it "
            "states the condition under which that figure means anything."
        ),
    ),
    _r(
        key="repeatability.averageness_construct_swing_pp",
        metric="How far each averageness construct moves when only the measurement conditions change",
        value=(32.4, 3.4, 66.7, 11.1),
        units="percentile points (full-shape bank sweep, symmetric bank sweep, full-shape subject pose, symmetric subject pose)",
        anchor_kind="threshold_ladder",
        tradition="measured in this project, 2026-08-04",
        evidence_tier=TIER_MEASUREMENT,
        source=(
            "MEASURED HERE, two independent sweeps. (1) REFERENCE BANK, gate_trade.py: "
            "tightening the reference pose gate 14->8->5->2 deg moved the full-shape "
            "percentile 68.1->61.2->38.5->35.7 while the symmetric component held "
            "39.1->38.8->38.5->35.7. Two gated subsets fell outside the range that "
            "equal-sized RANDOM subsets produce, so it is not small-n noise. (2) SUBJECT "
            "POSE, the 26-second burst where the face is constant: inside |yaw|<=14 deg "
            "(n=3) the full-shape percentile spanned 66.7 points (13.9%-80.6%) where the "
            "symmetric component spanned 11.1 (38.9%-50.0%). Both collapse to 0% past "
            "~60 deg, where landmark detection itself fails."
        ),
        what_deviation_means=(
            "That the report's HEADLINE averageness figure is the fragile one and the "
            "figure it prints as a secondary disclosure is the robust one, on BOTH axes "
            "and by roughly 10x and 6x respectively. The shipped bootstrap CI (~+-10 "
            "points) describes sampling error in the reference pool ONLY; it does not "
            "cover either sensitivity measured here, so it understates the real "
            "uncertainty on the full-shape number. NOTE the symmetric component is not "
            "pose-PROOF: 11.1 points in-gate is small, not zero, a residual cos-theta "
            "horizontal compression survives mirror-averaging, and the bank sweep gated "
            "with a noisy yaw proxy. n=3 in-gate frames is thin."
        ),
        module="gate_trade.py",
        code_location="gate_trade.py; reference_pose_audit.py; consumed by build_report.py",
        bands=(
            "full-shape averageness — includes asymmetry, matches the literature "
            "construct, and absorbs BOTH the reference bank's pose contamination and "
            "the subject's own head turn through the same channel.",
            "symmetric component — mirror-averaging cancels the antisymmetric part of a "
            "yaw distortion, which is why it is robust; the cost is that asymmetry no "
            "longer contributes, and Part C must carry it instead (it already does).",
        ),
        note=(
            "The consistency check that makes this convincing rather than convenient: at "
            "the 5 and 2 deg reference gates the two constructs return the IDENTICAL "
            "figure. Clean bank -> they agree; contaminated bank -> only the full-shape "
            "one inflates. That is the signature of contamination in one metric and not "
            "the other, and it is not what a merely-insensitive metric would look like."
        ),
    ),
    _r(
        key="repeatability.averageness_session_spread_pp",
        metric="How far the PRIMARY averageness figure moves when the same face is photographed on another day, inside the pose gate",
        value=(23.6, 30.2, 20.6, 11.1),
        units="percentile points (best reference set, worst reference set, worst WITHIN one sitting, worst from re-framing alone at constant distance)",
        anchor_kind="threshold_ladder",
        tradition="measured in this project, 2026-08-12 (revised same day on a third sitting)",
        evidence_tier=TIER_MEASUREMENT,
        source=(
            "MEASURED HERE, averageness_session_probe.py. Five frontal photographs of one "
            "unchanging face, EVERY ONE inside the shipped pose gate (|yaw| 0.4-4.0 deg, "
            "against a refuse threshold of 5), across three sittings 2026-06-16, 08-01 and "
            "08-12. Each is scored through the SHIPPED path -- the same plain detector "
            "load_subject uses and the same analyze_set the report calls -- so the only "
            "thing that differs between readings is which photograph was taken. The "
            "symmetric component, the PRIMARY figure since 2026-08-04, spans 18.0 points "
            "on white_20_29, 21.5 on white_all, 27.0 on white_10_19 and 27.8 on "
            "broad_10_19. Decomposed: up to 13.9 points arise WITHIN a single sitting "
            "(frames minutes apart, same light and distance) and up to 27.8 across "
            "sittings. The full-shape secondary is worse throughout, 22.2 to 33.3 points."
        ),
        what_deviation_means=(
            "⭐⭐ REVISED 2026-08-12 ON A THIRD SITTING, AND THE DIAGNOSIS CHANGED: this is "
            "NOT primarily a between-session problem. The third sitting contributed three "
            "near-frontals taken minutes apart in one room, and they span 16.7-20.6 points "
            "depending on the bank -- most of the 23.6-30.2 total across all comparable-"
            "resolution photographs. Two of them, IMG_2061 and IMG_2062, are at effectively "
            "IDENTICAL distance (330 and 329 px IPD) with the phone put down and re-framed "
            "between shots, and they differ by up to 11.1 points. So the figure does not need "
            "a different day, a different light or a different distance to move: putting the "
            "phone down and picking it up again is enough. The earlier reading of this claim "
            "-- 13.9 within a sitting against 27.8 across sittings -- came from ONE sitting "
            "with more than one frontal, and a second such sitting shows the within term is "
            "larger and carries most of the total. "
            "That the interval the report prints beside this figure does not cover the "
            "variation a returning user will actually see. That interval is a bootstrap "
            "over THE SUBJECT'S OWN COHORT POOL, so it describes sampling error of the "
            "bank and nothing else; "
            "in ALL FOUR reference sets at least one in-gate photograph of the same face "
            "falls OUTSIDE it. This is the same defect class already recorded for the "
            "full-shape construct in repeatability.averageness_construct_swing_pp, now "
            "measured for the promoted primary and across sessions rather than within one "
            "burst -- 18.0-27.8 points against the 11.1 that a single sitting could see, "
            "so the burst understated it by roughly 1.6-2.5x. CONSEQUENCE FOR THE REPORT: "
            "the placement WORDS refuse when this spread straddles a placement cutoff, "
            "which on broad_10_19 it does (24.1 reads 'bottom third', 51.9 reads 'middle "
            "of the pack'), and shape-typicality is no longer badged ROBUST."
        ),
        module="averageness_session_probe.py",
        code_location="averageness_session_probe.py; consumed by build_report.py",
        bands=(
            "within one sitting (<=13.9 pp) — detector and expression drift on frames "
            "minutes apart; this is the floor, and it is not small.",
            "across sittings (<=27.8 pp) — adds light, distance, camera framing and day. "
            "This is what a user who re-photographs themselves experiences.",
        ),
        note=(
            "⛔ This measurement establishes that the reading MOVES and roughly how far. It "
            "does NOT calibrate a distribution and a shipped error bar must not be fitted "
            "to it: n=1 subject, five photographs, three sittings, and the two constructs "
            "disagree about whether within- or between-session dominates depending on the "
            "reference set. The fix this licenses is a refusal and a disclosure, not a "
            "widened bar. ⛔ It also cannot be fixed by averaging several of the subject's "
            "photographs together: a subject cleaned by averaging, scored against a bank of "
            "raw single photographs, is the MULTIVIEW-WIRING 5.1 hazard measured on "
            "2026-08-11 -- distance to a mean is a MAGNITUDE, so a cleaner subject reads "
            "more average whether or not the face changed. The bank cannot be levelled up "
            "(2D-only caches, FairFace source images deleted), so the honest options are "
            "pose-controlled multi-photo references or refusal."
            " ⛔ THIS FIGURE IS THE COMPARABLE-RESOLUTION ONE AND MUST STAY THAT WAY. On "
            "2026-08-12 two further in-gate frontals were found in calibration_subject own camera roll "
            "(2026-05-25 and 06-18) and pooling them moves the worst spread from 27.8 to 48.1 "
            "points. They are NOT an extension of this measurement: they are the only two "
            "photographs in the pool below 200px IPD (180 and 186, against 298-675 for the "
            "rest) and they returned the two HIGHEST readings of the seven -- every "
            "low-resolution photograph above every high-resolution one. Each is also its own "
            "sitting, so resolution and session are perfectly confounded and the pooled figure "
            "cannot be read as cross-session variation. ⭐ NOTE THE DIRECTION, because it is a "
            "different mechanism from the 5.1 hazard rather than more of it: lower resolution "
            "reads MORE average, and independent landmark noise inflates a distance, which "
            "would read LESS average. A detector falling back toward its own prior when there "
            "is less detail to fit would produce this. n=2 photographs, so it is a hypothesis "
            "with its own test to run, not a result -- and the 200px line was drawn AFTER "
            "seeing which photographs it separates, so it cannot carry an argument either."
        ),
    ),
    _r(
        key="repeatability.jawline_angle_deg",
        metric="Range the soft-tissue jawline angle covers on ONE unchanging jaw",
        value=(98.9, 136.2),
        units="degrees (min, max across a single sitting)",
        anchor_kind="range",
        tradition="measured in this project, 2026-08-04",
        evidence_tier=TIER_MEASUREMENT,
        source=(
            "MEASURED HERE — jawline_angle.py over the four usable frames of "
            "captures/2026-08-01_profile-burst, one sitting, so the jaw is "
            "constant and every degree of spread is measurement error by "
            "construction. yaw 51->127.1, 57->113.9, 63->136.2, 64->98.9. Mean "
            "119.0, spread 37.3 deg, CV 13.6%, r = -0.294 with turn angle."
        ),
        what_deviation_means=(
            "Nothing about the face. This measurement is REFUSED. The scale a "
            "jawline angle is read against spans roughly 100-140 deg (the range "
            "commercial reports use to separate a 'strongly masculine' jaw from a "
            "'strongly feminine' one) and the 37.3 deg spread on ONE jaw covers "
            "almost all of it. Critically it does NOT track turn angle (r = -0.29), "
            "so unlike facial convexity (r = -0.976, a correctable systematic) it "
            "cannot be corrected — it is noise, the same category the cervicomental "
            "angle was refused for (r = -0.31). On these same four frames the "
            "nasolabial angle moved 0.3% and facial convexity 0.8%, so this is not "
            "the silhouette being unusable; it is this construct being unrecoverable "
            "from it."
        ),
        module="jawline_angle.py",
        code_location="jawline_angle.py (jawline_angle, the measured refusal)",
        note=(
            "A first pass without a jaw window read the SHOULDER and produced an "
            "80 deg spread; windowing to the mandible improved it to 37.3 and it "
            "still fails. Recorded because the intermediate number looked plausible "
            "and would have shipped had it not been checked against a face known "
            "not to have changed. NOTE this is the soft-tissue contour, never the "
            "skeletal gonial angle, which needs a lateral cephalogram."
        ),
    ),
    _r(
        key="ruler.mesh_ipd_over_iris_ipd",
        metric="How well an IPD built from the 468-point mesh reproduces the detected iris IPD",
        value=0.9863,
        units="ratio (mesh-derived / detected), inside the pose gate",
        anchor_kind="point",
        tradition="measured in this project, 2026-08-04 (ruler)",
        evidence_tier=TIER_MEASUREMENT,
        source=(
            "MEASURED HERE — eye-aperture ring centroids vs pts[468]/pts[473] on 10 "
            "real frames (the 9 usable frames of captures/2026-08-01_pose-burst plus "
            "the report's own frontal). In-gate (|yaw|<=20, n=6): mean 0.9863, SD "
            "0.0080, CV 0.81%. Ungated it degrades to mean 0.9954, SD 0.0355, CV "
            "3.56%, drifting to 1.08 at -63 deg as the far eye foreshortens. The two "
            "rejected candidates: canthal midpoints 0.9524 (in-gate CV 0.71%) and "
            "OUTER CANTHI 1.4023 (in-gate CV 0.58%). All three are equally stable; "
            "only the ring centroids are already at the right level."
        ),
        what_deviation_means=(
            "This is a units conversion, not a fact about a face. It exists because a "
            "fitted multi-view identity is a 468-point mesh with no iris landmarks, "
            "while every asymmetry figure this project publishes is a percentage OF "
            "THE IPD taken from pts[468]-pts[473]. Using the outer canthi instead — "
            "which is what the code did until 2026-08-04 (ruler) — divided the fitted side of "
            "the headline comparison by a length 1.40x larger than the single-photo "
            "side used, making fitted readings ~29% smaller for free. Correcting it "
            "moved the multi-view reproducibility advantage from 2.61x to 1.9x, and a later occlusion-weighting fix took it to 4.7x. The "
            "residual 1.37% shortfall biases the resulting percentage slightly HIGH, "
            "which is the conservative direction for a figure published about a person."
        ),
        module="fa_common.py",
        code_location="fa_common.py (iris_centres_from_mesh, mesh_ipd, with_mesh_iris)",
        note=(
            "Scope is IN-GATE. The proxy is validated where the pose gates already "
            "keep every asymmetry figure; outside them it has not been shown to hold "
            "and the figures it would normalise are refused anyway."
        ),
    ),
    _r(
        key="scale.mm_reproducibility_pct",
        metric="How much an unchanging physical length moves, in mm, between photographs of the same face",
        value=(4.6, 6.1),
        units="% (within-session in-gate max, cross-session pose-matched max)",
        anchor_kind="threshold_ladder",
        tradition="measured in this project, 2026-08-04 scale probe",
        evidence_tier=TIER_MEASUREMENT,
        source=(
            "MEASURED HERE — scale_probe.py over captures/2026-08-01_pose-burst "
            "(one 26-second sitting, so the face is constant and every mm of "
            "spread is scale error by construction) and the pose-matched "
            "cross-session pair 47 days apart. Six physical lengths tracked. "
            "Within-session |yaw|<=14 deg (n=3): 2.4%-4.6% range. |yaw|<=21 deg "
            "(n=6): 4.0%-8.6%. Cross-session at |delta pose| < 1.5 deg on every "
            "axis: mean 3.5%, max 6.1%, and ALL SIX lengths moved the same "
            "direction — a common-mode scale offset, not scatter. One subject."
        ),
        what_deviation_means=(
            "That a millimetre printed by this pipeline is reproducible to a few "
            "percent, not to a millimetre. It is a FLOOR: it excludes the 11.7 mm "
            "iris assumption itself (see scale.iris_diameter_mm — +-4.3% from the "
            "population SD, systematic per subject and therefore invisible to any "
            "repeatability test, since re-photographing the same person cannot "
            "reveal that their iris is not average). Combined with that term a "
            "realistic absolute-mm uncertainty is ~+-6%, and up to ~+-8% at the "
            "worst observed session offset. Outside the pose gate the same "
            "lengths ran 58%-109% — the gate is not optional for mm."
        ),
        module="scale_probe.py",
        code_location="scale_probe.py; consumed by analyze_anatomy.py (philtrum bracket)",
        bands=(
            "ratios and angles — UNAFFECTED. A common-mode scale error cancels in "
            "any dimensionless quantity, which is where 'more detailed' should be "
            "built.",
            "absolute mm — carries the full budget. Any band it is classified "
            "against must be wider than ~2x this, or the verdict is not earned.",
        ),
        note=(
            "This is the geometric analogue of the phototype finding: the "
            "self-referential quantity survives and the absolute one does not. "
            "It was measured before porting metrics onto the mesh precisely so "
            "that P3 would not print seven numbers carrying an unmeasured term."
        ),
    ),
    _r(
        key="scale.anchor_disagreement_pct",
        metric="Disagreement between the pipeline's two independent metric anchors",
        value=2.0,
        units="% (iris anchor relative to bizygomatic anchor, in-gate mean)",
        anchor_kind="point",
        tradition="measured in this project, 2026-08-04 scale probe",
        evidence_tier=TIER_MEASUREMENT,
        source=(
            "MEASURED HERE — scale_probe.py. This pipeline carries TWO metric "
            "anchors and had never compared them: fa_refine.iris_scale uses an "
            "assumed 11.7 mm iris, while analyze_anatomy.py used "
            "face_width_px/137.0, an assumed population-mean bizygomatic width. "
            "In-gate (n=6) the iris anchor reads +2.0% against the bizygomatic "
            "one, range -0.4% to +4.5%; on the June baseline +2.2%, on the "
            "pose-matched August frame -3.7%."
        ),
        what_deviation_means=(
            "The two anchors share NO assumption, so their agreement is a real "
            "cross-check rather than two procedures agreeing because they were "
            "told the same thing — the failure mode that retired the 3D "
            "reconstruction, whose depth axis WAS the profile silhouette it was "
            "being checked against. Neither is badly wrong; neither is good "
            "to better than a few percent. The disagreement swings ~6 percentage "
            "points between sessions, so it is model uncertainty and belongs "
            "INSIDE the bracket rather than beside it."
        ),
        module="scale_probe.py",
        code_location="scale_probe.py; consumed by analyze_anatomy.py",
        note=(
            "The bizygomatic anchor makes the stronger claim about the subject — "
            "that their face is of average width — where the iris is a rigid "
            "anatomical constant. The philtrum bracket therefore spans BOTH "
            "anchors rather than picking one."
        ),
    ),
    _r(
        key="pose_gate.pitch_notice_deg",
        metric="Head pitch beyond which the capture check warns that the vertical thirds are tilted",
        value=12.0,
        units="degrees |pitch|",
        anchor_kind="point",
        tradition="this pipeline's capture gating",
        evidence_tier=TIER_AUTHOR,
        source=UNSOURCED,
        what_deviation_means=(
            "Only that the chin was far enough off level for the capture check to "
            "say so. Unlike the yaw gate beside it, this cutoff was NOT measured "
            "against a pose burst — no equivalent pitch series exists in this "
            "project — so it states a project preference, not a measured tolerance."
        ),
        module="capture_check.py",
        code_location="capture_check.py (pitch check), analyze_extended.py (styling capture_note)",
        note=(
            "Lifted here from a bare literal so the capture check and the Part F "
            "styling read cannot drift apart. They did: the styling read used to "
            "tell the subject a pitch of -8.8 deg 'foreshortens the lower third, "
            "shoot at or slightly above eye level' while the capture check, on the "
            "same frame, passed it without comment. Both now consult this row."
        ),
    ),
    _r(
        key="grade_band.dominant_side_pct_ipd",
        metric="Mean bilateral asymmetry above which the report stops calling a face's two sides interchangeable",
        value=5.0,
        units="% of IPD (mean asymmetry)",
        anchor_kind="point",
        tradition="this pipeline's display wording",
        evidence_tier=TIER_AUTHOR,
        source=UNSOURCED,
        what_deviation_means=(
            "Only which adjective the prose uses ('mild' vs 'moderate') and whether "
            "the styling read says the two three-quarter angles look alike. It is a "
            "wording cutoff, not a finding, and it is applied to a figure the pose "
            "gate above may already have withheld — see pose_gate.symmetry_yaw_deg, "
            "which measured this same quantity spanning a 23x range on one "
            "unchanging face."
        ),
        module="build_report.py",
        code_location="build_report.py (Part C summary), analyze_extended.py (styling asymmetry_note)",
        note=(
            "Lifted here from two separate bare literals that had to agree and were "
            "never checked against each other."
        ),
    ),
    _r(
        key="pose_gate.averageness_yaw_deg",
        metric="Head yaw beyond which a shape-averageness percentile is pose, not face",
        value=(2.0, 5.0),
        units="degrees |yaw| (measured max, pose-deflated max)",
        anchor_kind="threshold_ladder",
        tradition="measured in this project, 2026-08-01 pose burst",
        evidence_tier=TIER_MEASUREMENT,
        source=(
            "MEASURED HERE — captures/2026-08-01_pose-burst, same 26-second sitting "
            "as pose_gate.symmetry_yaw_deg, so the face is constant and every "
            "difference is error. n=5 inside |yaw|<=20 deg. Extracted with the PLAIN "
            "detector, matching how the FairFace reference was extracted. The probe "
            "was validated by reproducing all four shipped percentiles exactly from "
            "the project's own frontal.png before any burst frame was measured. "
            "One subject: directional, not a population constant."
        ),
        what_deviation_means=(
            "That the percentile is reporting head position rather than the face. "
            "Measured on the primary matched cohort across 5 near-frontal frames of "
            "a CONSTANT face: 4.2% to 84.7%, i.e. 80.5 points of a 100-point scale, "
            "CV 107%. This is a LARGER collapse than bilateral symmetry suffers on "
            "the same frames. The fall with yaw is steep and monotone — 84.7% at "
            "0.4 deg, 62.5% at 4.5 deg, 13.9% at 12.9 deg — against a published "
            "bootstrap CI of only about +/-10 points. CRITICALLY, the project's own "
            "is_frontal() gate (|yaw| <= 14 deg, analyze_averageness2.py:151) does "
            "NOT protect this: the 12.9 deg frame PASSES that gate and still reads "
            "71 points low. is_frontal is a reference-population filter and is far "
            "too loose to qualify a subject reading — which is the evidence for "
            "per-metric tolerances rather than one global pose gate."
        ),
        module="analyze_averageness2.py",
        code_location="analyze_averageness2.py (pose_condition on each set)",
        bands=(
            "|yaw| <= 2 deg — 'measured': the pose shift is small next to the "
            "bootstrap CI already reported.",
            "2 < |yaw| <= 5 deg — 'pose_deflated': usable only as a LOWER bound; yaw "
            "moves this percentile down, so the true value is higher than shown.",
            "|yaw| > 5 deg — 'refused': dominated by head position.",
        ),
        note=(
            "Direction matters and is the opposite of symmetry's. Yaw INFLATES "
            "measured asymmetry and DEFLATES measured averageness, so both robust "
            "axes read worse as capture degrades. The subject's own baseline sits at "
            "|yaw| 1.5 deg and grades 'measured' on both — the shipped numbers are "
            "therefore taken at the favourable end of what a casual re-shoot would "
            "produce. That is not cherry-picking (a frontal photo is also the least "
            "foreshortened, i.e. the correct instrument) but it belongs in the "
            "favourable-drift disclosure the report already carries."
        ),
    ),
    _r(
        key="capture_gate.skin_calibration_ipd_px",
        metric="Face resolution the skin thresholds were calibrated at",
        value=676.0,
        units="px interpupillary distance",
        anchor_kind="point",
        tradition="measured in this project, 2026-08-01",
        evidence_tier=TIER_MEASUREMENT,
        source=(
            "MEASURED HERE — the same 26-second burst as the pose gates, plus the "
            "2026-08-01 portrait frame, run through the REAL analyze_frontal + "
            "analyze_skin stages rather than a reimplementation. n=5 burst frames at "
            "IPD 283-336 px, one portrait at 377 px, one baseline at 676 px. One "
            "subject, one skin condition, one session per resolution point."
        ),
        what_deviation_means=(
            "That the skin numbers are not comparable to the ones this pipeline "
            "publishes. analyze_skin's thresholds are pixel-bound — the texture "
            "kernel is a fixed 6.0 px sigma, the mask dilations are 26/20/14 px, and "
            "the lesion opening and contrast ring are fixed too — so face size "
            "changes the answer. MEASURED: the inflammatory-lesion estimate reads 20 "
            "at IPD 676 and 6 at both IPD ~300 and IPD 377, a 3.3x under-count on "
            "identical skin. The transition between those points is unlocalised, so "
            "there is no safe interpolation: treat any capture below the calibration "
            "resolution as producing a DIFFERENT measurement, not a noisier one. "
            "T-zone shine behaves the same way (14.5% at 676 px vs 0.9-2.2% below)."
        ),
        module="analyze_skin.py",
        code_location="analyze_skin.py (capture_condition), thresholds throughout",
        note=(
            "This is a defect being disclosed, not a tolerance being recommended. "
            "⚠️ THE OBVIOUS FIX WAS TRIED AND ONLY HALF WORKED — recorded here so it "
            "is not attempted again on the assumption it will. Every structuring "
            "element in the layer was converted from fixed pixels to millimetres of "
            "face on 2026-08-01 (mask dilations, boundary erosion, texture kernel, "
            "lesion contrast ring, perioral band; the lesion AREA bounds were already "
            "mm-bound). It is a no-op at the calibration resolution, and it DID fix "
            "the texture label, which previously flipped smooth<->moderate across "
            "resolutions and is now invariant and in agreement with the baseline. It "
            "did NOT close the lesion-count gap at all: still ~20 versus ~5. The "
            "reason is that the gap was never a threshold-calibration problem. At "
            "this lower resolution a 2 mm lesion subtends roughly 4-5 px, which is at "
            "the detection floor — there is no signal for a better-scaled operator to "
            "find. That is an INFORMATION limit, the same class as self-occlusion, "
            "and no amount of rescaling recovers detail the sensor did not capture. "
            "The only honest handling is the resolution gate this constant defines."
        ),
    ),
    _r(
        key="repeatability.skin_erythema_spread_factors",
        metric="Measured multiplicative spread of the erythema-area axis on ONE unchanging face",
        value=(0.386, 2.593),
        units="multiplier on a single reading, (low, high)",
        anchor_kind="threshold_ladder",
        tradition="measured in this project, 2026-08-01",
        evidence_tier=TIER_MEASUREMENT,
        source=(
            "MEASURED HERE — the same seven captures behind "
            "'repeatability.skin_iga_grade'. The erythema-area axis ranged 2.1%-14.1% "
            "on skin that could not have changed. Expressed as multipliers about the "
            "geometric centre of that range (sqrt(2.1*14.1) = 5.44): 2.1/5.44 = 0.386 "
            "and 14.1/5.44 = 2.593."
        ),
        what_deviation_means=(
            "These are what turn the measured instability into a REPORTABLE BRACKET "
            "rather than a caveat sitting next to a point estimate. Applying them to a "
            "reading gives the grade range that reading actually supports. ⚠️ They come "
            "from ONE subject's illumination spread and are applied to others as an "
            "assumption, not a measurement of them — but the alternative is presenting "
            "a single grade whose own measured spread crosses this module's escalation "
            "threshold. The geometric centre is used because the axis is a positive "
            "ratio-like quantity, where a symmetric arithmetic band would put the low "
            "end below zero."
        ),
        module="analyze_skin.py",
        code_location="analyze_skin.py (iga_bracket)",
    ),
    _r(
        key="repeatability.skin_iga_grade",
        metric="Measured repeatability of the automated acne-severity grade",
        value=None,
        units="",
        anchor_kind="none",
        tradition="measured in this project, 2026-08-01",
        evidence_tier=TIER_MEASUREMENT,
        source=(
            "MEASURED HERE — 5 near-frontal frames from one 26-second sitting, so the "
            "skin is necessarily unchanged, plus a same-day portrait and the baseline."
        ),
        what_deviation_means=(
            "There is no reference value and deliberately none is offered: this "
            "records that the grade is NOT REPEATABLE. On one unchanging face the "
            "automated grade returned two ADJACENT bands lying either side of this "
            "module's own documented escalation threshold, so three of seven captures "
            "recommended in-person dermatology review and four did not. (The grades "
            "themselves are withheld: they describe a real person's skin and are not "
            "needed to make the methodological point.) Within the "
            "shared-resolution burst the flip is driven entirely by the erythema-area "
            "axis, which ranged 2.1%-14.1% (CV 59%) while the lesion count held at "
            "6-7 — and the spread is NOT monotone in head yaw, so it is illumination "
            "and exposure, not pose. A pose gate is therefore the wrong instrument "
            "here. Consequence for the design: the grade may never be used to "
            "REASSURE, and the escalation advice must not be withheld on a low "
            "reading, because under-grading is the direction that causes harm."
        ),
        module="analyze_skin.py",
        code_location="analyze_skin.py:189 (iga_grade), :483-488 (escalation clause)",
    ),
    _r(
        key="repeatability.extended_frontal_sd",
        metric="Measured within-session SD of each frontal extended-morphometry ratio",
        value=(0.0539, 0.0394, 0.0135, 0.0303, 0.0230),
        units="ratio units, in row order: eye-spacing, fWHR, jaw:cheek, midface, lip",
        anchor_kind="threshold_ladder",
        tradition="measured in this project, 2026-08-01",
        evidence_tier=TIER_MEASUREMENT,
        source=(
            "MEASURED HERE — 5 near-frontal frames of one constant face from the "
            "26-second burst. CVs are 1.7%-4.9%, i.e. these ratios are far steadier "
            "than symmetry or averageness. The problem is not the numbers."
        ),
        what_deviation_means=(
            "Used to decide whether a verdict is RESOLVABLE, not to grade anything. "
            "The published bands are narrow relative to this measurement noise, so "
            "the three-state verdict the report shows ('in range' / 'borderline' / "
            "'outside') flipped on ALL FIVE metrics across frames of an unchanging "
            "face. Ratio of one SD to the full width of the in-range band: eye "
            "spacing 0.90, midface 0.20, lip 0.14, fWHR 0.13, jaw:cheek 0.13. "
            "EYE SPACING IS THE SEVERE ONE — one standard deviation covers 90% of "
            "its entire 0.97-1.03 band, so that verdict is not resolvable by this "
            "instrument at all and no capture will fix it; the band itself is finer "
            "than the measurement. The other four are resolvable except within one "
            "SD of a band edge, which is exactly where this subject happens to sit. "
            "CAVEAT: the burst is ~2.2x lower face resolution than the baseline "
            "photograph, so these SDs likely OVERSTATE the noise at full framing. "
            "That errs toward declaring verdicts unresolved, which is the safe "
            "direction, but it means they are an upper bound and not a final answer."
        ),
        module="analyze_extended.py",
        code_location="analyze_extended.py:57 (verdict), :109-150 (bands)",
    ),
    _r(
        key="repeatability.dimorphism_leans",
        metric="Measured repeatability of the per-feature sexual-dimorphism leans",
        value=None,
        units="",
        anchor_kind="none",
        tradition="measured in this project, 2026-08-01",
        evidence_tier=TIER_MEASUREMENT,
        source=(
            "MEASURED HERE — 5 near-frontal frames of one constant face from the "
            "26-second burst, run through analyze_advanced.dimorphism() unchanged."
        ),
        what_deviation_means=(
            "No reference value, and none should exist — this records how far the "
            "leans can be trusted. THIS IS THE BEST-BEHAVED AXIS MEASURED IN THIS "
            "PROJECT. Per-feature CVs are 1.7%-4.7% and 5 of the 7 features held "
            "their lean across every frame: jaw:bizygomatic (neutral), lower-third "
            "share (masculine), nose:bizygomatic (masculine), brow-eyelid gap "
            "(masculine), eye:bizygomatic (neutral). Two flipped, and only between "
            "ADJACENT bands rather than opposite ones: fWHR and lip-height:mouth-width "
            "each moved feminine<->neutral. That fWHR is one of them is consistent "
            "with the report already flagging it as weak and contested (Kramer 2017/"
            "2025) and partly adiposity rather than bone. The tally sentence therefore "
            "moves between '3 masculine / 2 neutral / 2 feminine' and '3 masculine / "
            "3 neutral / 1 feminine' on an unchanging face — which is a further "
            "argument for the existing refusal to sum it into a score, since the "
            "count itself is not stable to the photograph."
        ),
        module="analyze_advanced.py",
        code_location="analyze_advanced.py (dimorphism)",
    ),
    _r(
        key="repeatability.dimorphism_metric_spread",
        metric=(
            "How far each dimorphism ratio moves on ONE UNCHANGED FACE "
            "— ⚠️ SUPERSEDED 2026-08-19, HISTORICAL ONLY"
        ),
        value={
            "fWHR (bizygo width / upper-face height)": 0.036,
            "Jaw width / bizygomatic width": 0.003,
            "Lower-third share of face height": 0.002,
            "Nose width / bizygomatic width": 0.006,
            "Lip height / mouth width": 0.005,
            "Brow-to-eyelid gap / eye width": 0.003,
            "Eye width / bizygomatic width": 0.002,
        },
        units="ratio units (full observed range, max - min)",
        anchor_kind="threshold_ladder",
        tradition="measured in this project, 2026-08-19",
        evidence_tier=TIER_MEASUREMENT,
        source=(
            "MEASURED HERE across 12 readings of ONE UNCHANGED FACE, combining "
            "both FINISH-LINE measurement steps: 3 set-path SITTINGS spanning two "
            "days and a 46-point median-brightness range (162/120/124), plus 9 "
            "CAPTURE VARIANTS of a single photograph from one of those sittings "
            "(undegraded control, downscale 0.5x and 0.25x, dim 0.6x and 0.35x, "
            "JPEG q60 and q35, warm and cool white-balance shift). Every reading "
            "went through the full multi-view fit, so this is the movement that "
            "survives pose being solved and removed. It CONFIRMS AND EXTENDS the "
            "2026-08-01 burst result in repeatability.dimorphism_leans, which "
            "named fWHR and lip-height/mouth-width as the only two features that "
            "flip -- three independent regimes, the same two features, no others."
        ),
        what_deviation_means=(
            "⛔ SUPERSEDED 2026-08-19 AND NO LONGER READ BY ANYTHING. The seven "
            "ratios these margins were measured on no longer exist: the panel was "
            "rebuilt after measurement showed four of them could not distinguish "
            "male from female faces at all, and that nose width / bizygomatic "
            "width separated them BACKWARDS because its denominator is itself "
            "male-larger. The live margins are in dimorphism_reference.FEATURES "
            "under `movement`, keyed by a stable feature id rather than by the "
            "printed label. ⭐ THE MARGINS WERE RE-MEASURED, NOT CARRIED ACROSS, "
            "and that is the point of this note: a margin measured on `nose width "
            "/ bizygomatic width` applied to `nose width / mouth width` would be a "
            "number describing a different quantity, still passing its guard, and "
            "silently wrong. Kept here rather than deleted so the supersession is "
            "legible instead of looking like the entry never existed. What follows "
            "is the original text and describes the retired ratios.\n\n"
            "This is not a reference value for a face; it is how much each ratio "
            "moves when the FACE DOES NOT CHANGE and only the photograph does. It "
            "is consumed by analyze_advanced.dimorphism as the MARGIN around each "
            "value: a feature whose value sits within this range of its own "
            "neutral-band edge reports lean 'unresolved' instead of picking the "
            "side this photograph happened to land on. The margin is the FULL "
            "range rather than half of it, because the question is whether "
            "ANOTHER capture could read on the other side, and a reading at the "
            "top of its own observed range would otherwise clear the edge and "
            "assert a side -- measured: lip height wanders 0.455-0.477 across a "
            "0.46 cutoff, so the same face has been read on both sides of it, yet "
            "a half-range margin reported 'feminine' at 0.477. On the shipped bands that "
            "refuses exactly two of seven -- fWHR (0.074 across a 1.80 cutoff) and "
            "lip height / mouth width (0.022 across 0.46) -- while the other five "
            "move 0.006-0.020 and sit clear of their bands, so they still resolve. "
            "⛔ THE BANDS THEMSELVES ARE NOT WIDENED. Moving a cutoff would change "
            "what the feature means for every subject; bracketing changes only the "
            "confidence this instrument claims. ⚠️ LIMITS: one person, one camera, "
            "and 3 real sittings -- the degradation variants are synthetic and share "
            "one sitting's pose, so this is a FLOOR on the true movement, not a "
            "ceiling. A second real subject can only widen it."
            "⭐ RE-SCOPED 2026-08-19 AFTER calibration_subject PUSHED BACK, and he was right: the "
            "brackets were refusing on GOOD photographs. Two things were wrong. (1) "
            "SCOPE -- the first version pooled three sittings AND nine synthetic "
            "degradations, then applied that margin to a pristine capture. A "
            "cross-sitting margin answers 'will this person read the same if they "
            "shoot again'; synthetic degradations of ONE sitting do not measure that, "
            "and Step 2 measures capture-quality sensitivity separately with its own "
            "pass condition. (2) ADMISSIBILITY -- one of the three sittings spans 38 "
            "degrees against 85 for the other two and was the outlier on most axes. It "
            "passed only because intake's floor was 26, a MEASURED-SIM number for "
            "SHAPE RECOVERY. A ladder on real photographs put the floor for the DERIVED "
            "readings at 52 (capture_gate.min_effective_spread_deg_real) and that "
            "sitting is now REFUSED. So these margins are the GATE-PASSING CROSS-SITTING "
            "movement, they shrank 3-5x, and calibration_subject report went from 4 unresolved "
            "harmony axes and 2 unresolved dimorphism features back to 1 and 0. "
            "⚠️ n=2 SITTINGS, WHICH IS THIN. These are a FLOOR and can only grow when a "
            "third well-spread sitting exists. A measured range of 0.0 is undersampling, "
            "NOT evidence of a motionless axis."
        ),
        module="analyze_advanced.py",
        code_location="analyze_advanced.py (dimorphism, _bracket)",
    ),
    _r(
        key="repeatability.harmony_magnitude_spread",
        metric="How far each harmony axis's GRADED MAGNITUDE moves on one unchanged face",
        value={
            "Vertical balance": 1.69,
            "Horizontal balance": 0.435,
            "Midface proportion": 0.03,
            "Nose\u2013mouth\u2013eye proportion": 0.044,
            "Lip balance": 0.01,
            "Lower-face / chin balance": 0.1625,
            "Profile balance": 0.9278,
            "Left\u2013right balance": 0.59,
        },
        units="axis-native units of the magnitude band() grades (NOT the reported value)",
        anchor_kind="threshold_ladder",
        tradition="measured in this project, 2026-08-19",
        evidence_tier=TIER_MEASUREMENT,
        source=(
            "MEASURED HERE as the FULL OBSERVED RANGE of each axis's graded magnitude "
            "across the THREE GATE-PASSING SITTINGS of one unchanged face -- second-set "
            "(84.5 deg), third-set (84.8 deg) and fourth-sitting (81.8 deg effective "
            "spread) -- read from the per_axis_magnitude recorded in step1_readings.json "
            "by step1_cross_sitting.py, which refuses to record unless every artifact is "
            "provably newer than the run it describes. ⭐ THESE ARE THE GRADED MAGNITUDES, NOT THE REPORTED "
            "VALUES, and the distinction is the whole reason this entry exists. For "
            "four of the eight axes they are different quantities entirely -- "
            "'Profile balance' reports 126.6 and grades 0.66 against 1.0/2.2; "
            "'Lower-face / chin balance' reports the philtrum split (0.52-0.55) and "
            "grades a combination against 0.12/0.28. FINISH-LINE Step 2 found two "
            "axes flipping verdict under degradation alone and could NOT apply the "
            "dimorphism fix, because bracketing the reported number against the "
            "threshold would have bracketed the wrong quantity: principled-looking, "
            "and refusing or resolving for reasons unrelated to the measurement. "
            "analyze_harmony.band() now returns the magnitude it graded and the "
            "thresholds it graded against, so the two travel together and cannot "
            "drift into two hand-maintained lists."
        ),
        what_deviation_means=(
            "Not a reference value for a face: how much each axis's graded "
            "magnitude moves when the FACE DOES NOT CHANGE and only the photograph "
            "does. Consumed by analyze_harmony as the margin around each band edge "
            "-- an axis whose magnitude sits within this much of `mild_at` or "
            "`off_at` reports 'unresolved' rather than picking the side this "
            "photograph landed on. On the measured readings that refuses the two "
            "axes that were flipping (Horizontal balance, 2.105-3.715 across a 3.0 "
            "cutoff; Lower-face / chin, 0.173-0.490 across 0.28) across their WHOLE "
            "observed range, so neither can flip side again. ⛔ NO BAND WAS WIDENED "
            "AND NO CUTOFF MOVED. ⚠️ Two further axes (Midface proportion, "
            "Nose-mouth-eye) sit near an edge at part of their range and so move "
            "between an asserted side and 'unresolved'. That is the confidence "
            "tracking the evidence rather than an instability: 'cannot tell' is a "
            "WEAKER statement than 'leans X', never a contradiction of it. ⚠️ LIMITS: "
            "one person, one camera, 3 real sittings; the 9 variants are synthetic "
            "and share one sitting's pose, so this is a FLOOR."
            "⭐ RE-SCOPED 2026-08-19 AFTER calibration_subject PUSHED BACK, and he was right: the "
            "brackets were refusing on GOOD photographs. Two things were wrong. (1) "
            "SCOPE -- the first version pooled three sittings AND nine synthetic "
            "degradations, then applied that margin to a pristine capture. A "
            "cross-sitting margin answers 'will this person read the same if they "
            "shoot again'; synthetic degradations of ONE sitting do not measure that, "
            "and Step 2 measures capture-quality sensitivity separately with its own "
            "pass condition. (2) ADMISSIBILITY -- one of the three sittings spans 38 "
            "degrees against 85 for the other two and was the outlier on most axes. It "
            "passed only because intake's floor was 26, a MEASURED-SIM number for "
            "SHAPE RECOVERY. A ladder on real photographs put the floor for the DERIVED "
            "readings at 52 (capture_gate.min_effective_spread_deg_real) and that "
            "sitting is now REFUSED. So these margins are the GATE-PASSING CROSS-SITTING "
            "movement, they shrank 3-5x, and calibration_subject report went from 4 unresolved "
            "harmony axes and 2 unresolved dimorphism features back to 1 and 0. "
            "⭐⭐ n=3 SINCE 2026-08-19d, AND THE MARGINS GREW. calibration_subject shot a fourth "
            "sitting -- different day, fresh setup, NEW ROOM, 81.8 deg effective spread "
            "against the 52-deg floor -- and it is the third genuinely independent one, "
            "which is the session variation these margins exist to capture. Folding it "
            "in WIDENED five of the eight axes: Vertical balance 0.09 -> 1.69 (18.8x), "
            "Lower-face / chin 0.0208 -> 0.1625 (7.8x), Profile balance 0.3445 -> 0.9278 "
            "(2.7x), Horizontal balance 0.33 -> 0.435, Lip balance 0.0 -> 0.01. Three "
            "did not move, because the fourth sitting landed INSIDE their existing "
            "range -- which is what a full-observed-range margin does and is not a "
            "failure to measure. ⛔ THE GROWTH IS NOT A REGRESSION. The margin is the "
            "full observed range, so a third sitting can only widen it; Profile balance "
            "and Lower-face / chin consequently stop asserting a side and report "
            "unresolved. That is correct: Lower-face / chin was READ ON BOTH SIDES of "
            "its own 0.28 cutoff by the same unchanged face (0.2517, 0.2725, 0.4142), "
            "so an asserted side was never supported -- step1_cross_sitting reported it "
            "as a CROSS-SITTING flip until this margin was widened to the measured "
            "movement, which is the module's own prescribed fix. ⛔ AND NOTHING GOES "
            "SILENT: the axis still prints its value and its relation; only the SIDE is "
            "withheld. ⚠️ STILL n=1 PERSON and one camera. A FLOOR, not a ceiling. "
            "A measured range of 0.0 is undersampling, NOT evidence of a motionless "
            "axis -- Lip balance read exactly 0.0 at n=2 and 0.01 at n=3, which is that "
            "warning coming true on the first extra sitting."
        ),
        module="analyze_harmony.py",
        code_location="analyze_harmony.py (band, build_axes)",
    ),
    _r(
        key="capture_gate.skin_calibration_face_lstar",
        metric="Median face L* of the capture the skin thresholds were calibrated on",
        value=55.29,
        units="CIELAB L* over the gated skin mask",
        anchor_kind="point",
        tradition="measured in this project, 2026-08-19",
        evidence_tier=TIER_MEASUREMENT,
        source=(
            "MEASURED HERE off the two-photo frontal -- the same capture "
            "capture_gate.skin_calibration_ipd_px (676 px IPD) refers to, so the "
            "resolution and exposure halves of the comparability gate describe one "
            "photograph rather than two. MEASURED WITH analyze_skin's OWN skin mask, "
            "not an ad-hoc one: a first pass used a hand-built face-oval-plus-colour-gate "
            "mask and got 53.3, which left the calibration capture reading 3.7 percent "
            "off its own calibration and silently ate a third of the tolerance below. "
            "A calibration constant has to be measured by the code that consumes it."
        ),
        what_deviation_means=(
            "How far this photograph's exposure sits from the one the lesion-count "
            "and IGA thresholds were set on. Consumed by analyze_skin's capture "
            "condition: the counts are called comparable only when BOTH the face "
            "resolution and this ratio are in band. Before 2026-08-19 only "
            "resolution was checked, which is why the flag read False on every "
            "set-path run including the UNDEGRADED control and True on the "
            "two-photo path regardless of lighting -- a standing disclaimer rather "
            "than a gate that responds to anything."
        ),
        module="analyze_skin.py",
        code_location="analyze_skin.py (capture condition)",
    ),
    _r(
        key="repeatability.skin_counts_vs_exposure",
        metric="How far the lesion count and IGA grade move when only exposure changes",
        value=(0.85, 1.30, 0.35),
        units="exposure multipliers at which the counts were measured to move",
        anchor_kind="threshold_ladder",
        tradition="measured in this project, 2026-08-19",
        evidence_tier=TIER_MEASUREMENT,
        source=(
            "MEASURED HERE on ONE photograph of one unchanged face, scaled and re-run "
            "end to end (FINISH-LINE Step 4). L* ratio tracks the multiplier closely: "
            "x1.30 -> L* 67.8 (ratio 1.27), 11 lesions against the control's 3 (+267%) "
            "and texture flipping smooth -> moderate; x0.85 -> L* 45.5 (ratio 0.85), "
            "4 lesions (+33%); x0.70 -> L* 37.7, IGA 3 against the control's 2; "
            "x0.60 -> 2 lesions; x0.35 -> L* 18.0, IGA 0 and 0 lesions. ⭐ THE "
            "BRIGHTENING DIRECTION HAD NEVER BEEN TESTED and is the worst result in "
            "the ladder -- every earlier probe only dimmed."
        ),
        what_deviation_means=(
            "That the acne counts are a property of the exposure as much as of the "
            "skin. ⛔ NO TOLERANCE HAS BEEN DEMONSTRATED: the smallest deviation ever "
            "tested is 15% and it already moved the count by a third, so there is no "
            "measured safe band -- only an untested gap between 1.00 and 0.85. The "
            "gate's +/-10% band is therefore AUTHOR-CHOSEN, and chosen to sit INSIDE "
            "the smallest deviation measured to move anything rather than outside it. "
            "Tightening it further is defensible; loosening it is not, without "
            "measuring the gap first."
        ),
        module="analyze_skin.py",
        code_location="analyze_skin.py (capture condition)",
    ),
    _r(
        key="capture_gate.min_effective_spread_deg_real",
        metric="Effective angular spread below which the DERIVED readings stop being usable",
        value=52.0,
        units="degrees of effective angular spread across the fitted views",
        anchor_kind="point",
        tradition="measured in this project, 2026-08-19",
        evidence_tier=TIER_MEASUREMENT,
        source=(
            "MEASURED-REAL, on ONE sitting of one face subsetted to a spread ladder "
            "and run end to end, so only angular coverage changes. Worst deviation of "
            "any harmony axis from the 84.8-degree reference: 28.8 deg -> 2.48, "
            "36.8 deg -> 3.55, 52.4 deg -> 0.55, 64.9 deg -> 0.47. The axis that "
            "collapses is LEFT-RIGHT BALANCE, reading 4.87 at 36.8 degrees against "
            "1.32 at 84.8 -- unsolved head turn leaking into the symmetry reading. "
            "SUPERSEDES NOTHING AUTOMATICALLY: intake's own floor is MEASURED-SIM, "
            "from a synthetic sweep where the true shape is known, and is "
            "authoritative for SHAPE RECOVERY. This is the DERIVED-READING floor, "
            "and intake.py's docstring already says MEASURED-REAL wins for failure "
            "modes the simulation does not model. The gate takes the MAX of the two."
        ),
        what_deviation_means=(
            "That the set does not span enough angle for pose to be solved and "
            "removed, so what is left contaminates the derived axes rather than "
            "widening them. 52 is the lowest rung TESTED GOOD; 37 is catastrophic "
            "and the 37-52 gap is untested, so a floor between them would be "
            "author-chosen dressed as measured. LIMITS: one subject, one sitting, "
            "one camera. It can only TIGHTEN the gate, never admit more. In practice "
            "it asks a person to turn about 26 degrees each way rather than 13."
        ),
        module="intake.py",
        code_location="intake.py (load_calibration)",
    ),
    _r(
        key="capture_gate.expression_blendshape_max",
        metric="Blendshape scores above which a capture is rejected as posed",
        value=(0.45, 0.40, 0.20, 0.15, 0.25),
        units="blendshape score 0-1: eyeSquint, eyeBlink, mouthSmile, jawOpen, browDown",
        anchor_kind="threshold_ladder",
        tradition="this pipeline's capture gate",
        evidence_tier=TIER_AUTHOR,
        source=(
            UNSOURCED
            + " — AUTHOR-CHOSEN, but informed by measurement rather than guessed. "
            "FaceLandmarker blendshapes were read across 11 real frames of one "
            "subject (a 10-frame burst plus the baseline). Relaxed near-frontal "
            "frames sat at eyeSquint 0.15-0.33, eyeBlink 0.03-0.15, mouthSmile "
            "<=0.003, jawOpen <=0.004, browDown <=0.025. Each cutoff is set clear "
            "of that observed range so an ordinary neutral face passes. NOT "
            "validated against posed captures, and n=1 subject."
        ),
        what_deviation_means=(
            "That the face is doing something — squinting, smiling, opening the jaw, "
            "pulling the brows — rather than resting. Expression moves the soft "
            "tissue the measurements are taken from, so it is a capture defect, not "
            "a finding. ⚠️ TWO CONSTANTS WERE DELIBERATELY EXCLUDED after measuring "
            "them: mouthPucker read 0.10-0.59 on faces that were doing nothing at "
            "all, so it carries no signal about posing and would have produced "
            "false rejections; cheekPuff was identically zero on every frame and "
            "carries no signal either way. ⚠️ ORDERING CONSTRAINT: blendshapes are "
            "POSE-CONTAMINATED. At yaw the far eye is occluded and the model reports "
            "it as a squint — measured 0.88 on a 58-degree frame against 0.15-0.33 "
            "relaxed. The expression check is therefore only meaningful once the "
            "pose gate has passed, and the implementation SKIPS it rather than "
            "guessing when pose fails."
        ),
        module="capture_check.py",
        code_location="capture_check.py (EXPRESSION)",
    ),
    _r(
        key="capture_gate.sharpness_exposure",
        metric="Focus and exposure floors for a usable capture",
        value=(60.0, 2.0, 60.0, 205.0),
        units="Laplacian variance on a 512px face crop; % clipped; luma min, max",
        anchor_kind="threshold_ladder",
        tradition="this pipeline's capture gate",
        evidence_tier=TIER_AUTHOR,
        source=UNSOURCED + " — AUTHOR-CHOSEN. The baseline photograph scores 487 "
        "focus, 0.9% clipped, median luma 123, so the floors sit well below a known "
        "good capture. They have NOT been calibrated against a set of known-bad ones.",
        what_deviation_means=(
            "That the photograph is soft, motion-blurred, or lit in a way that "
            "crushes or blows out skin detail. Exposure matters more here than it "
            "sounds: the acne-severity grade was measured moving a whole band, "
            "across its own dermatologist-referral threshold, on unchanging skin — "
            "and the driver was illumination, not pose. Sharpness is normalised on a "
            "512 px face crop so the threshold does not itself become "
            "resolution-dependent, which is the defect the skin layer already had."
        ),
        module="capture_check.py",
        code_location="capture_check.py (SHARPNESS_MIN, CLIP_MAX_PCT, LUMA_BAND)",
    ),
    _r(
        key="pose_gate.profile_yaw_deg",
        metric="Head-turn band in which profile cephalometrics are comparable",
        value=(62.0, 75.0),
        units="degrees |yaw| (comparable band, absolute usable max)",
        anchor_kind="threshold_ladder",
        tradition="measured in this project, 2026-08-01",
        evidence_tier=TIER_MEASUREMENT,
        source=(
            "MEASURED HERE — captures/2026-08-01_profile-burst. Eight profile "
            "captures in one sitting, of which 4 were inside the detector's usable "
            "range; the other 4 were turned too far for the landmarker to anchor. "
            "Measured turns: 51, 57, 63, 64 deg, plus the June baseline at 67 deg "
            "from a separate session. Run through the real analyze_profile stage."
        ),
        what_deviation_means=(
            "That the cephalometric angles are not comparable to the ones this "
            "pipeline published, because they move predictably with head turn. "
            "MEASURED slopes across 51-64 deg, all with near-perfect correlation: "
            "facial convexity -0.72 deg per degree of turn (r=-0.976), nasolabial "
            "+0.29 (r=+0.964), chin projection -0.45 pp (r=-0.935). Ten degrees of "
            "extra turn moves convexity by about 7 degrees, which is larger than any "
            "real difference between two faces this report would call meaningful. "
            "The 4-of-8 usable rate is itself worth knowing: half of a deliberate "
            "attempt by an informed subject fell outside the detector's range."
        ),
        module="analyze_profile.py",
        code_location="analyze_profile.py (pose_condition)",
        bands=(
            "|yaw| 62-75 deg — 'measured': near the 67 deg baseline these figures "
            "were published from, so the angle term is small.",
            "|yaw| 50-62 deg — 'pose_shifted': usable, but the measured slopes above "
            "apply and the values are NOT comparable to the published ones.",
            "outside 50-75 deg — 'refused': below, it is a turned frontal; above, the "
            "far eye is gone and the landmarker cannot anchor at all.",
        ),
    ),
    _r(
        key="repeatability.profile_cephalometrics",
        metric="Measured repeatability of the profile cephalometric angles",
        value=None,
        units="",
        anchor_kind="none",
        tradition="measured in this project, 2026-08-01",
        evidence_tier=TIER_MEASUREMENT,
        source=(
            "MEASURED HERE — the same 8-capture sitting. The fixed-pose figure comes "
            "from the two frames that landed 1 degree apart (63 and 64 deg), which is "
            "the closest this data gets to a re-shot identical pose. n=2 for that "
            "pair: it is a single difference, NOT a standard deviation, and should be "
            "read as one observation rather than a distribution."
        ),
        what_deviation_means=(
            "No reference value; this records how far these angles can be trusted, "
            "and the answer has three distinct tiers that must not be collapsed. "
            "(1) AT FIXED POSE, WITHIN A SITTING, these are the STEADIEST numbers in "
            "this project: re-shot 1 degree apart, nasolabial moved 0.3% and facial "
            "convexity 0.8% — better than symmetry, averageness or the skin grade. "
            "(2) ACROSS TURN ANGLE they move strongly but PREDICTABLY (see "
            "pose_gate.profile_yaw_deg), so the variation is a correctable systematic, "
            "not noise. (3) ACROSS SESSIONS THEY DO NOT REPRODUCE, and this is the "
            "consequential one: the June baseline at 67 deg sits far off the line "
            "fitted to the August sitting — nasolabial -9.2 deg, convexity +4.6 deg, "
            "chin projection +10.5 pp from prediction. Angle does not explain that, so "
            "a session term (day, lighting, camera distance, framing) dominates. This "
            "is the project's first genuine cross-session test-retest, and it FAILED. "
            "Two further notes: cervicomental moved 12.8 deg between the fixed-pose "
            "pair and shows no angle trend at all (r=-0.31) — it is simply noise, "
            "which independently vindicates the module already declining to report it "
            "as well-defined; and chin projection is the most fragile of the surviving "
            "three, moving 7.6% on a 1-degree change."
        ),
        module="analyze_profile.py",
        code_location="analyze_profile.py, analyze_extended.py (profile-derived rows)",
    ),
    # -----------------------------------------------------------------------
    # fa_refine.py / analyze_anatomy.py / analyze_skin.py — scale constants
    # -----------------------------------------------------------------------
    _r(
        key="scale.iris_diameter_mm",
        metric="Assumed horizontal visible iris (limbus) diameter — px→mm scale",
        value=11.7,
        units="mm (SD ~0.5 mm)",
        anchor_kind="point",
        tradition="ophthalmic / forensic photogrammetry scale constant",
        evidence_tier=TIER_MEASUREMENT,
        source=(
            UNSOURCED + ' — fa_refine.py:47-50 describes it as a "standard ophthalmic/'
            'forensic constant", but names no publication, author or dataset.'
        ),
        what_deviation_means=(
            "Not an ideal at all — nothing is graded against it. It converts "
            "pixels to millimetres, so its ±4.3% 1-sigma error propagates into "
            "every mm figure downstream (notably the reported IPD, which is "
            "DERIVED from this scale and therefore not an independent check of "
            'it — README.md\'s "Limitations"). Foreshortening and detection error '
            "sit on top."
        ),
        module="fa_refine.py",
        code_location="fa_refine.py:50 (IRIS_MM), fa_refine.py:51 (IRIS_SD_MM)",
    ),
    _r(
        key="scale.assumed_bizygomatic_mm",
        metric="Assumed adult male bizygomatic (cheekbone) width",
        value=137.0,
        units="mm",
        anchor_kind="point",
        tradition="anthropometric population mean, asserted",
        evidence_tier=TIER_ANTHROPOMETRIC,
        source=(
            UNSOURCED
            + ' — analyze_skin.py:42-43 calls it an "honest single assumption" '
            "and analyze_anatomy.py:25 uses the same literal, but neither records "
            "a population, sex-specific dataset, or citation."
        ),
        what_deviation_means=(
            "Not an ideal — a second, independent px→mm scale. Any subject whose "
            "true bizygomatic width differs from 137 mm has every derived mm "
            "figure scaled by that error: the philtrum-length band "
            "(anatomy.philtrum_mm_bands) and the acne-lesion size window "
            "(skin.lesion_size_window) both inherit it. analyze_anatomy.py:204-206 "
            "already tells the reader to treat the philtrum mm as a band, not an "
            "exact value."
        ),
        module="analyze_anatomy.py, analyze_skin.py",
        code_location="analyze_anatomy.py:25, analyze_skin.py:43",
        note=(
            "The pipeline carries TWO unrelated px→mm scales — this one and the "
            "iris scale — and does not reconcile them."
        ),
    ),
    # -----------------------------------------------------------------------
    # analyze_profile.py — the annotation-only references (the real defect)
    # -----------------------------------------------------------------------
    _r(
        key="profile.nasolabial_angle",
        metric="Nasolabial angle (columella vs upper-lip tangents)",
        value=(90.0, 95.0),
        units="degrees (stated as 'male')",
        anchor_kind="range",
        tradition="cephalometric / facial-surgery aesthetic convention",
        evidence_tier=TIER_CLINICAL,
        source=UNSOURCED,
        what_deviation_means=(
            "Almost nothing as currently wired, for two compounding reasons. "
            "First, " + _ANNOTATION_ONLY_DEFECT + " Second, the measured value is "
            "not commensurable with the reference: analyze_profile.py:135-139 "
            "records that silhouette-based nasolabial values read ~15-25° higher "
            "than the clinical columella-based measurement the 90-95° convention "
            "comes from. Comparing the drawn number to the drawn ideal is "
            "therefore an apples-to-oranges comparison shown to the reader with "
            "no correction applied."
        ),
        module="analyze_profile.py",
        code_location="analyze_profile.py:366 (panel label) + res['reference_bands'] in metrics_profile.json",
        emitted_to_json=True,
        note=(
            "A sex qualifier ('male') is asserted with no female counterpart "
            "anywhere in the codebase and no source for either."
        ),
    ),
    _r(
        key="profile.facial_convexity",
        metric="Facial convexity angle (glabella–subnasale–pogonion)",
        value=(165.0, 175.0),
        units="degrees (labelled '= straight')",
        anchor_kind="range",
        tradition="cephalometric / orthodontic soft-tissue profile convention",
        evidence_tier=TIER_CLINICAL,
        source=UNSOURCED,
        what_deviation_means=(
            "A more convex (lower-angle) profile in the orthodontic vocabulary; "
            "the convention describes profile straightness, not attractiveness. "
            + _ANNOTATION_ONLY_DEFECT
        ),
        module="analyze_profile.py",
        code_location="analyze_profile.py:367 (panel label) + res['reference_bands'] in metrics_profile.json",
        emitted_to_json=True,
        note=(
            "Distinct from ext.total_convexity (125-140°), which includes the "
            "nose. The two are different measurements with different reference "
            "bands and the report shows both."
        ),
    ),
    _r(
        key="profile.cervicomental_angle",
        metric="Cervicomental angle (menton–cervical point–neck)",
        value=(105.0, 120.0),
        units="degrees",
        anchor_kind="range",
        tradition="cephalometric / facial-surgery aesthetic convention",
        evidence_tier=TIER_CLINICAL,
        source=UNSOURCED,
        what_deviation_means=(
            "On this subject, nothing: analyze_profile.py:176-182 flags the angle "
            "as NOT reliably measurable (no distinct submental concavity; shadow "
            "and an equipment bar cross the lower neck) and the annotation appends "
            "'?' / '(ill-defined)'. So an unsourced ideal is drawn next to a value "
            "the module itself disowns. " + _ANNOTATION_ONLY_DEFECT
        ),
        module="analyze_profile.py",
        code_location="analyze_profile.py:369 (panel label) + res['reference_bands'] in metrics_profile.json",
        emitted_to_json=True,
    ),
    # -----------------------------------------------------------------------
    # analyze_extended.py — the seven finer ratio/angle reference ranges
    # -----------------------------------------------------------------------
    _r(
        key="ext.eye_spacing",
        metric="Eye spacing (intercanthal distance / eye width)",
        value=(0.97, 1.03),
        units="ratio",
        anchor_kind="range",
        tradition="neoclassical canon (eyes one eye-width apart)",
        evidence_tier=TIER_ARTISTIC,
        source=UNSOURCED,
        what_deviation_means=(
            "Wider- or closer-set eyes than the drawing convention. "
            + _CANON_MEANS_NOTHING
        ),
        module="analyze_extended.py",
        code_location="analyze_extended.py:104-111",
        bands=("in range", "borderline within ±25% of span", "outside"),
    ),
    _r(
        key="ext.fwhr",
        metric="fWHR (bizygomatic width / upper-face height)",
        value=(1.75, 2.05),
        units="ratio",
        anchor_kind="range",
        tradition="facial-morphology research metric, used here as an aesthetic band",
        evidence_tier=TIER_AUTHOR,
        source=UNSOURCED,
        what_deviation_means=(
            "Very little. This project's OWN sourced material contradicts reading "
            "fWHR as informative: dimorphism.fwhr records a meta-analytic sex "
            "effect of d~0.11, heavy BMI confounding, and failures to replicate. "
            "Treat a value outside this band as near-uninformative."
        ),
        module="analyze_extended.py",
        code_location="analyze_extended.py:112-119",
        bands=("in range", "borderline within ±25% of span", "outside"),
        note=(
            "COLLISION: the same metric carries a different reference band in "
            "analyze_advanced.py (dimorphism.fwhr, 1.80-2.00), which IS sourced. "
            "Two bands for one measurement, in one report."
        ),
    ),
    _r(
        key="ext.jaw_over_cheekbone",
        metric="Jaw / cheekbone width (bigonial : bizygomatic)",
        value=(0.70, 0.80),
        units="ratio",
        anchor_kind="range",
        tradition="facial-shape description (tapered vs square)",
        evidence_tier=TIER_AUTHOR,
        source=UNSOURCED,
        what_deviation_means=(
            "A squarer (higher) or more tapered (lower) lower face than the band. "
            "Descriptive of shape only; analyze_extended.py:6 states the reference "
            "ranges are conventional/clinical aesthetic targets, NOT validated "
            "predictors of attractiveness."
        ),
        module="analyze_extended.py",
        code_location="analyze_extended.py:120-127",
        bands=("in range", "borderline within ±25% of span", "outside"),
        note=(
            "COLLISION: dimorphism.jaw_bizygomatic uses 0.74-0.82 for the same "
            "ratio and that band IS sourced. Also note analyze_advanced.py:249-251 "
            "warns this pipeline's mesh 'jaw width' is not true bigonial width."
        ),
    ),
    _r(
        key="ext.midface_ratio",
        metric="Midface ratio (inner-eye line → upper lip / interpupillary distance)",
        value=(0.95, 1.10),
        units="ratio",
        anchor_kind="range",
        tradition="conventional aesthetic 'compact midface' target",
        evidence_tier=TIER_AUTHOR,
        source=UNSOURCED,
        what_deviation_means=(
            "A taller or shorter midface relative to eye spacing than the "
            "convention. Descriptive; no validated link to anything recorded here."
        ),
        module="analyze_extended.py",
        code_location="analyze_extended.py:128-135",
        bands=("in range", "borderline within ±25% of span", "outside"),
        note="Re-used as harmony.midface_proportion with a centre of 1.0.",
    ),
    _r(
        key="ext.lip_ratio",
        metric="Lip ratio (upper : lower vermilion height)",
        value=(0.55, 0.72),
        units="ratio",
        anchor_kind="range",
        tradition="conventional aesthetic target (lower lip fuller)",
        evidence_tier=TIER_AUTHOR,
        source=UNSOURCED,
        what_deviation_means=(
            "A relatively fuller upper or lower lip than the convention. "
            "Vermilion height also changes markedly with age (thinning), which "
            "analyze_advanced.py:294-297 records as confounding any fixed read."
        ),
        module="analyze_extended.py",
        code_location="analyze_extended.py:136-143",
        bands=("in range", "borderline within ±25% of span", "outside"),
        note="Re-used as harmony.lip_balance with a centre of 0.62.",
    ),
    _r(
        key="ext.total_convexity",
        metric="Total profile convexity including the nose (glabella–nasal tip–pogonion)",
        value=(125.0, 140.0),
        units="degrees",
        anchor_kind="range",
        tradition="cephalometric profile convention (nose-inclusive variant)",
        evidence_tier=TIER_CLINICAL,
        source=UNSOURCED,
        what_deviation_means=(
            "Lower = a more projected nose relative to the brow-to-chin line. "
            "Describes profile geometry, not quality."
        ),
        module="analyze_extended.py",
        code_location="analyze_extended.py:157-164",
        bands=("in range", "borderline within ±25% of span", "outside"),
        note=(
            "Its midpoint (132.5°) is re-used as the centre of "
            "harmony.profile_balance (analyze_harmony.py:280)."
        ),
    ),
    _r(
        key="ext.mentolabial_angle",
        metric="Mentolabial angle (lower lip – sulcus – chin)",
        value=(110.0, 130.0),
        units="degrees",
        anchor_kind="range",
        tradition="cephalometric / facial-surgery aesthetic convention",
        evidence_tier=TIER_CLINICAL,
        source=UNSOURCED,
        what_deviation_means=(
            "Higher = a shallower chin fold, lower = a deeper one. Contour "
            "description, not a quality judgement."
        ),
        module="analyze_extended.py",
        code_location="analyze_extended.py:165-172",
        bands=("in range", "borderline within ±25% of span", "outside"),
        note=(
            "Its midpoint (120°) is re-used inside harmony.lower_face_chin "
            "(analyze_harmony.py:251)."
        ),
    ),
    _r(
        key="ext.borderline_margin",
        metric="'Borderline' margin around every extended reference range",
        value=0.25,
        units="× the range span, added to each side",
        anchor_kind="threshold_ladder",
        tradition="this pipeline's verdict logic",
        evidence_tier=TIER_AUTHOR,
        source=UNSOURCED,
        what_deviation_means=(
            "Only that a value fell outside a project-chosen soft margin. Because "
            "the margin scales with span, a wide reference range gets a wide "
            "borderline zone — a metric with a loose band is structurally harder "
            "to call 'outside'."
        ),
        module="analyze_extended.py",
        code_location="analyze_extended.py:58-66 (verdict)",
    ),
    _r(
        key="ext.nasion_distinct_gate",
        metric="Nasal-root concavity gate — drop nasion-based angles below this depth",
        value=0.04,
        units="× (pogonion.y − glabella.y)",
        anchor_kind="threshold_ladder",
        tradition="this pipeline's refuse-rather-than-fabricate rule",
        evidence_tier=TIER_AUTHOR,
        source=UNSOURCED,
        what_deviation_means=(
            "Not a facial judgement — a measurability gate. Below it the brow "
            "flows smoothly into the bridge, nasion collapses onto glabella, and "
            "the nasofrontal / nasomental / nasofacial angles are DROPPED rather "
            "than reported as noise (analyze_extended.py:152-156, 213-217). The "
            "cutoff value itself is unsourced, but the refusal it triggers is the "
            "behaviour this project wants."
        ),
        module="analyze_extended.py",
        code_location="analyze_extended.py:155",
    ),
    _r(
        key="ext.face_shape_bands",
        metric="Face-shape classification cutoffs (height:width and jaw taper)",
        value=(1.45, 1.15, 0.74, 0.92),
        units="height/width >1.45 = oblong; >1.15 with taper <0.74 = oval; taper >0.92 = square/round",
        anchor_kind="threshold_ladder",
        tradition="styling heuristic",
        evidence_tier=TIER_AUTHOR,
        source=UNSOURCED,
        what_deviation_means=(
            "Which styling label the face is given. Pure convenience "
            "categorisation with no measurement or aesthetic claim behind the "
            "cutoffs."
        ),
        module="analyze_extended.py",
        code_location="analyze_extended.py:190-199",
    ),
    # -----------------------------------------------------------------------
    # analyze_anatomy.py
    # -----------------------------------------------------------------------
    _r(
        key="anatomy.cheekbone_height_bands",
        metric="Cheekbone (malar) height — position of the widest point between eyes and nose base",
        value=(0.30, 0.62),
        units="fraction of canthus→subnasale span (<0.30 high, ≤0.62 mid, >0.62 low)",
        anchor_kind="threshold_ladder",
        tradition="facial-aesthetics description (high vs low cheekbones)",
        evidence_tier=TIER_AUTHOR,
        source=UNSOURCED,
        what_deviation_means=(
            "Which of three words the cheekbone position is labelled with. The "
            "two cutoffs are bare literals with no recorded derivation, so the "
            "boundary between 'high' and 'mid' is a project choice, not a "
            "measured population split."
        ),
        module="analyze_anatomy.py",
        code_location="analyze_anatomy.py:125",
    ),
    _r(
        key="anatomy.philtrum_mm_bands",
        metric="Philtrum length (subnasale → labrale superius) classification",
        value=(11.0, 15.0),
        units="mm (<11 short, ≤15 average, >15 long)",
        anchor_kind="threshold_ladder",
        tradition="anthropometric description, asserted",
        evidence_tier=TIER_AUTHOR,
        source=UNSOURCED,
        what_deviation_means=(
            "Which of three words the philtrum is labelled with. Two error "
            "sources sit under the label: the cutoffs are unsourced, AND the mm "
            "value depends on scale.assumed_bizygomatic_mm (137 mm), which "
            "analyze_anatomy.py:204-206 already warns should be read as a band."
        ),
        module="analyze_anatomy.py",
        code_location="analyze_anatomy.py:130",
    ),
    # -----------------------------------------------------------------------
    # analyze_harmony.py — eight relational axes, all author-chosen
    # -----------------------------------------------------------------------
    _r(
        key="harmony.vertical_balance",
        metric="Vertical balance — lower third minus upper third",
        value=0.0,
        units="percentage points (0 = thirds even)",
        anchor_kind="point",
        tradition="relational aesthetic description (this pipeline's synthesis)",
        evidence_tier=TIER_AUTHOR,
        source=UNSOURCED,
        what_deviation_means=(
            "Bottom-weighted (positive) or top-weighted (negative) relative to "
            "its own neighbours. analyze_harmony.py:9-15 is explicit that harmony "
            "is descriptive, NOT an empirically validated attractiveness "
            "predictor, and that 'off' means a direction, not a defect."
        ),
        module="analyze_harmony.py",
        code_location="analyze_harmony.py:98-99",
        bands=("mildly off at ≥6.0 pp", "off at ≥14.0 pp"),
    ),
    _r(
        key="harmony.horizontal_balance",
        metric="Horizontal balance — intercanthal fifth minus average eye-width fifth",
        value=0.0,
        units="percentage points (0 = eye gap equals one eye width)",
        anchor_kind="point",
        tradition="relational restatement of the neoclassical fifths canon",
        evidence_tier=TIER_AUTHOR,
        source=UNSOURCED,
        what_deviation_means=(
            "Eyes set wider or closer than their own width. Inherits whatever the "
            'fifths canon is worth — which, per README.md\'s "Limitations", is '
            "nothing as an "
            "attractiveness claim."
        ),
        module="analyze_harmony.py",
        code_location="analyze_harmony.py:122-124",
        bands=("mildly off at ≥3.0 pp", "off at ≥7.0 pp"),
    ),
    _r(
        key="harmony.midface_proportion",
        metric="Midface proportion — midface height / interpupillary distance",
        value=1.0,
        units="ratio (balance centre)",
        anchor_kind="point",
        tradition="conventional 'compact midface' target",
        evidence_tier=TIER_AUTHOR,
        source=UNSOURCED,
        what_deviation_means=(
            "A midface taller or shorter than the convention, relative to how "
            "wide-set the eyes are. Descriptive only."
        ),
        module="analyze_harmony.py",
        code_location="analyze_harmony.py:148-151",
        bands=("mildly off at |dev| ≥0.08", "off at ≥0.18"),
        note="Same quantity as ext.midface_ratio (0.95-1.10); centre 1.0 is consistent.",
    ),
    _r(
        key="harmony.nose_mouth_eye",
        metric="Nose–mouth–eye proportion — nose/eye width and mouth/nose width combined",
        value=(1.0, 1.618),
        units="ratio (nose/eye centre, mouth/nose centre)",
        anchor_kind="range",
        tradition="neoclassical equal-widths canon combined with φ",
        evidence_tier=TIER_ARTISTIC,
        source=UNSOURCED,
        what_deviation_means=(
            "The central features sit out of the drawing convention's proportion. "
            + _PHI_PROVENANCE
            + " The two sub-deviations are combined by taking "
            "the worse of them after rescaling the φ term by 1.618 "
            "(analyze_harmony.py:181) — a commensurability choice with no "
            "recorded justification."
        ),
        module="analyze_harmony.py",
        code_location="analyze_harmony.py:176-182",
        bands=("mildly off at combined ≥0.25", "off at ≥0.55"),
    ),
    _r(
        key="harmony.lip_balance",
        metric="Lip balance — upper : lower vermilion height",
        value=0.62,
        units="ratio (balance centre; lower modestly fuller)",
        anchor_kind="point",
        tradition="conventional aesthetic target",
        evidence_tier=TIER_AUTHOR,
        source=UNSOURCED,
        what_deviation_means=(
            "A relatively fuller upper (higher) or lower (lower) lip than the "
            "convention. Two-directional by construction; neither direction is "
            "framed as better."
        ),
        module="analyze_harmony.py",
        code_location="analyze_harmony.py:202-205",
        bands=("mildly off at |dev| ≥0.12", "off at ≥0.25"),
        note=(
            "0.62 is the approximate midpoint of ext.lip_ratio's 0.55-0.72 band — "
            "consistent, but derived from an unsourced band."
        ),
    ),
    _r(
        key="harmony.lower_face_chin",
        metric="Lower-face / chin balance — philtrum:lip-chin split with mentolabial fold",
        value=(0.5, 120.0),
        units="ratio (split centre), degrees (mentolabial centre)",
        anchor_kind="range",
        tradition="neoclassical 1:2 lower-third convention, applied to a PROXY",
        evidence_tier=TIER_ARTISTIC,
        source=UNSOURCED,
        what_deviation_means=(
            "The philtrum sits long or short relative to the lip-to-chin span, on "
            "a measurement the classical convention does not actually cover. "
            "analyze_harmony.py:228-243 documents this openly: the classical "
            "division runs subnasale→stomion→menton, whereas this measures "
            "subnasale→labrale superius and labrale inferius→menton, excluding "
            "the vermilion from both. So 0.5 is a convention BORROWED onto a "
            "nearby quantity and used only as a descriptive centre."
        ),
        module="analyze_harmony.py",
        code_location="analyze_harmony.py:244, 250-253",
        bands=("mildly off at combined ≥0.12", "off at ≥0.28"),
        note=(
            "DELIBERATE ASYMMETRY WITH PART A: the identical measurement is "
            "reported with NO canon in analyze_frontal.py "
            "(phi.lower_face_philtrum_over_lip_chin). That is the resolution of a "
            "three-way contradiction (1.618 / 0.5 / a comment claiming 1.0) "
            "recorded in README.md's \"Corrections this project has made to "
            'itself" — not an oversight to unify.'
        ),
    ),
    _r(
        key="harmony.profile_balance",
        metric="Profile balance — total convexity distance from band midpoint, with chin projection",
        value=(132.5, 0.0),
        units="degrees (convexity centre), % of face height (chin centre)",
        anchor_kind="range",
        tradition="cephalometric convention, recentred by this pipeline",
        evidence_tier=TIER_AUTHOR,
        source=UNSOURCED,
        what_deviation_means=(
            "The profile carries convexity, or the chin sits off neutral. The "
            "132.5° centre is simply the midpoint of ext.total_convexity's "
            "125-140° band. The two inputs are combined by normalising degrees by "
            "9.0 and percent by 4.0 so each reaches 'mild' at ~1.0 "
            "(analyze_harmony.py:282-283) — an author-chosen commensurability "
            "constant, unsourced."
        ),
        module="analyze_harmony.py",
        code_location="analyze_harmony.py:277-284",
        bands=("mildly off at combined ≥1.0", "off at ≥2.2"),
    ),
    _r(
        key="harmony.left_right_balance",
        metric="Left–right balance — mean bilateral landmark deviation",
        value=0.0,
        units="% of interpupillary distance (0 = perfect mirror)",
        anchor_kind="point",
        tradition="relational restatement of Part C symmetry",
        evidence_tier=TIER_AUTHOR,
        source=UNSOURCED,
        what_deviation_means=(
            "Larger side-to-side deviation. The centre of 0 is ASYMPTOTIC — see "
            "symmetry.no_ideal: no real face reaches it, so this axis can only "
            "ever move away from its own balance point. The mild/off cutoffs (3% "
            "and 6% of IPD) are project choices with no recorded population "
            "reference for what a typical face scores."
        ),
        module="analyze_harmony.py",
        code_location="analyze_harmony.py:305-306",
        bands=("mildly off at ≥3.0% IPD", "off at ≥6.0% IPD"),
    ),
    # -----------------------------------------------------------------------
    # analyze_advanced.py — the SOURCED dimorphism anchors (the template)
    # -----------------------------------------------------------------------
    _r(
        key="dimorphism.fwhr",
        metric="fWHR (bizygomatic width / upper-face height) — sex-typicality anchor",
        value=(1.80, 2.00),
        units="ratio (neutral band; higher = masculine direction)",
        anchor_kind="range",
        tradition="sexual-dimorphism literature",
        evidence_tier=TIER_ANTHROPOMETRIC,
        source=(
            "Kramer 2017 meta-analysis (32 samples, N>10k, d~0.11, fWHR-BMI "
            "r~0.31); Lefevre/Wilson commentary 2018; Kramer & Mc Auley 2025 "
            "Bayesian re-analysis (no robust sex difference once capture variance "
            "is modelled); Ozener 2012 (no dimorphism in white European faces, "
            "2D/3D/anthropometry). NOTE: the robust dimorphism is in bizygomatic "
            "WIDTH (d~1.07-1.39 after allometry; Whitehouse/Hodges-Simeon 2022), "
            "not the width-to-HEIGHT ratio. [recorded verbatim in "
            "analyze_advanced.py:234-241]"
        ),
        what_deviation_means=(
            "Near-nothing about sex. Evidence strength recorded as "
            "'weak/contested': the sex difference in fWHR is tiny (d~0.11), "
            "disputed, and confounded by BMI/adiposity rather than bone. Some "
            "samples find no difference or a reversal. " + _ADVANCED_ANCHOR_CAVEAT
        ),
        module="analyze_advanced.py",
        code_location="analyze_advanced.py:225-242",
        note="COLLISION: ext.fwhr uses 1.75-2.05 for the same metric, unsourced.",
    ),
    _r(
        key="dimorphism.jaw_bizygomatic",
        metric="Jaw width / bizygomatic width — sex-typicality anchor",
        value=(0.74, 0.82),
        units="ratio (neutral band; higher = masculine direction)",
        anchor_kind="range",
        tradition="sexual-dimorphism literature",
        evidence_tier=TIER_ANTHROPOMETRIC,
        source=(
            "Bigonial width strongly dimorphic: males ~217mm vs females ~193mm "
            "(orthopantomography, Indira 2024); facial-width/bigonial ratio ~1.12 "
            "male vs ~1.18 female -> bigonial/bizygomatic ~0.89 vs ~0.85 (AI "
            "celebrity morphometry, Knoops/Borsato 2023). Different landmark set "
            "than FaceMesh. [recorded verbatim in analyze_advanced.py:253-257]"
        ),
        what_deviation_means=(
            "A wide, square lower face reads masculine; female faces taper more "
            "from cheekbone to jaw. Evidence strength recorded as 'robust "
            "(direction); descriptive (this metric's anchor)' — the DIRECTION is "
            "well supported (bigonial width d~1+) but this pipeline's mesh 'jaw "
            "width' is not true bigonial width, so the numeric band is an "
            "orientation range only. " + _ADVANCED_ANCHOR_CAVEAT
        ),
        module="analyze_advanced.py",
        code_location="analyze_advanced.py:243-258",
        note="COLLISION: ext.jaw_over_cheekbone uses 0.70-0.80, unsourced.",
    ),
    _r(
        key="dimorphism.lower_third_share",
        metric="Lower-third share of face height — sex-typicality anchor",
        value=(0.34, 0.36),
        units="fraction of total face height (higher = masculine direction)",
        anchor_kind="range",
        tradition="sexual-dimorphism literature (Farkas-derived proportion indices)",
        evidence_tier=TIER_ANTHROPOMETRIC,
        source=(
            "Farkas/Munro proportion indices: males larger on lower-face/"
            "total-face (Sn-Gn/N-Gn) and mandibulo-face indices; females larger on "
            "upper-face/total-face (N-Sto/N-Gn) (Ngeow & Aljunid 2009, Nigerian "
            "young-adult soft-tissue study after Farkas; 3D Facial Norms DB, "
            "Weinberg 2016). [recorded verbatim in analyze_advanced.py:267-272]"
        ),
        what_deviation_means=(
            "A relatively longer lower face is masculine; females carry relatively "
            "more of total face height in the UPPER third. Evidence strength "
            "recorded as 'moderate-robust' — one of the more reliably dimorphic "
            "vertical proportions. " + _ADVANCED_ANCHOR_CAVEAT
        ),
        module="analyze_advanced.py",
        code_location="analyze_advanced.py:259-273",
        note=(
            "Reads the SAME lower-third percentage that thirds.equal_share grades "
            "against an unsourced 33.333% artistic canon. Same number, two tiers: "
            "here it has a sourced anthropometric anchor, there it does not."
        ),
    ),
    _r(
        key="dimorphism.nose_bizygomatic",
        metric="Nose width / bizygomatic width — sex-typicality anchor",
        value=(0.24, 0.26),
        units="ratio (higher = masculine direction)",
        anchor_kind="range",
        tradition="sexual-dimorphism literature",
        evidence_tier=TIER_ANTHROPOMETRIC,
        source=(
            "Alar width ~44.4mm male vs ~39.3mm female (p<0.001); nasal breadth "
            "~41.2mm vs ~36.8mm (p<0.05) (Turkish & multi-population "
            "anthropometry, Ozdemir 2015 and reviews after Farkas 2005). All "
            "linear nasal dimensions larger in males. [recorded verbatim in "
            "analyze_advanced.py:283-287]"
        ),
        what_deviation_means=(
            "A wider nose reads masculine. Evidence strength recorded as 'robust "
            "(absolute width); moderate (this ratio)' — absolute nasal width is "
            "consistently dimorphic across populations, but the ratio is weaker "
            "because the bizygomatic denominator is itself dimorphic. "
            + _ADVANCED_ANCHOR_CAVEAT
        ),
        module="analyze_advanced.py",
        code_location="analyze_advanced.py:274-288",
    ),
    _r(
        key="dimorphism.lip_height_over_mouth_width",
        metric="Lip height / mouth width — sex-typicality anchor",
        value=(0.40, 0.46),
        units="ratio (LOWER = masculine direction)",
        anchor_kind="range",
        tradition="sexual-dimorphism literature",
        evidence_tier=TIER_ANTHROPOMETRIC,
        source=(
            "Increased vermilion height associated with femininity; attractive "
            "Caucasian female upper/lower vermilion ~6.5/11.6mm (Penna 2024, "
            "ideal-lips anthropometry); women smaller mouth + fuller lips (Qoves "
            "dimorphism review of 3D-morphology literature; Velemínská 2015 3D "
            "analysis). [recorded verbatim in analyze_advanced.py:299-303]"
        ),
        what_deviation_means=(
            "Fuller lips relative to mouth width read feminine. Evidence strength "
            "recorded as 'moderate'; lip fullness also thins markedly with age, "
            "confounding a fixed read. " + _ADVANCED_ANCHOR_CAVEAT
        ),
        module="analyze_advanced.py",
        code_location="analyze_advanced.py:289-304",
        note=(
            "One of the two cited works ('Qoves dimorphism review') is a "
            "commercial review of the literature rather than a primary study — "
            "recorded as-is from the codebase, not upgraded."
        ),
    ),
    _r(
        key="dimorphism.brow_lid_gap",
        metric="Brow-to-eyelid gap / eye width — sex-typicality anchor",
        value=(0.50, 0.60),
        units="ratio (LOWER = masculine direction)",
        anchor_kind="range",
        tradition="sexual-dimorphism literature",
        evidence_tier=TIER_ANTHROPOMETRIC,
        source=(
            "Brow ridge robustly more prominent in males (testosterone-driven; "
            "Godinho/Spikins 2018); 3D analysis: brow top-edge height ~equal "
            "between sexes, female brows thinner -> larger apparent eye-brow gap "
            "(Gao 2023 3D eyebrow/eyelid morphology, Caucasian & Chinese; "
            "virtualffs FFS thesis synthesis). [recorded verbatim in "
            "analyze_advanced.py:318-322]"
        ),
        what_deviation_means=(
            "A larger visible brow-to-lid gap reads feminine, mostly via brow-"
            "RIDGE prominence. Evidence strength recorded as 'moderate (ridge); "
            "weak (eyebrow position) / confounded by grooming' — the popular "
            "'women have much higher brows' claim is largely an artefact of "
            "THINNER female brows, and eyebrow grooming confounds the measure "
            "directly. " + _ADVANCED_ANCHOR_CAVEAT
        ),
        module="analyze_advanced.py",
        code_location="analyze_advanced.py:305-323",
    ),
    _r(
        key="dimorphism.eye_width_bizygomatic",
        metric="Eye width / bizygomatic width — sex-typicality anchor",
        value=(0.19, 0.22),
        units="ratio (LOWER = masculine direction)",
        anchor_kind="range",
        tradition="sexual-dimorphism literature",
        evidence_tier=TIER_ANTHROPOMETRIC,
        source=(
            "Females greater vertical eye-fissure height and higher eye "
            "height/width ratios contributing to feminine appearance (Qoves "
            "dimorphism review; Velemínská 2015 3D facial morphology). "
            "Bizygomatic denominator is itself strongly dimorphic, diluting the "
            "signal. [recorded verbatim in analyze_advanced.py:333-338]"
        ),
        what_deviation_means=(
            "Relatively larger eyes read feminine. Evidence strength recorded as "
            "'moderate (eye-fissure height); weak (this width ratio)' — the "
            "dimorphic eye measure is fissure HEIGHT, which this pipeline does "
            "not use here; width over a dimorphic denominator is an indirect cue. "
            + _ADVANCED_ANCHOR_CAVEAT
        ),
        module="analyze_advanced.py",
        code_location="analyze_advanced.py:324-339",
    ),
    _r(
        key="dimorphism.point_anchor_band",
        metric="Neutral band synthesised around a POINT dimorphism anchor",
        value=(0.96, 1.04),
        units="× the anchor (±4%)",
        anchor_kind="threshold_ladder",
        tradition="this pipeline's classification logic",
        evidence_tier=TIER_AUTHOR,
        source=UNSOURCED,
        what_deviation_means=(
            "Nothing on the current data — all seven anchors above are ranges, so "
            "this ±4% fallback path is never exercised. It is catalogued because "
            "it would silently become load-bearing the moment anyone adds a "
            "point-valued anchor, and ±4% has no recorded derivation."
        ),
        module="analyze_advanced.py",
        code_location="analyze_advanced.py:348, 363-364",
    ),
    # -----------------------------------------------------------------------
    # The rows where NO ideal exists — the most valuable rows in the table
    # -----------------------------------------------------------------------
    _r(
        key="symmetry.no_ideal",
        metric="Bilateral symmetry (roll-corrected, normalised to IPD)",
        value=None,
        units="% of interpupillary distance",
        anchor_kind="none",
        tradition="none — the ideal is asymptotic and unreachable",
        evidence_tier=TIER_NONE,
        source=(
            UNSOURCED + " for the symmetry–attractiveness link specifically. "
            "analyze_advanced.py:3-5 asserts at construct level that symmetry "
            '(with dimorphism and averageness) has "actual support in the '
            'attractiveness literature", but names no study for symmetry. The '
            "citations recorded in README.md Credits cover FairFace, Oosterhof & "
            "Todorov, Zebrowitz and Langlois & Roggman — none of them a symmetry "
            "reference."
        ),
        what_deviation_means=(
            "THERE IS NO TARGET TO HIT. Perfect symmetry (0% deviation) is "
            "asymptotic: no real face reaches it, so every face scores above zero "
            "and 'closer to zero' is a description of this face, not progress "
            "toward an achievable ideal. Reporting a percentage against an "
            "unreachable point invites the reader to treat the gap as a shortfall. "
            "It is not one. Downstream, harmony.left_right_balance does put "
            "author-chosen mild/off cutoffs on this number — those cutoffs are "
            "the only thing resembling a threshold, and they are unsourced."
        ),
        module="analyze_advanced.py",
        code_location="analyze_advanced.py:128-186 (symmetry)",
    ),
    _r(
        key="symmetry.display_saturation",
        metric="Symmetry-map colour saturation point",
        value=8.0,
        units="% of IPD (asymmetry at which a pair renders fully red)",
        anchor_kind="threshold_ladder",
        tradition="this pipeline's visualisation scaling",
        evidence_tier=TIER_AUTHOR,
        source=UNSOURCED,
        what_deviation_means=(
            "Only how red a line is drawn. It is a rendering constant, not a "
            "finding — but it sets the visual impression of severity in "
            "symmetry_map.png, so a reader can absorb an unsourced severity scale "
            "without ever seeing a number."
        ),
        module="analyze_advanced.py",
        code_location="analyze_advanced.py:467",
    ),
    _r(
        key="averageness.percentile_no_target",
        metric="Averageness — Procrustes closeness to a matched population mean",
        value=None,
        units="percentile against a matched reference population",
        anchor_kind="none",
        tradition="attractiveness literature (averageness construct)",
        evidence_tier=TIER_NONE,
        source=(
            "Construct definition: Langlois & Roggman 1990 and successors, "
            "recorded in analyze_averageness2.py:3-4 and README.md Credits. "
            "Reference population: FairFace (Karkkainen & Joo, 2021, CC-BY-4.0), "
            "README.md Credits."
        ),
        what_deviation_means=(
            "THERE IS NO TARGET TO HIT. The output is a PERCENTILE against a "
            "demographically matched, pose-gated reference population — a "
            "position in a distribution, not a distance from an ideal. A "
            "percentile cannot be 'met' or 'missed'; it can only be reported with "
            "its interval. This pipeline reports a bootstrap 95% CI for exactly "
            "that reason. Two further limits are recorded: the effect is weak "
            "(r~.3-.5, analyze_averageness2.py:40), and most of the real-world "
            "averageness effect comes from average SKIN AND COLOURING, which this "
            "shape-only measure captures nothing of (build_report.py:608). "
            "Because it is population-relative, changing the reference "
            "demographic changes the answer — which is why the reference set is "
            "matched rather than universal."
        ),
        module="analyze_averageness2.py",
        code_location="analyze_averageness2.py:269-296",
    ),
    _r(
        key="averageness.template_disparity_no_target",
        metric="Template averageness — Procrustes disparity vs the MediaPipe canonical face model",
        value=None,
        units="Procrustes disparity (dimensionless)",
        anchor_kind="none",
        tradition="none — a raw distance to a modelling artefact",
        evidence_tier=TIER_NONE,
        source=(
            UNSOURCED
            + ". The canonical_face_model.obj is MediaPipe's modelling mesh; no "
            "source in this project characterises it as a population average of "
            "anything, and no reference disparity value is recorded."
        ),
        what_deviation_means=(
            "THERE IS NO TARGET, and unlike averageness.percentile_no_target "
            "there is not even a population to be a percentile within. The number "
            "is a distance to a software template. It is comparable across runs "
            "of this pipeline and to nothing else. The literature-faithful "
            "averageness figure the report headlines comes from "
            "analyze_averageness2.py, not from here."
        ),
        module="analyze_advanced.py",
        code_location="analyze_advanced.py:428-451",
    ),
    _r(
        key="averageness.pose_gate",
        metric="Pose gate for admitting a reference face to the averageness population",
        value=(14.0, 16.0, 25.0),
        units="degrees (|yaw| ≤14, |pitch| ≤16, |roll| ≤25)",
        anchor_kind="threshold_ladder",
        tradition="this pipeline's data-quality gate",
        evidence_tier=TIER_AUTHOR,
        source=UNSOURCED,
        what_deviation_means=(
            "Not a facial judgement — an inclusion criterion. It decides WHICH "
            "reference faces the subject's percentile is computed against, so the "
            "cutoffs shape the answer without appearing in it. Tightening them "
            'shrinks an already small reference (README.md\'s "Limitations" records '
            "825 faces kept across eight cohorts from 2,181 landmarked) and widens "
            "the CI; loosening them admits pose variance as if it were shape. The "
            "three values are unsourced and asymmetric across axes with no "
            "recorded reason."
        ),
        module="analyze_averageness2.py",
        code_location="analyze_averageness2.py:151-152 (is_frontal)",
    ),
    # -----------------------------------------------------------------------
    # analyze_skin.py
    # -----------------------------------------------------------------------
    _r(
        key="skin.iga_lesion_ladder",
        metric="IGA-style grade from focal inflammatory-lesion count",
        value=(1, 4, 12, 28),
        units="inclusive upper bound on lesion count for grades 0,1,2,3 (above → 4)",
        anchor_kind="threshold_ladder",
        tradition=(
            "dermatology Investigator's Global Assessment — the 0-4 word labels "
            "are borrowed; the numeric cutoffs are this pipeline's"
        ),
        evidence_tier=TIER_AUTHOR,
        source=(
            UNSOURCED
            + '. analyze_skin.py:166-186 calls this an "IGA-style" ladder and an '
            '"explicit threshold table" and records why its logic was changed '
            "(a conjunctive gate capped single-axis severity at 2), but records no "
            "source for the four numbers."
        ),
        what_deviation_means=(
            "Which of five words (clear / almost clear / mild / moderate / severe) "
            "the automated lesion count is labelled with. Crossing 12 crosses the "
            "IGA ≥ 3 line that this module uses to tell a reader to see a "
            "dermatologist, so this cutoff carries real consequence from an "
            "unsourced literal. The count itself is an automated proxy from one "
            "uncalibrated 2D photo, NOT a clinician's exam (analyze_skin.py:16-22)."
        ),
        module="analyze_skin.py",
        code_location="analyze_skin.py:184 (IGA_LESION_MAX)",
    ),
    _r(
        key="skin.iga_erythema_ladder",
        metric="IGA-style grade from erythema area",
        value=(2.0, 4.0, 9.0, 16.0),
        units="exclusive upper bound on % of the acne zone for grades 0,1,2,3",
        anchor_kind="threshold_ladder",
        tradition=("dermatology IGA word labels; area cutoffs are this pipeline's"),
        evidence_tier=TIER_AUTHOR,
        source=UNSOURCED,
        what_deviation_means=(
            "Which severity word the erythema area is labelled with. The reported "
            "grade is the WORSE of this and the lesion ladder, so either axis "
            "alone can escalate to 4 — a property the module asserts with a "
            "self-check (analyze_skin.py:203-245). Redness cannot be perfectly "
            "separated from shadow in an uncalibrated photo, so the input to this "
            "ladder is itself noisy."
        ),
        module="analyze_skin.py",
        code_location="analyze_skin.py:186 (IGA_ERYTHEMA_MAX)",
    ),
    _r(
        key="skin.iga_referral_threshold",
        metric="Escalation threshold — advise an in-person dermatologist",
        value=3,
        units="IGA-style grade (≥3, i.e. moderate+)",
        anchor_kind="threshold_ladder",
        tradition="clinical escalation convention for acne management",
        evidence_tier=TIER_CLINICAL,
        source=UNSOURCED,
        what_deviation_means=(
            "The single highest-consequence threshold in this pipeline: above it "
            "the module tells the reader to seek prescription escalation. It sits "
            "on top of two unsourced ladders fed by an automated proxy from one "
            "photo. The surrounding text does hedge correctly (also escalate on "
            "nodular/cystic lesions, any scarring, or no improvement in ~12 weeks; "
            '"Don\'t self-manage past mild"), which makes the numeric grade less '
            "load-bearing than it looks — but the number is still unsourced."
        ),
        module="analyze_skin.py",
        code_location="analyze_skin.py:483-488",
    ),
    _r(
        key="skin.ita_fitzpatrick_ladder",
        metric="Individual Typology Angle (ITA°) → Fitzpatrick phototype estimate",
        value=(55.0, 41.0, 28.0, 10.0, -30.0),
        units=(
            "ITA degrees, descending: >55 very light (I–II), >41 light (II), "
            ">28 intermediate (III), >10 tan (IV), >−30 brown (V), else dark (VI)"
        ),
        anchor_kind="threshold_ladder",
        tradition="dermatology colorimetry (ITA°) mapped to Fitzpatrick phototypes",
        evidence_tier=TIER_CLINICAL,
        source=(
            UNSOURCED + ". The ITA→phototype mapping is a recognised dermatology "
            "classification, but this project records no publication for these "
            "specific cutoffs — only its own caveat at analyze_skin.py:527-529 "
            "that the phototype is a 1-bucket estimate limited by uncalibrated "
            "white balance."
        ),
        what_deviation_means=(
            "Which tone bucket and phototype estimate the skin is assigned. It is "
            "a descriptive classification, not an aesthetic reference — no "
            "direction along this ladder is better. The module already declares "
            "roughly one bucket of uncertainty, so a value near any cutoff should "
            "be read as straddling two buckets."
        ),
        module="analyze_skin.py",
        code_location="analyze_skin.py:374-386",
    ),
    _r(
        key="skin.texture_index_bands",
        metric="Surface-texture label cutoffs",
        value=(2.4, 4.0),
        units="texture index (<2.4 smooth, <4.0 moderate, else textured / uneven)",
        anchor_kind="threshold_ladder",
        tradition="this pipeline's labelling",
        evidence_tier=TIER_AUTHOR,
        source=UNSOURCED,
        what_deviation_means=(
            "Which of three words the texture proxy is labelled with. The index "
            "is the 85th percentile of high-pass luminance energy in arbitrary "
            "units, so the cutoffs are only meaningful within this pipeline. "
            "analyze_skin.py:538-540 records that pores are near the resolution "
            "limit and this is a relative proxy, not a pore count."
        ),
        module="analyze_skin.py",
        code_location="analyze_skin.py:363-368",
    ),
    _r(
        key="skin.seborrhea_shine_bands",
        metric="Skin-type label cutoffs from T-zone specular shine",
        value=(6.0, 1.6, 2.0),
        units=(
            "T-zone shine % ≥6 with >1.6× cheek shine = oily(T-zone)/combination; "
            "≥6 = oily; ≥2 = normal–combination; else normal–dry"
        ),
        anchor_kind="threshold_ladder",
        tradition="this pipeline's labelling",
        evidence_tier=TIER_AUTHOR,
        source=UNSOURCED,
        what_deviation_means=(
            "Which skin-type word is assigned. Specular highlight fraction "
            "depends heavily on the lighting of the single photo, which is "
            "uncontrolled and not cross-polarised, so the same face under "
            "different light can cross these cutoffs."
        ),
        module="analyze_skin.py",
        code_location="analyze_skin.py:349-356",
    ),
    _r(
        key="skin.erythema_pixel_threshold",
        metric="Redness threshold for classifying a pixel as erythema/PIE",
        value=(3.5, 2.2),
        units="max(3.5 a* units, 2.2 × MAD of the a* residual)",
        anchor_kind="threshold_ladder",
        tradition="this pipeline's segmentation",
        evidence_tier=TIER_AUTHOR,
        source=UNSOURCED,
        what_deviation_means=(
            "Not a facial judgement — it decides which pixels count as red at "
            "all, and therefore feeds both skin.iga_erythema_ladder and the "
            "lesion count. An unsourced constant upstream of the referral "
            "threshold. The adaptive (MAD-scaled) half is self-referential, which "
            "is what makes white-balance error largely cancel."
        ),
        module="analyze_skin.py",
        code_location="analyze_skin.py:284-286",
    ),
    _r(
        key="skin.lesion_size_window",
        metric="Blob size and compactness window for counting a papule",
        value=(1.0, 4.5, 0.42),
        units="mm diameter min, mm diameter max, minimum fill ratio of bounding box",
        anchor_kind="threshold_ladder",
        tradition="this pipeline's blob filter",
        evidence_tier=TIER_AUTHOR,
        source=UNSOURCED,
        what_deviation_means=(
            "Which red blobs are counted as active lesions rather than flat PIE. "
            "The mm bounds are converted through scale.assumed_bizygomatic_mm, so "
            "the lesion count inherits that assumption's error, and the count "
            "feeds skin.iga_lesion_ladder and hence the referral threshold."
        ),
        module="analyze_skin.py",
        code_location="analyze_skin.py:398-408",
    ),
)


# ---------------------------------------------------------------------------
# Integrity checks — cheap, pure, and run at import so a malformed catalogue
# fails loudly instead of rendering a wrong provenance table.
# ---------------------------------------------------------------------------
def _validate(records: Sequence[Reference]) -> None:
    seen = set()
    for rec in records:
        if rec.key in seen:
            raise ValueError(f"duplicate reference key: {rec.key}")
        seen.add(rec.key)
        if not rec.what_deviation_means.strip():
            raise ValueError(f"{rec.key}: what_deviation_means must not be empty")
        if not rec.source.strip():
            raise ValueError(f"{rec.key}: source must not be empty")


_validate(CATALOGUE)

BY_KEY: Mapping[str, Reference] = MappingProxyType({r.key: r for r in CATALOGUE})


# ---------------------------------------------------------------------------
# Accessors. Every one returns records or subsets of records.
#
# There is deliberately NO helper that scores, counts, or rolls up how a face
# fares against these references. That would be a score by another name and
# would breach the report's founding refusal. Do not add one.
# ---------------------------------------------------------------------------
def get(key: str) -> Optional[Reference]:
    """Look up one reference by key, or None."""
    return BY_KEY.get(key)


def require(key: str) -> Reference:
    """Look up one reference by key; raise KeyError if absent."""
    return BY_KEY[key]


def keys() -> Tuple[str, ...]:
    return tuple(r.key for r in CATALOGUE)


def all_records() -> Tuple[Reference, ...]:
    return CATALOGUE


def iter_records() -> Iterator[Reference]:
    return iter(CATALOGUE)


def by_module(module: Optional[str] = None):
    """Records for one module, or a dict of module → records (catalogue order)."""
    if module is not None:
        return tuple(r for r in CATALOGUE if module in r.module)
    grouped: dict = {}
    for rec in CATALOGUE:
        grouped.setdefault(rec.module, []).append(rec)
    return {k: tuple(v) for k, v in grouped.items()}


def by_tier(tier: Optional[str] = None):
    """Records for one evidence tier, or a dict of tier → records in TIER_ORDER."""
    if tier is not None:
        if tier not in TIER_ORDER:
            raise ValueError(f"unknown evidence tier: {tier!r}")
        return tuple(r for r in CATALOGUE if r.evidence_tier == tier)
    return {
        t: tuple(r for r in CATALOGUE if r.evidence_tier == t)
        for t in TIER_ORDER
        if any(r.evidence_tier == t for r in CATALOGUE)
    }


def empty_tiers() -> Tuple[str, ...]:
    """Tiers in the vocabulary that no record occupies.

    Not decoration: TIER_ATTRACTIVENESS is currently empty, which says that not
    one reference VALUE in this pipeline is backed by the attractiveness
    literature. The three constructs that do have literature behind them
    (symmetry, dimorphism, averageness) either carry anthropometric anchors or
    carry no reference value at all. An empty tier is a finding.
    """
    occupied = {r.evidence_tier for r in CATALOGUE}
    return tuple(t for t in TIER_ORDER if t not in occupied)


def by_anchor_kind(kind: str) -> Tuple[Reference, ...]:
    if kind not in ANCHOR_KINDS:
        raise ValueError(f"unknown anchor_kind: {kind!r}")
    return tuple(r for r in CATALOGUE if r.anchor_kind == kind)


def sourced_records() -> Tuple[Reference, ...]:
    """Records with a citation traceable to something recorded in this project."""
    return tuple(r for r in CATALOGUE if r.is_sourced)


def unsourced_records() -> Tuple[Reference, ...]:
    """Records this project asserts with no citation recorded. Show these."""
    return tuple(r for r in CATALOGUE if r.provenance == "unsourced")


def no_source_required_records() -> Tuple[Reference, ...]:
    """Records that claim no reference value, so need no citation."""
    return tuple(r for r in CATALOGUE if r.provenance == "not applicable")


def records_without_reference() -> Tuple[Reference, ...]:
    """Rows where NO ideal exists. The most informative rows in the catalogue."""
    return tuple(r for r in CATALOGUE if not r.has_reference)


def records_not_emitted_to_json() -> Tuple[Reference, ...]:
    """References shown to the reader but absent from the pipeline's JSON output."""
    return tuple(r for r in CATALOGUE if not r.emitted_to_json)


__all__ = [
    "Reference",
    "CATALOGUE",
    "BY_KEY",
    "UNSOURCED",
    "NO_SOURCE_REQUIRED",
    "TIER_ARTISTIC",
    "TIER_CLINICAL",
    "TIER_ANTHROPOMETRIC",
    "TIER_ATTRACTIVENESS",
    "TIER_AUTHOR",
    "TIER_MEASUREMENT",
    "TIER_NONE",
    "TIER_ORDER",
    "ANCHOR_KINDS",
    "get",
    "require",
    "keys",
    "all_records",
    "iter_records",
    "by_module",
    "by_tier",
    "by_anchor_kind",
    "empty_tiers",
    "sourced_records",
    "unsourced_records",
    "no_source_required_records",
    "records_without_reference",
    "records_not_emitted_to_json",
]


if __name__ == "__main__":
    # Self-inspection dump. Deliberately prints provenance facts about the
    # CATALOGUE — never anything about a face.
    print("=" * 78)
    print("REFERENCE VALUE CATALOGUE — grouped by evidence tier")
    print("=" * 78)
    for tier, recs in by_tier().items():
        print(f"\n### {tier}  ({len(recs)} entries)")
        for r in recs:
            mark = {
                "cited": "CITED",
                "unsourced": "UNSOURCED",
                "not applicable": "N/A",
            }[r.provenance]
            print(f"  [{mark:>9}] {r.key}")
            print(f"        {r.metric}")
            print(f"        value: {r.value_text()}  ({r.anchor_kind})")
            if r.bands:
                print(f"        bands: {'; '.join(r.bands)}")
            print(f"        tradition: {r.tradition}")
            print(
                f"        source: {r.source[:150]}{'…' if len(r.source) > 150 else ''}"
            )
            print(
                f"        deviation means: {r.what_deviation_means[:150]}"
                f"{'…' if len(r.what_deviation_means) > 150 else ''}"
            )
            print(f"        used in: {r.code_location}")
            if not r.emitted_to_json:
                print("        !! NOT emitted to any JSON — annotation-only")
            print()

    print("=" * 78)
    print("PROVENANCE OF THE CATALOGUE ITSELF")
    print("=" * 78)
    print(f"  total reference entries   : {len(CATALOGUE)}")
    print(f"  CITED (real citation)     : {len(sourced_records())}")
    print(f"  UNSOURCED                 : {len(unsourced_records())}")
    print(f"  no source required (N/A)  : {len(no_source_required_records())}")
    print(f"  rows with NO ideal        : {len(records_without_reference())}")
    print(f"  shown but not in JSON     : {len(records_not_emitted_to_json())}")
    print()
    print("  cited keys:")
    for r in sourced_records():
        print(f"    - {r.key}")
    print()
    print("  rows where no ideal exists:")
    for r in records_without_reference():
        print(f"    - {r.key}")
    print()
    missing = records_not_emitted_to_json()
    if missing:
        print("  references shown to the reader but absent from JSON:")
        for r in missing:
            print(f"    - {r.key}  ({r.code_location})")
    else:
        print(
            "  references shown to the reader but absent from JSON: none\n"
            "    (three profile bands used to live only in annotation label strings;\n"
            "     they are now emitted to metrics_profile.json and are auditable)"
        )
    print()
    print("  tiers with ZERO entries (an empty tier is a finding, not a gap):")
    for t in empty_tiers():
        print(f"    - {t}")
