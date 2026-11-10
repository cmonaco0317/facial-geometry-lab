"""Scale-free angles measured on the FITTED 3D shape, not on a silhouette.

WHAT IS ACTUALLY NEW HERE, STATED BEFORE ANYTHING ELSE
------------------------------------------------------
These are not new axes. The project already has all of them, and two of the
three are already REFUSED:

  * the jaw angle exists in `jawline_angle.py` as the SOFT-TISSUE jawline, and is
    refused: across four frames of one unchanging jaw in a single sitting it ran
    98.9-136.2 deg, covering almost the whole 100-140 scale such angles are
    judged on, and unlike facial convexity it does not track head turn
    (r = -0.29) so it cannot be corrected;
  * the cervicomental angle is already declined as not-well-defined -- it moved
    12.8 deg between two frames 1 deg apart, with no angle trend at all;
  * the nasolabial angle already ships, measured off the profile photograph.

So this module is not "three more measurements". It is the same measurements
taken on a better instrument, to find out whether the instrument was the reason
they failed. The fitted shape differs from a silhouette in the two ways that
matter: pose is solved for across several photographs rather than assumed, and
the fit carries a per-landmark posterior, so each angle can be given an error bar
derived from the actual disagreement between real photographs.

⛔ **WHETHER THEY SHIP IS NOT DECIDED HERE.** This module computes; a measurement
decides. `angle_reproducibility_probe.py` recomputes every angle on the fitted
shape of each photo subset of a 26-second burst -- one face that cannot have
changed -- and an angle whose spread there is as wide as the scale it is judged
on goes back to being refused, on a better instrument or not.

WHY ANGLES AND NOT MILLIMETRES
------------------------------
A millimetre from this pipeline is worth about +-6%: +-4.3% systematic from the
assumed 11.7mm iris, up to 4.6% within a session, 6.1% across sessions at matched
pose with all six lengths moving the same direction. That last one is the useful
part -- a COMMON-MODE scale error, which cancels EXACTLY in a ratio or an angle.
Absolute millimetres are the one thing the anchor gates; angles are free of it.

⚠️ That argument retires the SCALE objection and nothing else. It says these
angles are not blocked by the ruler. It does not say they reproduce, which is a
separate question and the one that killed the jaw angle the first time.

WHAT CANNOT BE BUILT HERE, AND WHY
----------------------------------
  * **Cervicomental angle** needs the neck. The 468-point mesh is a FACE mesh --
    its lowest contour is the mandibular border, and below that there are no
    vertices at all. There is nothing to measure the angle against. It stays on
    the silhouette, where it is already refused.
  * **Nasolabial FOLD depth** is a surface depression, not a landmark. A sparse
    mesh has no vertex in the fold. It needs dense depth (photometric stereo is
    the researched candidate) and is not obtainable from landmark correspondence.

Naming: this is the SOFT-TISSUE mandibular contour, never the skeletal gonial
angle -- that is read off a lateral cephalogram, and the two differ by however
much tissue is draped over the mandible, which is exactly the variable a jawline
reading is usually being consulted about.
"""

from __future__ import annotations

import os

import numpy as np

# Right-side mandibular chain, ear-side first, taken from the mesh's own face
# oval. The ramus and body are FITTED AS LINES rather than read as a three-point
# turn: a turn angle at one vertex is set by which two neighbours you pick, and
# hunting for the "corner" per subject is precisely the fragility that made the
# silhouette version swing 37 degrees on one sitting.
RAMUS_R = [454, 323, 361, 288]
BODY_R = [288, 397, 365, 379, 378, 400, 377, 152]
GONION_R = 288

# Midsagittal points for the nasolabial angle.
COLUMELLA = 94
SUBNASALE = 2
LABRALE_SUP = 0


def _direction(points: np.ndarray, away_from: np.ndarray) -> np.ndarray:
    """Unit direction of a point chain, oriented to point AWAY from a vertex.

    First principal axis rather than an endpoint difference: it uses every point
    in the segment, so a single mis-detected vertex moves it by a fraction of
    what it would move a two-point line.
    """
    centred = points - points.mean(axis=0)
    _u, _s, vt = np.linalg.svd(centred, full_matrices=False)
    d = vt[0]
    far = points[np.argmax(np.linalg.norm(points - away_from, axis=1))]
    if d @ (far - away_from) < 0:
        d = -d
    return d / np.linalg.norm(d)


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    c = float(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def mirror_indices(idx: list[int], perm: np.ndarray) -> list[int]:
    """The left-side counterparts of a right-side index list."""
    return [int(perm[i]) for i in idx]


def mandibular_contour_angle(shape3d: np.ndarray, perm: np.ndarray) -> dict:
    """Soft-tissue mandibular contour angle, per side, on a fitted 3D shape.

    The angle at the jaw corner between the ramus line (running up toward the
    ear) and the mandibular body line (running forward toward the chin). Measured
    in 3D on the fitted shape, so it is not a silhouette reading and does not
    depend on the subject having been photographed at a true lateral.

    Returns both sides. Two sides of one jaw is a free consistency check that a
    single silhouette cannot offer: they should be close on a roughly symmetric
    face, and a large left-right gap on a face whose asymmetry is known to be
    small is evidence about the METHOD rather than about the jaw.
    """
    out: dict = {}
    for side, ramus_idx, body_idx, gon in (
        ("right", RAMUS_R, BODY_R, GONION_R),
        (
            "left",
            mirror_indices(RAMUS_R, perm),
            mirror_indices(BODY_R, perm),
            int(perm[GONION_R]),
        ),
    ):
        g = shape3d[gon]
        ramus = _direction(shape3d[ramus_idx], g)
        body = _direction(shape3d[body_idx], g)
        out[side] = round(_angle_between(ramus, body), 2)
    out["mean"] = round((out["left"] + out["right"]) / 2.0, 2)
    out["left_right_gap"] = round(abs(out["left"] - out["right"]), 2)
    return out


def nasolabial_angle(shape3d: np.ndarray) -> float:
    """Angle at subnasale between the columella and the upper lip, in 3D.

    The shipped figure is read off the profile PHOTOGRAPH, so it inherits
    whatever that one frame's head position was. This one is measured on a shape
    fitted across several photographs with pose solved out.
    """
    s = shape3d[SUBNASALE]
    return round(_angle_between(shape3d[COLUMELLA] - s, shape3d[LABRALE_SUP] - s), 2)


# ---------------------------------------------------------------------------
# PROPORTION RATIOS — added 2026-08-12, and they are NOT new axes either
# ---------------------------------------------------------------------------
# Every ratio below already exists in this pipeline, measured on ONE photograph
# by analyze_frontal / analyze_extended. What is new is the instrument, exactly
# as it was for the jaw: measured here on the shape fitted across several
# photographs, so the reading is pose-free rather than a property of the frame
# that happened to be chosen, and each one inherits the fit's posterior and can
# therefore be given an error bar earned from the disagreement between real
# photographs.
#
# ⛔ WHETHER ANY OF THEM SHIP IS NOT DECIDED HERE. `angle_reproducibility_probe`
# recomputes them across photo subsets of a constant face and a ratio whose
# spread is a large fraction of the scale it is judged on stays refused, on a
# better instrument or not. That is what happened to the nasolabial angle.
#
# ⚠️ MEASURED IN THE FRONTAL PLANE (x,y) of the Procrustes-aligned shape, not as
# 3D Euclidean distances. A 3D chord across the cheeks is a different quantity
# from the projected width every facial canon is defined on, and quietly
# substituting one for the other would be the same class of error as the outer
# canthi standing in for the iris IPD -- a ruler swap that makes numbers move for
# free. The alignment is what makes this legitimate: the frontal plane of a
# pose-free shape is the view the canons always wanted and a single photograph
# cannot supply.
import fa_common as _FC  # noqa: E402


def _d2(shape3d: np.ndarray, i: int, j: int) -> float:
    """Frontal-plane distance between two landmarks of the aligned shape."""
    return float(np.linalg.norm(shape3d[i, :2] - shape3d[j, :2]))


def proportion_ratios(shape3d: np.ndarray, *, y_up: bool) -> dict:
    """Scale-free proportions. Every index comes from fa_common's vetted table."""
    brow_y = (shape3d[_FC.BROW_L, 1] + shape3d[_FC.BROW_R, 1]) / 2.0
    mid = abs(float(brow_y - shape3d[_FC.SUBNASALE, 1]))
    low = abs(float(shape3d[_FC.SUBNASALE, 1] - shape3d[_FC.MENTON, 1]))
    upper = abs(float(brow_y - shape3d[_FC.LABRALE_SUP, 1]))
    bizyg = _d2(shape3d, _FC.ZYGION_L, _FC.ZYGION_R)
    intercanthal = _d2(shape3d, _FC.L_CANTHUS_MED, _FC.R_CANTHUS_MED)
    biocular = _d2(shape3d, _FC.L_CANTHUS_LAT, _FC.R_CANTHUS_LAT)
    mouth_w = _d2(shape3d, _FC.MOUTH_L, _FC.MOUTH_R)
    nose_w = _d2(shape3d, _FC.ALA_L, _FC.ALA_R)

    # --- second batch, 2026-08-12. Same rule as the first: every index below is
    # already in fa_common's vetted table, because a guessed landmark index
    # produces a plausible number nothing can catch.
    bigonial = _d2(shape3d, _FC.JAW_L, _FC.JAW_R)
    full_face = abs(float(brow_y - shape3d[_FC.MENTON, 1]))
    eye_w_l = _d2(shape3d, _FC.L_CANTHUS_LAT, _FC.L_CANTHUS_MED)
    eye_w_r = _d2(shape3d, _FC.R_CANTHUS_LAT, _FC.R_CANTHUS_MED)
    eye_h_l = abs(float(shape3d[_FC.L_EYE_TOP, 1] - shape3d[_FC.L_EYE_BOT, 1]))
    eye_h_r = abs(float(shape3d[_FC.R_EYE_TOP, 1] - shape3d[_FC.R_EYE_BOT, 1]))

    def _safe(a: float, b: float) -> float:
        return float(a / b) if b > 1e-9 else float("nan")

    # --- third batch, 2026-08-12. Still vetted indices only.
    # ⛔ FACIAL CONVEXITY WAS THE OBVIOUS CANDIDATE AND IS DELIBERATELY ABSENT:
    # the clinical construct is glabella-subnasale-POGONION, and pogonion is the
    # most ANTERIOR chin point, which is not in fa_common's table. Substituting
    # menton (the LOWEST chin point) would silently measure a different angle and
    # then get compared against the silhouette's convexity figure -- the same
    # class of error as the outer canthi standing in for the iris IPD. It needs a
    # properly derived pogonion first.
    lip_h = abs(float(shape3d[_FC.LABRALE_SUP, 1] - shape3d[_FC.LABRALE_INF, 1]))
    chin_h = abs(float(shape3d[_FC.LABRALE_INF, 1] - shape3d[_FC.MENTON, 1]))

    # ⚠️⚠️ THE FIRST SIGNED Y-METRIC IN THIS MODULE, AND IT EXPOSED A GAP.
    # Every other ratio here takes abs() of its vertical distances, so all of them
    # are invariant to a vertical flip and none was ever at risk. Canthal tilt is
    # not: "lateral corner above the medial one" reverses if y does.
    #
    # The two inputs this function receives DISAGREE about y. A fitted shape is
    # Procrustes-aligned to the canonical model, which is Y-UP. The reference bank
    # is pixel coordinates from `p.y * h`, which is Y-DOWN. Computing a signed
    # tilt on both and comparing them would score the subject against a mirrored
    # population -- the same flip that hid an upside-down render for two sessions,
    # invisible precisely because bilateral symmetry survives it.
    #
    # So `y_up` is REQUIRED, with no default. A default would be silently wrong
    # for one of the two callers, and this is the class of error that does not
    # announce itself. The bank mean is the check: real faces average a small
    # POSITIVE tilt, so a negative bank mean means the convention is inverted.
    ysign = 1.0 if y_up else -1.0

    def _tilt(med: int, lat: int) -> float:
        d = shape3d[lat, :2] - shape3d[med, :2]
        return float(np.degrees(np.arctan2(ysign * d[1], abs(d[0]) + 1e-12)))

    tilt_l = _tilt(_FC.L_CANTHUS_MED, _FC.L_CANTHUS_LAT)
    tilt_r = _tilt(_FC.R_CANTHUS_MED, _FC.R_CANTHUS_LAT)

    return {
        "canthal_tilt_deg": (tilt_l + tilt_r) / 2.0,
        "chin_height_ratio": _safe(chin_h, low),
        "lip_face_width_ratio": _safe(mouth_w, bizyg),
        "lip_height_ratio": _safe(lip_h, mouth_w),
        "facial_thirds_ratio": _safe(mid, low),
        "fwhr": _safe(bizyg, upper),
        "intercanthal_ratio": _safe(intercanthal, biocular),
        "mouth_nose_width_ratio": _safe(mouth_w, nose_w),
        # jaw width against cheekbone width -- how much the face tapers. Distinct
        # from the mandibular ANGLE, which is the corner's sharpness rather than
        # the jaw's breadth; they can disagree and both are worth having.
        "jaw_taper_ratio": _safe(bigonial, bizyg),
        # lower third against the whole brow-to-chin height.
        "lower_face_height_ratio": _safe(low, full_face),
        # nose breadth against cheekbone breadth.
        "nasal_width_ratio": _safe(nose_w, bizyg),
        # eye opening height against its width, averaged over both eyes.
        "eye_aspect_ratio": _safe((eye_h_l + eye_h_r) / 2.0, (eye_w_l + eye_w_r) / 2.0),
    }


# Which ratios have earned a printed number, and which have not. ⛔ THIS IS A
# CROSS-SITTING VERDICT and must not be re-derived from a single sitting: the
# 26-second burst scores intercanthal at 16% of its scale, which would ship it,
# while three independent sittings score it at 28%, which refuses it. The burst
# has now understated three separate things (averageness 1.6-2.5x, the ratios
# ~2x, and this), so a within-sitting figure is a FLOOR, never a verdict.
# Percentages are the WORST of two bank scales: the p5-p95 span, and the
# interquartile range rescaled to a p5-p95 equivalent (x2.44). Carrying both
# stops a metric passing on a contaminated tail -- eye aspect ratio was checked
# for exactly that, since 6.6% of the bank has near-closed eyes and its raw
# minimum is 0.006, which would inflate the span with blinks rather than anatomy.
# It survived: the blinks sit below p5, which already excludes them.
RATIO_SHIPPABLE = {
    "facial_thirds_ratio": True,  # 4%
    "fwhr": True,  # 13%
    "mouth_nose_width_ratio": True,  # 22% -- closest to the line, see note
    "intercanthal_ratio": False,  # 27% -- REFUSED
    "jaw_taper_ratio": True,  # 13%
    "lower_face_height_ratio": True,  # 4%
    "nasal_width_ratio": True,  # 8%
    "eye_aspect_ratio": True,  # 4%
    # third batch. canthal tilt is the first SIGNED y-metric here; see the
    # y_up note in proportion_ratios for why its convention is explicit.
    "canthal_tilt_deg": True,  # 7%
    "chin_height_ratio": True,  # 3%
    "lip_face_width_ratio": True,  # 13%
    "lip_height_ratio": True,  # 4%
}

_BANK_CACHE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "ref_landmarks2.npz"
)


def bank_ratio_distribution() -> dict[str, np.ndarray]:
    """Every ratio computed across the reference bank, for norms and for scale.

    ⛔ LEGITIMATE ONLY BECAUSE RATIOS ARE NOT MAGNITUDES. Scoring a multi-view
    fitted subject against a bank of raw single photographs inflates any
    magnitude-valued measure, because independent noise can only ADD to a
    distance -- that is MULTIVIEW-WIRING 5.1, measured on 2026-08-11, and it is
    why averageness must never read the fit. A ratio is perturbed SYMMETRICALLY,
    so a cleaner subject carries no systematic advantage here. Do not reuse this
    function for anything magnitude-valued.
    """
    d = np.load(_BANK_CACHE)
    # ⛔ SELECT BY SHAPE, NOT BY SUFFIX. This filtered out `_raw` and took everything
    # else, which was correct only while those were the sole companion arrays. The
    # cache also carries `_race` provenance (1-D), so the first rebuild after that
    # feature landed made this raise inside a guard rather than in the code it
    # guards. A suffix denylist has to be updated every time the cache gains a
    # column; "the entries that are stacks of landmarks" cannot go stale.
    faces = np.concatenate(
        [d[k] for k in d.files if d[k].ndim == 3], axis=0
    )
    out: dict[str, list[float]] = {}
    for f in faces:
        for k, v in proportion_ratios(f, y_up=False).items():
            out.setdefault(k, []).append(v)
    return {
        k: np.asarray([x for x in v if np.isfinite(x)], float) for k, v in out.items()
    }


def bank_percentile(
    name: str, value: float, dist: dict[str, np.ndarray] | None = None
) -> float | None:
    """Where this subject sits among the reference faces, 0-100."""
    dist = dist if dist is not None else bank_ratio_distribution()
    arr = dist.get(name)
    if arr is None or arr.size == 0 or not np.isfinite(value):
        return None
    return float(100.0 * np.mean(arr < value))


# ---------------------------------------------------------------------------
# DERIVED PROFILE LANDMARKS — glabella and pogonion, 2026-08-12
# ---------------------------------------------------------------------------
# Facial convexity was deliberately skipped in the third batch because its
# clinical construct is glabella-subnasale-POGONION, and pogonion is the most
# ANTERIOR chin point, which is not an index in fa_common's table. Substituting
# menton (the LOWEST chin point) would have measured a different angle and then
# compared it against the silhouette's convexity figure -- the outer-canthi ruler
# swap again.
#
# ⭐ THE FIX IS TO DERIVE THEM THE WAY THE SILHOUETTE ALREADY DOES, rather than
# to guess an index. `analyze_profile` finds glabella as the most forward point
# of the outline between the forehead and the nasion -- an EXTREMUM within a
# vertical band, not a named vertex. The same rule works on the fitted mesh, and
# using the same rule is what makes the two figures comparable at all.
#
# Verified on the fitted shape rather than assumed: anterior is +z, y is UP
# (trichion sits 16 units above menton), and the midline is x = 0.
MIDLINE_FRAC = 0.06  # of face width; a band about the sagittal plane


def _midline_extremum(
    shape3d: np.ndarray, y_lo: float, y_hi: float, most_anterior: bool = True
) -> np.ndarray | None:
    """The most ANTERIOR (or most posterior) midline vertex inside a vertical band.

    Returns None rather than a fallback if the band catches nothing: a silently
    substituted point is exactly the failure this function exists to avoid.
    """
    half = MIDLINE_FRAC * float(np.ptp(shape3d[:, 0]))
    m = (
        (np.abs(shape3d[:, 0]) <= half)
        & (shape3d[:, 1] >= y_lo)
        & (shape3d[:, 1] <= y_hi)
    )
    if not m.any():
        return None
    idx = np.where(m)[0]
    pick = np.argmax if most_anterior else np.argmin
    return shape3d[idx[int(pick(shape3d[idx, 2]))]]


def profile_points(shape3d: np.ndarray) -> dict:
    """Glabella, sublabiale and pogonion, derived as extrema between vetted landmarks.

    🔴 POGONION WAS WRONG ON MOST FACES UNTIL 2026-08-12, AND THE CAUSE IS
    ANATOMY RATHER THAN CODE. It searched for the most anterior midline point
    between MENTON and LABRALE_INF -- but on **70% of the real scanned faces and
    65% of FLAME samples the lower lip genuinely protrudes further than the
    chin**, so that search returned the LIP. Measured: `pogonion` came out
    bitwise equal to `labrale_inf` on 28 of 40 real meshes. Anything built on it
    was quietly measuring lip position.
    ⚠️ It never showed up on the ONE subject it was verified against, which is
    how it survived: a derivation checked on a single face is checked against
    that face's anatomy, not against the range of human anatomy.
    ⛔ `facial_convexity` is built on this and stays REFUSED regardless -- it
    fails reproducibility for separate reasons -- but its recorded 5.86-degree
    swing was measured through the broken chain and should not be quoted as a
    property of the corrected one.

    ✅ THE FIX IS THE CLINICAL CHAIN, not a different guess. Pogonion is the most
    anterior point of the CHIN, which lies below the mentolabial sulcus. So the
    sulcus is derived first -- the most POSTERIOR midline point between the lower
    lip and menton -- and pogonion is then sought only BELOW it. That is how the
    landmark is defined anatomically, and it removes the lip from the search by
    construction rather than by a margin that would need tuning.
    """
    brow_y = (shape3d[_FC.BROW_L, 1] + shape3d[_FC.BROW_R, 1]) / 2.0
    menton_y = float(shape3d[_FC.MENTON, 1])
    lip_y = float(shape3d[_FC.LABRALE_INF, 1])
    # the mentolabial sulcus: the dip between lower lip and chin
    sublabiale = _midline_extremum(shape3d, menton_y, lip_y, most_anterior=False)
    sub_y = float(sublabiale[1]) if sublabiale is not None else lip_y
    return {
        # between brow level and the hairline
        "glabella": _midline_extremum(shape3d, brow_y, float(shape3d[_FC.TRICHION, 1])),
        "sublabiale": sublabiale,
        # between the chin bottom and the SULCUS — never up into the lip
        "pogonion": _midline_extremum(shape3d, menton_y, sub_y),
    }


# ⛔ MEASURED AND REFUSED, 2026-08-12. Built correctly and it still does not
# reproduce: across three sittings of one unchanging face it read 163.18 /
# 166.82 / 169.05, a range of 5.86 deg. That is 59% of the 10-degree-wide
# 165-175 "straight" band it is judged against, and the interval CROSSES 165 --
# so the same face is convex on one sitting and straight on another. The philtrum
# rule, for the fourth time: a reading that straddles its own cutoff cannot
# support the verdict printed beside it.
#
# ⭐ THE DERIVATION IS NOT WHAT FAILED. Pogonion lands 1.83 anterior to and above
# menton, glabella above brow level, and the value sits in the normal clinical
# range -- every anatomical check passes. Keep `profile_points`: it is correct,
# and any future sagittal metric needs it.
#
# 🔴 A STRUCTURAL LIMIT SURFACED HERE, and it bounds the whole profile family:
# the 243-face reference bank is 2D LANDMARKS ONLY, so it has no z and can supply
# NO population norm for any sagittal quantity. Frontal-plane ratios can be
# normed against it; profile angles cannot be, by construction. Qoves ships
# normed profile measurements (E-line, submental); this instrument cannot match
# that with this bank, whatever it does about reproducibility.
CONVEXITY_REFUSED = True


def facial_convexity(shape3d: np.ndarray) -> float:
    """Glabella-subnasale-pogonion, in the SAGITTAL plane.

    Measured on (y, z) rather than in full 3D, because the construct this is
    being compared against is read off a profile photograph and a profile
    photograph has no x. All three points sit near the midline anyway, so the two
    barely differ -- but "barely differ" is not the same as "is the same thing",
    and the whole reason this function exists is that a near-enough substitution
    was refused once already.
    """
    p = profile_points(shape3d)
    if p["glabella"] is None or p["pogonion"] is None:
        return float("nan")
    s = shape3d[_FC.SUBNASALE, 1:]
    a, b = p["glabella"][1:] - s, p["pogonion"][1:] - s
    c = float(
        np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12), -1.0, 1.0)
    )
    return float(np.degrees(np.arccos(c)))


def all_angles(shape3d: np.ndarray, perm: np.ndarray) -> dict:
    """Every scale-free quantity this module can measure on a fitted shape."""
    jaw = mandibular_contour_angle(shape3d, perm)
    return {
        "mandibular_contour_deg": jaw,
        "nasolabial_deg": nasolabial_angle(shape3d),
        "ratios": proportion_ratios(shape3d, y_up=True),
    }


def flatten(angles: dict) -> dict[str, float]:
    """One flat name -> value map, for spread bookkeeping."""
    jaw = angles["mandibular_contour_deg"]
    out = {
        "mandibular_left": jaw["left"],
        "mandibular_right": jaw["right"],
        "mandibular_mean": jaw["mean"],
        "nasolabial": angles["nasolabial_deg"],
    }
    out.update(angles.get("ratios", {}))
    return out
