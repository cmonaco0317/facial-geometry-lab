#!/usr/bin/env python3
"""The one-page "how to shoot the set", DERIVED from the gate that enforces it.

WHY THIS IS GENERATED AND NOT WRITTEN
-------------------------------------
A hand-written instruction sheet is a second copy of the thresholds, and the
copy does not move when the gate does. This project has already been bitten by
exactly that: intake's spread floor was raised from 26 to 52 degrees on
2026-08-19 after a 38-degree set was found passing the gate and reading its
symmetry axis nearly 4x wrong. Any sheet written before that morning would still
be telling a friend that 26 degrees is enough, and it would look authoritative.

So every number below is READ FROM THE MODULE THAT ENFORCES IT. If someone moves
a threshold and does not regenerate this page, `selfcheck_shoot_guide.py` goes
red rather than the page going quietly stale.

⚠️ It states the thresholds, not a promise. Meeting all of them is what makes a
set ELIGIBLE; intake still measures the actual photographs and can refuse.
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import capture_check as CC  # noqa: E402
import intake as IN  # noqa: E402


def thresholds() -> dict:
    """Every number the guide prints, read from the code that enforces it."""
    cal = IN.load_calibration()
    return {
        # The shipped floor is the MAX of the simulated and measured-real floors,
        # so regenerating intake_calibration.json cannot lower it.
        "min_spread_deg": max(
            float(cal.get("min_spread_deg", IN.DEFAULT_MIN_SPREAD)),
            IN.DEFAULT_MIN_SPREAD,
        ),
        "min_views": int(cal.get("min_views", IN.DEFAULT_MIN_VIEWS) or IN.DEFAULT_MIN_VIEWS),
        "min_ipd_px": float(IN.MIN_IPD_PX),
        "profile_yaw_lo": float(CC.PROFILE_YAW_BAND[0]),
        "profile_yaw_hi": float(CC.PROFILE_YAW_BAND[1]),
        "dead_abs_yaw": float(IN.DEAD_ABS_YAW),
        # ⛔ THE AGE RANGE WAS HAND-TYPED HERE TOO, AND IT WENT FALSE. This page
        # told every reader "comparisons exist only for ages 10-29" from
        # 2026-08-23, when 42 matched cohorts covering 20-49 landed, until it was
        # found on 2026-08-25 -- the same sentence, in the same wrong state, as
        # the one in `run_pipeline.cohort_limits`. Fixing one copy and leaving
        # the other is how a retraction fails to propagate, so BOTH now read the
        # cohorts. Section 1 of this file's own docstring is the argument for it.
        "cohort_age_lo": _cohort_ages()[0],
        "cohort_age_hi": _cohort_ages()[1],
        "n_cohorts": _n_cohorts(),
    }


def _cohort_ages():
    """Lowest and highest age any matched cohort covers, read from the cohorts."""
    import bank_reference as RR

    # ⚠️ A WIDENED COHORT'S BAND READS "40-49 to 50-59", not "40-49".
    # partition("-") handed int() the string "49 to 50-59" and the whole shoot
    # guide raised. Every integer in the label is an age bound whatever the
    # label's shape, so read them all rather than assuming one separator --
    # a future band naming cannot break this the way the last one did.
    import re

    lo, hi = [], []
    open_ended = False
    for c in RR.COHORTS.values():
        age_band = c["age_band"]
        nums = [int(n) for n in re.findall(r"\d+", age_band)]
        if not nums:
            continue
        lo.append(min(nums))
        hi.append(max(nums))
        open_ended = open_ended or age_band.endswith("70-plus")
    if not lo:
        return None, None
    return min(lo), "70+" if open_ended else max(hi)


def _n_cohorts():
    import bank_reference as RR

    return len(RR.COHORTS)


def guide_text() -> str:
    t = thresholds()
    half = t["min_spread_deg"] / 2.0
    return f"""HOW TO SHOOT A SET FOR THIS SOFTWARE
{'=' * 44}
Every number here is read from the gate that enforces it, not typed out
separately, so this page cannot drift from what the software actually checks.

WHAT TO TAKE
  * At least {t['min_views']} photographs of the same face, same day if you can.
  * Turn your HEAD, not the camera, and keep turning past what feels like enough.
    The set needs {t['min_spread_deg']:.0f} degrees of total spread between the most left-facing
    and the most right-facing shot -- about {half:.0f} degrees each way from straight on.
    A row of near-frontal photos will be refused no matter how many there are:
    it is the ANGULAR SPREAD that makes the 3D fit possible, not the file count.
  * Include ONE three-quarter view turned {t['profile_yaw_lo']:.0f}-{t['profile_yaw_hi']:.0f} degrees.
    ⚠️ NOT a true 90-degree profile: past about {t['dead_abs_yaw']:.0f} degrees the far eye
    disappears, the landmark detector has nothing to anchor on, and the frame is
    dropped. "Turned, but both eyes still visible" is the target.

HOW TO LOOK
  * Relaxed, closed mouth. No smile, no pucker, no clenched jaw. Mouth posture
    measurably moves the typicality reading, which is why there is a gate on it.
  * Eyes open, looking at the lens.
  * Nothing across the face: no glasses, no hair over the brow, no hat.

LIGHT AND FRAMING
  * Even, indirect light on the whole face. Avoid hard shadows down one side and
    avoid strong colour casts -- a dim, green-cast room once made a fair-skinned
    subject read as dark-skinned, and the skin section had to refuse.
  * Fill the frame: head and shoulders. The eyes must be at least {t['min_ipd_px']:.0f} pixels
    apart or the photograph is dropped as too small.
  * Do not re-save, crop, filter or screenshot the photographs. Send the
    originals. A re-save can strip the orientation tag while leaving the pixels
    identical, and the face is then measured sideways.

IF YOU WOULD RATHER SEE IT THAN READ IT
  The angles are the hard part, and nobody estimates 62 degrees from a sentence.
  This writes a page with each shot drawn from above:

      ./venv/bin/python run_pipeline.py --shoot-page

BEFORE YOU SEND THEM
  Run the checker. It reports on a folder and writes nothing:

      ./venv/bin/python run_pipeline.py --check /path/to/your/photos

  It will tell you what is missing and which way to turn, so you can re-shoot
  while you are still set up.

WHAT THIS WILL AND WILL NOT ANSWER
  * Shape (proportions, symmetry, sex-typicality, typicality) transfers across
    cameras and rooms -- that is what the multi-photo fit is for.
  * Skin, colour and texture are measured on ONE photograph and do NOT transfer.
    They describe that photograph, not you.
  * Comparisons to other faces run against your own cohort -- your race, your
    sex and your age band -- and there are {t['n_cohorts']} of them, covering ages
    {t['cohort_age_lo']}-{t['cohort_age_hi']}. Outside that range the software refuses rather than
    comparing you to the wrong population. You set it when you run intake:
        ./venv/bin/python intake.py <photos> --age 35 --race Black --save-profile
"""


def main() -> int:
    print(guide_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
