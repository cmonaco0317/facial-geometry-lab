"""Who is being measured — read at run time, never hardcoded.

Several modules used to state the subject's age inline as a literal, in
docstrings, figure titles and report headings. That is a fact about one
particular person compiled into the source, which is wrong twice over: it is
personal data living in code meant to be published, and it silently mislabels the
output the moment the pipeline is pointed at anyone else.

This module reads an OPTIONAL `subject.json`. If it is absent, no demographic
claim is made anywhere — the report says "the subject" and omits the age rather
than inventing one. Absent configuration must produce silence, not a default
person.

    {
      "age": 20,
      "reference_set": "white_20_29",
      "pronoun": "they"
    }

`subject.json` is gitignored: it describes a real individual. `subject.example.json`
is the tracked template.
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(HERE, "subject.json")

# The averageness reference cohort used when nothing is configured. This is a
# COHORT NAME, not a claim about the subject — analyze_averageness2.py scores
# against all four sets regardless; this only picks which one the prose calls
# primary.
DEFAULT_REFERENCE_SET = "white_20_29"


def cohort_sex():
    """'male' | 'female' | None, from the configured reference bank.

    ⛔ NONE MEANS NONE, and callers must treat it as "say nothing" rather than as
    a default person -- the same rule the rest of this module follows. It exists
    because the report handed a female subject the advice "grow a defined beard"
    (found 2026-08-16 by running one end to end): the recommendation is not
    derived from anything measured, and this pipeline cannot see whether a face
    can grow a beard at all. The bank names are the only sex information anywhere
    in the configuration, so they are what the report is allowed to use.
    """
    prof = load()
    # ⛔ AN UNCONFIGURED SUBJECT HAS NO SEX, and until 2026-08-19 this returned
    # "male" for one. `reference_set` falls back to DEFAULT_REFERENCE_SET so the
    # scoring code always has a bank to index, and this function then read the
    # SEX off that fallback -- so a subject nobody had described came out male,
    # and so did a subject described as {"age": 24, "pronoun": "she"} with the
    # cohort simply left out.
    #
    # That is the beard defect's exact route, still open: the advice gate asks
    # this function, and it was answering "male" on the strength of a default
    # that subject.example.json promises is not a claim about anybody -- "omit
    # one and the report simply makes no claim about it rather than guessing".
    # Found while preparing FINISH-LINE Step 3, whose ten strangers carry no
    # sex/age/race labels at all, which is the same state a friend's first run is
    # in.
    if not prof["reference_set_configured"]:
        return None
    ref = prof["reference_set"]
    if ref.startswith("female"):
        return "female"
    if ref.startswith(("white", "broad")):
        return "male"
    return None


# The seven FairFace race groups, spelled as the dataset spells them and verified
# against its own schema on 2026-08-23 rather than assumed:
#   0 East Asian  1 Indian  2 Black  3 White  4 Middle Eastern
#   5 Latino_Hispanic  6 Southeast Asian
# ⛔ THIS IS A COMPARISON-POPULATION LABEL, NOT AN IDENTITY. It selects which bank
# of real faces the percentiles are computed in. Picking the wrong one does not
# widen the answer -- it answers a different question, the same trap the
# reference_set comment already warns about.
RACE_NAMES = ["East Asian", "Indian", "Black", "White",
              "Middle Eastern", "Latino_Hispanic", "Southeast Asian"]
RACE_SLUGS = {r.lower().replace(" ", "_"): r for r in RACE_NAMES}

# Race-matched banks, by the age they cover. A subject outside every band gets
# NO matched cohort -- comparing a 60-year-old against 40-49 faces is not a wider
# answer, it is a different question, which is the rule the reference_set comment
# already states and the reason these bands are enumerated rather than clamped.
RACE_COHORT_BANDS = [(20, 29, "20_29"), (30, 39, "30_39"), (40, 49, "40_49")]
RACE_COHORT_AGES = (RACE_COHORT_BANDS[0][0], RACE_COHORT_BANDS[-1][1])


def race():
    """The subject's configured race group as a slug, or None.

    ⛔ NONE MEANS NONE. An unconfigured race must produce no race-matched claim
    at all rather than falling back to White -- which is exactly the defect this
    whole cohort layer exists to close, and the same shape as the sex default
    that once handed a woman 'grow a defined beard'. An unrecognised value is
    also None: a typo must not silently become a different population.
    """
    prof = load()
    raw = (prof.get("race") or "").strip().lower().replace(" ", "_")
    if raw in RACE_SLUGS:
        return raw
    # ⭐ THE LEGACY BANK NAMES ALREADY CARRY A RACE, so read it rather than making
    # every existing subject reconfigure. `white_20_29` and `female_20_29` are
    # literally FairFace race=3 pulls (see fetch_reference2.SETS), so a subject
    # on one of them IS White-matched and should get the matched comparison.
    # ⛔ `broad_*` is all-race POOLED and carries no race, so it stays None --
    # inferring one there would invent the very matching this layer exists to
    # make honest. And an unconfigured bank infers nothing at all: "resolved"
    # and "chosen" are different facts, the same rule cohort_sex() follows.
    if not prof["reference_set_configured"]:
        return None
    ref = prof["reference_set"]
    if ref.startswith("broad") or ref.startswith("female_broad"):
        return None
    if ref.startswith(("white", "female")):
        return "white"
    return None


def race_cohort():
    """The race+sex+age-matched bank id for this subject, or None with a reason.

    Returns (cohort_id, None) when a matched bank exists, else (None, reason).
    The reason is written for a READER of the report, because the report has to
    say why it is comparing them against a wider population than it would like.
    """
    r = race()
    if r is None:
        return None, ("no race group is configured, so no race-matched "
                      "comparison population can be chosen")
    sex = cohort_sex()
    if sex is None:
        return None, ("no reference cohort is configured, so the sex of the "
                      "comparison population is unknown")
    age = load().get("age")
    lo, hi = RACE_COHORT_AGES
    if age is None:
        return None, (f"no age is configured, and race-matched banks exist only "
                      f"for ages {lo}-{hi}")
    band = next((b for a0, a1, b in RACE_COHORT_BANDS if a0 <= age <= a1), None)
    if band is None:
        return None, (f"race-matched reference banks exist only for ages {lo}-{hi} "
                      f"and this subject is {age}; comparing against a "
                      f"mismatched age band would answer a different question")
    return f"{'m' if sex == 'male' else 'f'}_{r}_{band}", None


def load():
    """Return the subject profile. Missing file / bad JSON -> no claims."""
    empty = {
        "age": None,
        "race": None,
        "reference_set": DEFAULT_REFERENCE_SET,
        # WHETHER A HUMAN ACTUALLY CHOSE THAT BANK. The bank name always resolves
        # so the scoring code has something to index, but "resolved" and "chosen"
        # are different facts and conflating them is what let an undescribed
        # subject be treated as male. Every consumer that makes a CLAIM about the
        # person -- the advice gate, the cohort label in the report -- must read
        # this, not the bank name.
        "reference_set_configured": False,
        "pronoun": "they",
        "is_validation_subject": False,
    }
    if not os.path.exists(CONFIG):
        return empty
    try:
        with open(CONFIG, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        # A malformed profile must not silently become a confident default.
        return empty
    return {
        "age": data.get("age"),
        "race": data.get("race"),
        "reference_set": data.get("reference_set") or DEFAULT_REFERENCE_SET,
        "reference_set_configured": bool(data.get("reference_set")),
        "pronoun": data.get("pronoun") or "they",
        # Defaults to FALSE on purpose. See validation_face().
        "is_validation_subject": bool(data.get("is_validation_subject", False)),
    }


def validation_face(cap: bool = False) -> str:
    """How the report should refer to the face the INSTRUMENT was measured on.

    Much of this report's honesty apparatus rests on facts established by
    photographing one person repeatedly -- the 26-second burst that showed the
    error bars understate spread 2-8x, the 23x symmetry swing, the acne grade
    flipping across its own referral threshold. Those are facts about the
    INSTRUMENT and they belong in every report.

    But they were written as "this face" and "this same face", which is only true
    for the one person the instrument was characterised on. Shown to anyone else
    they assert that the READER's face was photographed ten times in 26 seconds,
    which never happened. The finding is right; the referent is wrong.

    Defaults to FALSE -- an unconfigured subject is NOT the validation subject,
    so the prose degrades to "a reference face", which is true for everyone
    including the original. The opposite default would silently mislabel every
    new subject's report, which is the failure mode `subject.py` exists to stop.
    """
    if load()["is_validation_subject"]:
        return "This face" if cap else "this face"
    return "A reference face" if cap else "a reference face"


def age_phrase(prefix=" ", suffix=""):
    """e.g. ' (age 20)' when configured, '' when not. Never guesses."""
    age = load()["age"]
    return f"{prefix}(age {age}){suffix}" if age is not None else ""


def possessive():
    """Possessive pronoun for report prose. 'their' unless configured."""
    p = load()["pronoun"]
    return {"he": "his", "she": "her", "they": "their"}.get(p, "their")


def subject_pronoun():
    p = load()["pronoun"]
    return p if p in ("he", "she", "they") else "they"
