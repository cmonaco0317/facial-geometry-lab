"""FROZEN reference split-half movement — emitted by stranger_movement_probe.py.

⛔ DO NOT HAND-EDIT. Regenerate with:
    ./venv/bin/python stranger_movement_probe.py --emit

WHAT THIS IS. For each shipped ratio, how far the reference subject's reading
moves between TWO DISJOINT HALVES of one sitting that EACH independently pass
the real intake gate. It is a same-instant quantity: expression, lighting,
camera and posture are all held constant, so it is a FLOOR on cross-sitting
movement and must never be quoted as one.

WHAT IT IS FOR. It is the DENOMINATOR of a like-for-like comparison. A new
subject's own split-half movement, measured by the identical operation, divided
by this, says how much more that person's photographs disagree than the person
every margin in this instrument was measured on.

⛔ WHY NOT cross-sitting / split-half. That ratio was tried on 2026-08-28 and is
CONFOUNDED: cross-sitting movement is measured between FULL-set fits (7-9 views,
RMS 6.1-6.7px) and split-half between HALF-set fits (4-5 views, RMS up to
11.3px). Dividing them measures fit quality, not sittings -- it came out at a
median of 0.96x with a 43.6x spread and went BELOW 1.0 on 10 of 18 ratios, which
is impossible for a true floor relationship. Only same-operation ratios survive.

⚠️ n = 2 sittings of ONE person. `lo`/`hi` are the individual sittings,
not a confidence interval. The aggregation matters: the shipped 1.30x stranger
multiplier is 0.93x using `hi` and 2.79x using `lo` on this same data.
"""

from __future__ import annotations

N_SITTINGS = 2
SITTINGS = ['calibration participant:2026-08-12_third-set', 'calibration participant:2026-08-19_fourth-sitting']

# ratio -> {median, lo, hi, n} of split-half movement, in the ratio's own units
SPLIT_HALF = {
    'cat:alar_ratio': {'median': 0.0014976082457578421, 'lo': 0.0014559746048807276, 'hi': 0.0015392418866349566, 'n': 2},
    'cat:brow_position': {'median': 0.003311200594575625, 'lo': 0.0007626280732459245, 'hi': 0.005859773115905326, 'n': 2},
    'cat:canthal_tilt': {'median': 0.24416620762675212, 'lo': 0.08872548904302935, 'hi': 0.3996069262104749, 'n': 2},
    'cat:chin_jaw_ratio': {'median': 0.0051202592025488824, 'lo': 0.0027148365914047012, 'hi': 0.007525681813693064, 'n': 2},
    'cat:eye_almondness': {'median': 0.01817028981674143, 'lo': 0.0023809019333903003, 'hi': 0.03395967770009256, 'n': 2},
    'cat:eye_spacing': {'median': 0.004940515925827144, 'lo': 0.0012989533508047657, 'hi': 0.008582078500849522, 'n': 2},
    'cat:facial_index': {'median': 0.01309986206522662, 'lo': 0.0030819055944468765, 'hi': 0.023117818536006363, 'n': 2},
    'cat:forehead_ratio': {'median': 0.006095525487354536, 'lo': 0.0030714123764216117, 'hi': 0.00911963859828746, 'n': 2},
    'cat:fwhr': {'median': 0.02864404817995736, 'lo': 0.008713133020318065, 'hi': 0.048574963339596655, 'n': 2},
    'cat:jaw_ratio': {'median': 0.001391013692784937, 'lo': 0.0011848324421017686, 'hi': 0.0015971949434681054, 'n': 2},
    'cat:lip_lower_face': {'median': 0.014469145399918698, 'lo': 0.0057454096950315114, 'hi': 0.023192881104805885, 'n': 2},
    'cat:lower_third': {'median': 0.005731677731793311, 'lo': 0.0033994628911478975, 'hi': 0.008063892572438724, 'n': 2},
    'cat:middle_lower_ratio': {'median': 0.007614559646368424, 'lo': 0.007317870384320413, 'hi': 0.007911248908416435, 'n': 2},
    'cat:middle_third': {'median': 0.0021999804689971647, 'lo': 0.0005944148929140081, 'hi': 0.0038055460450803213, 'n': 2},
    'cat:mouth_ratio': {'median': 0.0037436619237601054, 'lo': 0.0013045952960643592, 'hi': 0.006182728551455852, 'n': 2},
    'cat:naso_oral': {'median': 0.01017085934387596, 'lo': 0.0009203573426156408, 'hi': 0.01942136134513628, 'n': 2},
    'cat:philtrum_ratio': {'median': 0.015171875625188305, 'lo': 0.008503214819535698, 'hi': 0.021840536430840912, 'n': 2},
    'cat:upper_third': {'median': 0.007337243307876454, 'lo': 0.0028050479982338894, 'hi': 0.011869438617519018, 'n': 2},
    'dim:brow_gap': {'median': 0.015407806753691175, 'lo': 0.015209302185811846, 'hi': 0.015606311321570504, 'n': 2},
    'dim:eye_size': {'median': 0.0029146789101001935, 'lo': 0.0008091095785187907, 'hi': 0.005020248241681596, 'n': 2},
    'dim:face_width': {'median': 0.02652614989383273, 'lo': 0.020066735089924403, 'hi': 0.032985564697741054, 'n': 2},
    'dim:jaw_width': {'median': 0.018048306054224916, 'lo': 0.013389804633644697, 'hi': 0.022706807474805135, 'n': 2},
    'dim:lip_height': {'median': 0.014469145399918698, 'lo': 0.0057454096950315114, 'hi': 0.023192881104805885, 'n': 2},
    'dim:lower_third': {'median': 0.01914420075520895, 'lo': 0.01612224950268315, 'hi': 0.022166152007734752, 'n': 2},
    'dim:nose_width': {'median': 0.01017085934387596, 'lo': 0.0009203573426156408, 'hi': 0.01942136134513628, 'n': 2},
}
