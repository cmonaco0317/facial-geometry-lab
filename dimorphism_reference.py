"""Measured sexual-dimorphism reference — GENERATED, do not hand-edit.

Regenerate with:  ./venv/bin/python dimorphism_bank_probe.py --emit

Every number below is measured on 825 real reference faces of known sex
(MediaPipe 468-point landmarks, frontal-filtered at |yaw| <= 14 deg), split
SELECT = the 10-19 banks / HELD-OUT = the disjoint 20-29 banks. A feature
carries a lean ONLY if it cleared, on faces it was not selected on:
held-out AUC >= 0.60 at p < 0.05, select AUC >= 0.58, and AUC > 0.5 in all
four race- and age-matched pairings separately.

⚠️ ONE FIELD IS NOT MEASURED ON THOSE FACES: 'movement' is the full observed
range of the ratio across the 3 GATE-PASSING SITTINGS OF ONE PERSON named in
MOVEMENT_SITTINGS below. It is a FLOOR -- one face, one camera -- and it can
only GROW as sittings are added, which is the point of adding them.

⛔ These bands are NOT literature anchors and must not be edited toward one.
⛔ Nor are they the population median: a band centred on the median would
   manufacture a 50/50 verdict from a feature carrying no signal. They are
   likelihood-ratio edges — the value beyond which one sex is
   1.5x more likely — computed from the measured distributions.
"""

LEAN_ODDS = 1.5
N_REFERENCE_FACES = 825

# ⛔ ROWS, NOT PEOPLE — and saying "825 real faces" overstated the sample by 63.
# The eight reference pools are NESTED (a white woman aged 20-29 is in
# `white_20_29`, `female_20_29`, `broad_20_29` AND `female_broad_20_29`) and the
# probes concatenate them, so 825 ROWS are drawn from 762 DISTINCT people.
# ⭐ FROZEN LITERALS ON PURPOSE: production must not open `ref_landmarks2.npz` at
# run time (it is gitignored), so these describe a frozen measurement. They are
# guarded by `selfcheck_reference_counts.py`, which re-derives both from the cache
# whenever it is present and fails if either has drifted.
# ✅ NOT a leakage risk: the held-out split is by AGE BAND (10-19 selects, 20-29
# validates) and no person is in two age bands -- measured overlap is exactly 0.
N_DISTINCT_REFERENCE_FACES = 762
REFERENCE_LABEL = (
    f"{N_REFERENCE_FACES} reference rows ({N_DISTINCT_REFERENCE_FACES} distinct faces)"
)

N_MOVEMENT_SITTINGS = 3
MOVEMENT_SITTINGS = ['2026-08-12_second-set', '2026-08-12_third-set', '2026-08-19_fourth-sitting']

FEATURES = {
    'face_width': {
        'label': 'Face width / eye spacing',
        'sex_high': 'M',
        'separates': True,
        'band': [2.0987, 2.3031],
        'auc_select': 0.584,
        'auc_holdout': 0.651,
        'p_holdout': 8.323e-07,
        'auc_by_pairing': {'white 20-29': 0.619, 'all-race 20-29': 0.686, 'white 10-19': 0.575, 'all-race 10-19': 0.586},
        'n_male': 381,
        'n_female': 444,
        'male_mean': 2.2189,
        'female_mean': 2.1828,
        'pooled_sd': 0.0954,
        'pop_median': 2.2065,
        'pct_neutral': 71.8,
        'empirical_odds_at_feminine_edge': 1.58,
        'empirical_odds_at_masculine_edge': 2.14,
        'movement': 0.01412,
        'why': 'greater facial width relative to eye spacing reads masculine. NOTE this is NOT the published fWHR construct (bizygomatic width / brow-to-upper-lip height): that exact ratio was measured here at AUC 0.531 (p=0.32, held-out) and did not clear the bar, which is consistent with the contested replication record for fWHR sex differences. Re-normalising the same width against inter-ocular distance -- a dimension that grows far less between the sexes than face width does -- recovers the separation.',
        'source': 'Bizygomatic breadth robustly larger in males across populations (Farkas anthropometry and later multi-population reviews); fWHR sex-difference claims are contested (Kramer 2017 and subsequent failures to replicate).',
        'percentiles': [1.8672, 1.9678, 1.9898, 2.0073, 2.0199, 2.0323, 2.0399, 2.0499, 2.0559, 2.0614, 2.0635, 2.0685, 2.0791, 2.0865, 2.0903, 2.0954, 2.0995, 2.1044, 2.1101, 2.1129, 2.1207, 2.1248, 2.1276, 2.1308, 2.1342, 2.1372, 2.1384, 2.1416, 2.1453, 2.1474, 2.1493, 2.1532, 2.1577, 2.162, 2.1657, 2.1693, 2.1733, 2.1766, 2.1802, 2.1815, 2.1841, 2.1854, 2.1876, 2.1907, 2.193, 2.1961, 2.1987, 2.2004, 2.2019, 2.2046, 2.2065, 2.2088, 2.2106, 2.2119, 2.214, 2.2163, 2.2181, 2.2234, 2.2253, 2.2264, 2.2296, 2.2314, 2.234, 2.2376, 2.2396, 2.2404, 2.2415, 2.2436, 2.245, 2.2474, 2.2496, 2.2528, 2.2561, 2.2578, 2.2594, 2.262, 2.2659, 2.269, 2.2731, 2.2754, 2.2781, 2.2826, 2.2853, 2.2884, 2.2904, 2.2951, 2.2973, 2.3018, 2.3074, 2.3126, 2.3188, 2.3252, 2.3334, 2.34, 2.346, 2.3535, 2.3655, 2.3697, 2.392, 2.414, 2.4882],
    },
    'jaw_width': {
        'label': 'Jaw width / eye spacing',
        'sex_high': 'M',
        'separates': True,
        'band': [1.6708, 1.8513],
        'auc_select': 0.584,
        'auc_holdout': 0.666,
        'p_holdout': 5.716e-08,
        'auc_by_pairing': {'white 20-29': 0.614, 'all-race 20-29': 0.719, 'white 10-19': 0.572, 'all-race 10-19': 0.585},
        'n_male': 381,
        'n_female': 444,
        'male_mean': 1.7805,
        'female_mean': 1.7416,
        'pooled_sd': 0.0931,
        'pop_median': 1.7607,
        'pct_neutral': 65.8,
        'empirical_odds_at_feminine_edge': 1.67,
        'empirical_odds_at_masculine_edge': 1.9,
        'movement': 0.02115,
        'why': 'a wider jaw relative to eye spacing reads masculine; mandibular breadth is one of the more robustly dimorphic facial dimensions. Measured against eye spacing rather than bizygomatic width, because the bizygomatic denominator is itself male-larger and cancels most of the signal (that ratio measures AUC 0.586 here against 0.666 for this one).',
        'source': 'Mandibular/bigonial breadth larger in males (Farkas 2005 anthropometry; Velemínská 2015 3D facial morphology).',
        'percentiles': [1.5069, 1.5429, 1.5706, 1.5854, 1.5954, 1.6044, 1.6088, 1.6157, 1.6197, 1.6223, 1.6278, 1.6343, 1.637, 1.6433, 1.6506, 1.6561, 1.6613, 1.6651, 1.6699, 1.6723, 1.6768, 1.6807, 1.6835, 1.6897, 1.6927, 1.6957, 1.6999, 1.7061, 1.7083, 1.7105, 1.7118, 1.7145, 1.7178, 1.7189, 1.7221, 1.7241, 1.726, 1.7292, 1.7324, 1.7346, 1.7364, 1.7405, 1.743, 1.746, 1.7502, 1.7516, 1.7525, 1.7537, 1.7569, 1.7594, 1.7607, 1.765, 1.7694, 1.7721, 1.7737, 1.7754, 1.7775, 1.7791, 1.7812, 1.7842, 1.7863, 1.7894, 1.7909, 1.7945, 1.7979, 1.8001, 1.8023, 1.804, 1.8056, 1.8085, 1.8115, 1.8128, 1.8152, 1.8179, 1.8196, 1.8232, 1.8251, 1.8279, 1.831, 1.8346, 1.8368, 1.839, 1.8425, 1.8443, 1.8505, 1.853, 1.8592, 1.8651, 1.8691, 1.8727, 1.8767, 1.8812, 1.8867, 1.8995, 1.9027, 1.9172, 1.926, 1.9365, 1.961, 1.9814, 2.0552],
    },
    'eye_size': {
        'label': 'Eye width / face width',
        'sex_high': 'F',
        'separates': False,
        'band': None,
        'auc_select': 0.583,
        'auc_holdout': 0.599,
        'p_holdout': 1.244e-03,
        'auc_by_pairing': {'white 20-29': 0.581, 'all-race 20-29': 0.613, 'white 10-19': 0.615, 'all-race 10-19': 0.548},
        'n_male': 381,
        'n_female': 444,
        'male_mean': 0.1933,
        'female_mean': 0.1968,
        'pooled_sd': 0.0108,
        'pop_median': 0.1947,
        'movement': 0.00199,
        'why': 'relatively larger eyes read feminine. Retained and PRINTED, but it carries no lean: measured held-out AUC 0.599 against a bar of 0.60 fixed before the measurement was made. It misses by 0.001. ⛔ The bar was NOT lowered to admit it -- a threshold moved after seeing where a candidate landed is fitted to that candidate, and this project has retired claims for less.',
        'source': 'Females show greater vertical eye-fissure height and larger relative eye dimensions (Velemínská 2015 3D facial morphology).',
        'percentiles': [0.1616, 0.1724, 0.1752, 0.1765, 0.1773, 0.1779, 0.1788, 0.1795, 0.1807, 0.1813, 0.1815, 0.1819, 0.1824, 0.1827, 0.1834, 0.1838, 0.1844, 0.1847, 0.1852, 0.1856, 0.186, 0.1863, 0.1867, 0.1869, 0.1873, 0.1876, 0.1878, 0.1882, 0.1886, 0.1887, 0.1889, 0.1892, 0.1895, 0.1898, 0.1902, 0.1904, 0.1906, 0.191, 0.1914, 0.1917, 0.192, 0.1923, 0.1924, 0.1927, 0.193, 0.1934, 0.1936, 0.1939, 0.1941, 0.1945, 0.1947, 0.1949, 0.1951, 0.1954, 0.1959, 0.1962, 0.1964, 0.1967, 0.1972, 0.1975, 0.1976, 0.1979, 0.1981, 0.1984, 0.1985, 0.1991, 0.1994, 0.1997, 0.2001, 0.2003, 0.2007, 0.2012, 0.2015, 0.2018, 0.2023, 0.2025, 0.2028, 0.2032, 0.2036, 0.2041, 0.2044, 0.2046, 0.205, 0.2055, 0.2061, 0.2066, 0.2069, 0.2073, 0.208, 0.2085, 0.2091, 0.2101, 0.2112, 0.2124, 0.213, 0.2139, 0.2155, 0.2165, 0.2188, 0.2219, 0.229],
    },
    'lower_third': {
        'label': 'Lower-face height / eye spacing',
        'sex_high': 'M',
        'separates': True,
        'band': [0.9428, 1.1847],
        'auc_select': 0.594,
        'auc_holdout': 0.621,
        'p_holdout': 7.415e-05,
        'auc_by_pairing': {'white 20-29': 0.609, 'all-race 20-29': 0.647, 'white 10-19': 0.577, 'all-race 10-19': 0.609},
        'n_male': 381,
        'n_female': 444,
        'male_mean': 1.0842,
        'female_mean': 1.0433,
        'pooled_sd': 0.1105,
        'pop_median': 1.0629,
        'pct_neutral': 72.6,
        'empirical_odds_at_feminine_edge': 1.78,
        'empirical_odds_at_masculine_edge': 1.93,
        'movement': 0.04535,
        'why': 'a taller lower face relative to eye spacing reads masculine; lower-face height is dimorphic and is the part of facial height that grows most with male pubertal development.',
        'source': 'Lower facial height greater in males; mandibular growth is the dominant male-specific adolescent change (Farkas anthropometry; Enlow craniofacial growth).',
        'percentiles': [0.8001, 0.8421, 0.8541, 0.8663, 0.8753, 0.8879, 0.8943, 0.9021, 0.9061, 0.9082, 0.9189, 0.9274, 0.935, 0.939, 0.9417, 0.9451, 0.9498, 0.9525, 0.9562, 0.9591, 0.9631, 0.9676, 0.972, 0.9787, 0.982, 0.9847, 0.9873, 0.9901, 0.9926, 0.9951, 0.9975, 1.0001, 1.0025, 1.0055, 1.0086, 1.0128, 1.0154, 1.0183, 1.0213, 1.0253, 1.0296, 1.0328, 1.0354, 1.039, 1.0438, 1.0477, 1.0498, 1.055, 1.0565, 1.0596, 1.0629, 1.0663, 1.0687, 1.0711, 1.073, 1.0753, 1.0774, 1.08, 1.0818, 1.0855, 1.0868, 1.0884, 1.0913, 1.0968, 1.099, 1.1012, 1.1048, 1.1066, 1.1088, 1.1151, 1.1182, 1.1216, 1.125, 1.1259, 1.1278, 1.13, 1.1332, 1.1393, 1.1421, 1.1444, 1.1494, 1.1551, 1.1597, 1.1649, 1.1718, 1.1773, 1.1807, 1.1852, 1.1916, 1.1978, 1.2084, 1.2118, 1.2186, 1.2314, 1.2365, 1.2483, 1.2644, 1.282, 1.3034, 1.341, 1.5488],
    },
    'nose_width': {
        'label': 'Nose width / mouth width',
        'sex_high': 'M',
        'separates': True,
        'band': [0.6696, 0.8331],
        'auc_select': 0.6,
        'auc_holdout': 0.635,
        'p_holdout': 1.046e-05,
        'auc_by_pairing': {'white 20-29': 0.611, 'all-race 20-29': 0.648, 'white 10-19': 0.61, 'all-race 10-19': 0.59},
        'n_male': 381,
        'n_female': 444,
        'male_mean': 0.7662,
        'female_mean': 0.7365,
        'pooled_sd': 0.0774,
        'pop_median': 0.7486,
        'pct_neutral': 66.5,
        'empirical_odds_at_feminine_edge': 2.53,
        'empirical_odds_at_masculine_edge': 1.28,
        'movement': 0.02535,
        'why': 'a wider nose relative to the mouth reads masculine. ⭐ THE PREVIOUS NORMALISATION WAS MEASURABLY BACKWARDS: nose width / bizygomatic width separates the sexes at AUC 0.445 here -- the WRONG side of a coin flip -- because the bizygomatic denominator is itself male-larger and more than cancels the numerator. Absolute nasal breadth is robustly dimorphic; the ratio that was shipped was not.',
        'source': 'Alar width ~44.4mm male vs ~39.3mm female (p<0.001); all linear nasal dimensions larger in males (Ozdemir 2015; reviews after Farkas 2005).',
        'percentiles': [0.5955, 0.6137, 0.623, 0.627, 0.6295, 0.6353, 0.64, 0.643, 0.6446, 0.6484, 0.6512, 0.6539, 0.6562, 0.6585, 0.6606, 0.6619, 0.6638, 0.6668, 0.6692, 0.6726, 0.6768, 0.6813, 0.6821, 0.6838, 0.6873, 0.6895, 0.6927, 0.6944, 0.6965, 0.6978, 0.6993, 0.7019, 0.7044, 0.7052, 0.7089, 0.7114, 0.713, 0.7154, 0.719, 0.7202, 0.7212, 0.7235, 0.7262, 0.7292, 0.7318, 0.7348, 0.7366, 0.7409, 0.7438, 0.7453, 0.7486, 0.7507, 0.7528, 0.7543, 0.7565, 0.7581, 0.7601, 0.7616, 0.763, 0.7637, 0.7682, 0.7701, 0.7717, 0.774, 0.7755, 0.7772, 0.7806, 0.7826, 0.7853, 0.7869, 0.788, 0.7892, 0.7929, 0.7964, 0.7996, 0.8035, 0.8064, 0.8101, 0.8118, 0.8142, 0.8166, 0.8195, 0.822, 0.8262, 0.8321, 0.834, 0.8379, 0.8412, 0.8457, 0.8482, 0.8525, 0.8586, 0.8644, 0.8683, 0.8793, 0.8861, 0.8944, 0.9091, 0.9187, 0.9588, 1.0997],
    },
    'lip_height': {
        'label': 'Lip height / lower-face height',
        'sex_high': 'F',
        'separates': True,
        'band': [0.235, 0.4412],
        'auc_select': 0.582,
        'auc_holdout': 0.638,
        'p_holdout': 6.728e-06,
        'auc_by_pairing': {'white 20-29': 0.676, 'all-race 20-29': 0.611, 'white 10-19': 0.633, 'all-race 10-19': 0.531},
        'n_male': 381,
        'n_female': 444,
        'male_mean': 0.3238,
        'female_mean': 0.3523,
        'pooled_sd': 0.0851,
        'pop_median': 0.3442,
        'pct_neutral': 78.5,
        'empirical_odds_at_feminine_edge': 1.58,
        'empirical_odds_at_masculine_edge': 1.89,
        'movement': 0.00215,
        'why': 'greater vermilion height relative to lower-face height reads feminine. Normalised against lower-face height rather than mouth width because mouth width is itself dimorphic in the same direction and dilutes the signal (that ratio measures AUC 0.546 here against 0.638 for this one). Lip fullness also thins markedly with age, which this cannot separate.',
        'source': 'Increased vermilion height associated with femininity (Penna 2024 ideal-lips anthropometry; Velemínská 2015 3D analysis).',
        'percentiles': [0.086, 0.1618, 0.1761, 0.1867, 0.1983, 0.2078, 0.2109, 0.2202, 0.2239, 0.2282, 0.2336, 0.2359, 0.2382, 0.2411, 0.2418, 0.2443, 0.2483, 0.2519, 0.2541, 0.2569, 0.2592, 0.2618, 0.2647, 0.2693, 0.2707, 0.2727, 0.2752, 0.2776, 0.28, 0.2816, 0.2861, 0.2884, 0.2925, 0.2952, 0.298, 0.3011, 0.3047, 0.3073, 0.3105, 0.3123, 0.3147, 0.3181, 0.3207, 0.3248, 0.3273, 0.3293, 0.3345, 0.3359, 0.3379, 0.3422, 0.3442, 0.3451, 0.3461, 0.3498, 0.354, 0.3557, 0.3586, 0.3606, 0.3625, 0.364, 0.365, 0.3666, 0.369, 0.3698, 0.3717, 0.375, 0.3762, 0.3773, 0.3785, 0.3806, 0.382, 0.3829, 0.3846, 0.3907, 0.393, 0.3956, 0.399, 0.4029, 0.4075, 0.4091, 0.4122, 0.4142, 0.4161, 0.4173, 0.4202, 0.4274, 0.431, 0.4342, 0.4373, 0.4403, 0.4429, 0.4485, 0.454, 0.4664, 0.4694, 0.4743, 0.4891, 0.4968, 0.5197, 0.5539, 0.6869],
    },
    'brow_gap': {
        'label': 'Brow-to-eyelid gap / eye width',
        'sex_high': 'F',
        'separates': False,
        'band': None,
        'auc_select': 0.491,
        'auc_holdout': 0.467,
        'p_holdout': 2.831e-01,
        'auc_by_pairing': {'white 20-29': 0.495, 'all-race 20-29': 0.452, 'white 10-19': 0.482, 'all-race 10-19': 0.495},
        'n_male': 381,
        'n_female': 444,
        'male_mean': 0.3044,
        'female_mean': 0.2936,
        'pooled_sd': 0.1318,
        'pop_median': 0.2906,
        'movement': 0.06151,
        'why': "a larger visible gap between brow and upper lid is CLAIMED to read feminine. ⛔ IT DOES NOT DO SO HERE, and the value is printed without a lean for that reason. Measured on 825 reference rows of known sex (762 distinct faces) this ratio separates at AUC 0.467 -- a coin flip, and if anything on the wrong side -- and it reads below 0.5 in ALL FOUR race- and age-matched pairings separately. Seventeen renormalisations across two rounds were tried (brow head, mid-brow, lateral brow, arch, brow span, against eye width, IPD, face height and lower-face height) and none cleared the bar. The mechanism is not mysterious: the sex-linked part of this anatomy is supraorbital RIDGE PROJECTION, which is a DEPTH measurement, and these are frontal landmarks. The profile path does see the glabella, but its own cross-session check failed (metrics_profile._meta.cross_session_warning), so it cannot carry a verdict either. ⚠️ The old band was 0.50-0.60 against a real population median of 0.291, so it labelled 93% of all 762 people -- men and women alike -- 'masculine'.",
        'source': 'Brow ridge robustly more prominent in males (Godinho/Spikins 2018); 3D analysis finds brow top-edge height ~equal between sexes, female brows thinner, so the apparent gap is largely a grooming artefact (Gao 2023).',
        'percentiles': [0.0007, 0.041, 0.0523, 0.076, 0.0878, 0.0964, 0.1061, 0.1148, 0.1214, 0.1266, 0.1301, 0.1348, 0.1414, 0.1471, 0.156, 0.159, 0.1631, 0.1691, 0.1724, 0.1756, 0.1856, 0.1889, 0.1913, 0.1929, 0.1963, 0.199, 0.2014, 0.2091, 0.2127, 0.2189, 0.2208, 0.2235, 0.2272, 0.2297, 0.2319, 0.2385, 0.2429, 0.2482, 0.2521, 0.2551, 0.26, 0.2634, 0.2661, 0.2688, 0.273, 0.2761, 0.2786, 0.2804, 0.2831, 0.2874, 0.2906, 0.2935, 0.2993, 0.3013, 0.3049, 0.3076, 0.3119, 0.3163, 0.3204, 0.3238, 0.3256, 0.3308, 0.3343, 0.3382, 0.344, 0.3478, 0.3518, 0.3554, 0.3605, 0.3646, 0.3675, 0.3709, 0.3751, 0.3785, 0.3847, 0.3892, 0.3935, 0.3985, 0.4037, 0.4064, 0.4095, 0.4131, 0.418, 0.4246, 0.4305, 0.4386, 0.4443, 0.4486, 0.4548, 0.4614, 0.4668, 0.472, 0.488, 0.4949, 0.512, 0.5237, 0.5383, 0.5544, 0.5825, 0.6263, 0.7752],
    },
}

# Features whose values move together, so the tally counts SIGNALS not rows.
# Single-link clustering at |r| >= 0.50 on the same reference faces.
CLUSTERS = [['eye_size', 'face_width', 'jaw_width'], ['lower_third'], ['nose_width'], ['lip_height'], ['brow_gap']]
CORRELATION = {
    'face_width': {'face_width': 1.0, 'jaw_width': 0.83, 'eye_size': -0.68, 'lower_third': 0.4, 'nose_width': -0.13, 'lip_height': -0.08, 'brow_gap': -0.07},
    'jaw_width': {'face_width': 0.83, 'jaw_width': 1.0, 'eye_size': -0.59, 'lower_third': 0.45, 'nose_width': -0.2, 'lip_height': -0.17, 'brow_gap': -0.07},
    'eye_size': {'face_width': -0.68, 'jaw_width': -0.59, 'eye_size': 1.0, 'lower_third': -0.26, 'nose_width': 0.08, 'lip_height': 0.08, 'brow_gap': -0.1},
    'lower_third': {'face_width': 0.4, 'jaw_width': 0.45, 'eye_size': -0.26, 'lower_third': 1.0, 'nose_width': -0.25, 'lip_height': 0.31, 'brow_gap': 0.23},
    'nose_width': {'face_width': -0.13, 'jaw_width': -0.2, 'eye_size': 0.08, 'lower_third': -0.25, 'nose_width': 1.0, 'lip_height': -0.3, 'brow_gap': 0.1},
    'lip_height': {'face_width': -0.08, 'jaw_width': -0.17, 'eye_size': 0.08, 'lower_third': 0.31, 'nose_width': -0.3, 'lip_height': 1.0, 'brow_gap': 0.11},
    'brow_gap': {'face_width': -0.07, 'jaw_width': -0.07, 'eye_size': -0.1, 'lower_third': 0.23, 'nose_width': 0.1, 'lip_height': 0.11, 'brow_gap': 1.0},
}
