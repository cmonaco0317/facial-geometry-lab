# Public reference format

The public app uses `public-reference-cdf-v1`. It does not distribute the private
working bank's individual reference meshes or row-linked measurement tables.

## Stored representation

`bank/derived.npz` contains independent empirical distributions. Each key is a
cohort/metric pair with separate strictly increasing threshold values (`|x`) and
integer frequencies (`|n`). No row number links one metric's thresholds to any
other metric. Querying a percentile preserves the original strict comparison,
including exact ties; threshold counts are expanded locally when the existing
arithmetic needs an array.

For typicality, each eligible cohort also has one aggregate symmetric mean shape
and a held-out distance distribution. The fit/held-out individual geometries and
indices are absent. The original source-row count preserves the seeded random
permutation consumption before the subject's noise draws. This keeps the original
one-, two- and seven-photo calculation paths reproducible for retained cohorts.

Dimorphism retains the existing fitted coefficients, scales and calibration
metadata. The public reference stores its calibrated score distribution rather
than a linked matrix of every reference individual's features. Symmetry and the
declared upper-third pitch reference are gated before ECDF conversion.

## Complementary suppression and fallback

Only references meeting the original 150 within-cohort identity-cluster threshold
are eligible. The export then selects disjoint declared base-cell member sets,
retaining 49 references. Overlapping distributions are withheld rather than
publishing both a union and members that could expose a smaller group through
subtraction. Each ECDF requires at least 50 finite values. Withheld cells retain
only cohort/fallback metadata and do not become populated reference arrays.

This partition is about declared dataset cells. Clustering was performed within
cells; the data does not establish that all participants are globally distinct.
Duplicate people or photographs may remain across cells. These aggregate
statistics are **not a formal differential-privacy or anonymity guarantee**.
They remain derived from people and retain FairFace's attribution and CC BY 4.0
terms.

The resolver discloses the comparison cohort it actually uses. Some ages or
comparison selections therefore use a broader reference than the private build.
Results from different reference editions should not be treated as interchangeable
progress measurements. The reference fingerprint changes and saved-history
comparisons retain the existing incompatibility checks.

The public app uses only materialized reference groups. It does not synthesize
unavailable virtual populations from withheld metadata. Unsupported comparison
selections receive an explicit intake message before photo/session storage.

## Validation and integrity

The loader checks schema, checksum, cohort depth, disjoint declared membership,
finite sorted thresholds, positive integer counts, valid aggregate shapes and
archive resource bounds. A separate uncached validation API checks the exact
source/staging directory requested by the builder. Per-person measurement
archives are rejected.

The new public provenance explicitly records an aggregate transformation of the
frozen private instrument. It does not claim new photographs were measured or
new accuracy established. Its measurement-source/model/dependency guard and the
app's signed payload manifest still apply. Public-reference reader and metadata
bytes enter the application's cache fingerprint.

The app's calculations and score remain experimental. Numerical parity for a
retained reference cohort verifies a representation change, not biological
accuracy, demographic fairness or calibrated uncertainty.
