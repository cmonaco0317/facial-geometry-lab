# 0.4.7 — public Apple Silicon beta

This release makes the current local facial-photo application available with its
source and a prebuilt Mac download. It retains automatic forehead-visibility
assessment, individual geometry readings, six report chapters, optional landmark
review, styling experiments and opt-in encrypted local history.

## Compatibility and installation

Apple Silicon (M1 or newer), macOS 13+. No Intel or Windows application is included.
The app contains its Python runtime and aggregate reference. It downloads four
checksum-pinned official Google model assets during setup before photo intake.
No API key is required. It is ad-hoc signed and not Apple-notarized; follow normal
macOS Open/security controls rather than disabling Gatekeeper.

## Reference edition change

The private working app's per-person reference bank is replaced with independent
empirical distributions and cohort-mean shapes. Individual reference geometries,
row IDs, poses and identity-cluster IDs are absent.49 references retain the
existing 150 within-cohort cluster floor, with complementary suppression of
overlapping base cells. Broader comparison fallbacks can change some results and
are disclosed in the report. See [the reference format](PUBLIC_REFERENCE.md).

Neither a public release nor a long test suite establishes anatomical accuracy.
The composite remains a disclosed experimental geometry formula, and hidden
hairline anatomy is never invented. This release is narrower than the full
feature sets advertised by commercial facial-analysis services.

Some otherwise usable reference groups lack an established dimorphism
calibration. That comparison remains unavailable while other readings can
continue. This limitation also exists in the original private reference data;
a clearer photograph cannot supply a missing population calibration.

## Verified source checks

- 217 affected public-source tests passed, including12 aggregate-reference tests.
- 94% statement coverage for the new public-reference reader; scoped Ruff passed.
- A fresh 49-cohort calculation audit matched private same-cohort numeric outputs
  for dimorphism and typicality with 1/2/7 input observations. 1,519 ordinary
  empirical distributions and 49 declared pitch-gate pools matched independently.
- The frozen external-photo consumer-path run rendered 248 reports from 124 assigned
  LFW, Multiface and Pexels source groups. All 248 regional numeric digests and
  axis statuses matched the private 0.4.6 run under the same comparison settings.
  Findings: 204 clear, 8 hair overlap, 35 uncertain and 1 no-face; 988 estimated and
  4 needs-photo axis outcomes. Originals were unchanged; temporary sessions removed.
- Model setup downloaded all four official pinned assets from a clean cache and
  verified their checksums. No face photographs were involved in the downloads.
- Scoped code/security and Python reviews approved the reference implementation.

The source groups and fixed comparison settings above are software workflow
checks, not verified participant demographics or an anatomical accuracy study.
The final signed Apple Silicon app passed two external-source upload/live-report/
HTML/JSON/end-session journeys, with kernel network/home denial and cleanup.
Fixed comparison settings included a fully supported age fallback. Neither
required manual review. The earlier age65 test expected a dimorphism calibration
that the original reference does not contain; its failure is preserved and this
coverage limit is disclosed above.

The ZIP was independently extracted: all 5,286 files and links matched; signature,
architecture, source, payload manifest and aggregate-reference checks passed.
Its SHA-256 is `66a26cd30340e72bcc32f1eb76dc5fc699ca1c5db587f987221d4e5e0bac2c35`.
Anonymous GitHub download verification is recorded after publication. First installation on another person's Mac and Apple
notarization are not established by local checks.
