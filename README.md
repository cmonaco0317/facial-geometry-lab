# Facial Analysis for Mac

A local facial-photo analysis app with inspectable measurements, illustrated
reports, optional landmark review and encrypted history.

## Download and open

**[Download the Mac app,0.4.7](https://github.com/cmonaco0317/facial-geometry-lab/releases/tag/v0.4.7)**

The current download is **Apple Silicon (M1 or newer), macOS 13 or later**.
Intel Macs and Windows are not supported by this release. No Python, Homebrew,
API key, subscription or separate reference-bank setup is needed for the app.

1. Download `Facial-Analysis-0.4.7-Apple-Silicon.zip` and `SHA256SUMS.txt` from the
   release page. Optionally check the ZIP with `shasum -a256` against its entry.
2. Extract the ZIP and drag **Facial Analysis.app** into Applications.
3. Open it using macOS's normal Open/security controls. This private-developer
   beta is ad-hoc signed and **not Apple-notarized**. If macOS blocks it, follow
   the standard **System Settings → Privacy & Security → Open Anyway** flow if
   you choose to trust the download. Do not disable Gatekeeper globally.
4. First setup needs internet for checksum-pinned Google models. Setup finishes
   before you select photos; subsequent analysis runs locally.
5. Create one adult's session, select that person's own clear photos and review
   the report. Start with a neutral frontal image; add other views when requested.
6. Export HTML/JSON if desired. **End session** removes temporary session inputs.
   Explicit exports and intentionally saved encrypted history remain until deleted.

## What it does

- Detects facial landmarks and computes up to 20 regional geometry readings,
  including two separate iris-centre observations.
- Presents harmony, dimorphism, symmetry and typicality with capture-quality
  explanations and a disclosed experimental composite.
- Checks hair/forehead visibility automatically. Unsupported forehead readings
  are withheld; other regions can continue. Optional review can correct a finding.
- Provides six illustrated report chapters, optional frontal/hairline/profile
  review, and preference-aware hair, facial-hair and eyewear styling experiments.
- Offers local encrypted history, explicit backups and cautious report comparisons.

A mesh point is not necessarily an anatomical landmark. A hidden hairline cannot
be reconstructed from a fringe. Measurements depend on photo quality, viewpoint,
expression and point placement. The composite is a design-weighted geometry
summary, **not a validated attractiveness rating or population beauty percentile**.
This is research software for adults, not medical or treatment guidance.

## Public reference edition

Version 0.4.7 replaces the private working app's per-person reference records with
independent empirical CDF threshold/count pairs and cohort-mean shapes. No
reference photographs, individual landmark meshes, row IDs, per-person poses or
identity-cluster IDs are included.

The public reference uses 49 eligible groups with disjoint declared base-cell
membership. Each meets the existing 150 identity-cluster threshold; each published
CDF needs at least 50 finite observations. Other cells remain metadata-only.
Complementary suppression prevents subtracting overlapping public age groups to
recover omitted small-cell distributions. These are source-dataset cells, not a
guarantee of globally distinct people or formal anonymity.

Where the selected cohort is retained, the goal is to preserve the private
instrument's numerical calculations. Some demographic combinations use broader
reference fallbacks in this public edition, so results can differ; the report
shows the requested and used comparison group. Detailed verification and limits
are recorded in [the release notes](docs/RELEASE_0.4.7.md).

Some retained reference groups lack an established dimorphism calibration;
that card remains unavailable while other readings can continue.

The public comparison bank currently supports male/female reference labels with
one listed ethnicity. Mixed, multiple, unlisted or unspecified comparison
selections may have no materialized reference. Intake explains that limitation
before storing photographs; it does not infer a replacement sex or ancestry.

Photos selected by the user remain on that user's Mac. The packaged analysis
process is confined to inbound loopback networking. Exported reports still
contain sensitive measurements and should be shared deliberately.

## Run from source

The prebuilt app is the easiest path. Source execution requires **CPython 3.11.15**
plus the pinned versions in `requirements.txt`; the reference instrument checks
its dependency and model versions rather than silently using different ones.

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -c 'from pathlib import Path; from macos_app.config import MODEL_ASSETS; from macos_app.bootstrap import ensure_asset; [ensure_asset(Path.cwd() / name, asset) for name, asset in MODEL_ASSETS.items()]'
python local_page.py
```

Source execution is a development mode; use the packaged app for its verified
macOS network confinement and installation behavior.

```sh
python -m unittest discover -s tests -v
python demo.py
```

`demo.py` retains the original offline demonstration using entirely fictional
schematic coordinates. Its invented display scores are clearly labelled and
are separate from the real-photo application.

## License and credits

Application source: Apache-2.0, see [LICENSE](LICENSE). Public reference statistics
derive from FairFace and retain its CC BY 4.0 attribution and terms. Google model
weights download from official sources and are not redistributed in the release.
See [NOTICE](NOTICE) and [reference format](docs/PUBLIC_REFERENCE.md).
