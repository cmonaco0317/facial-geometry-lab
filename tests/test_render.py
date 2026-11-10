import html
from pathlib import Path
import re
import unittest

from local_app.analysis import AnalysisReport
from local_app.contracts import AxisResult, CohortUse, PhotoRequest
from local_app.render import render_landing, render_results


class LandingRenderTests(unittest.TestCase):
    def test_landing_collects_demographics_and_one_or_more_photos(self) -> None:
        page = render_landing()

        self.assertRegex(page, r'<input[^>]+name="age"[^>]+min="18"')
        self.assertIn('name="sex"', page)
        self.assertGreaterEqual(page.count('name="ethnicity"'), 6)
        self.assertRegex(page, r'<input[^>]+type="file"[^>]+name="photos"[^>]+multiple')
        self.assertIn("1 or more", page)
        self.assertIn('enctype="multipart/form-data"', page)

    def test_landing_says_analysis_is_local_and_has_no_folder_path_field(self) -> None:
        page = render_landing()
        visible = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", page)).lower()

        self.assertIn("stays on this computer", visible)
        self.assertIn("deleted when you end", visible)
        self.assertNotIn("folder of your photographs", visible)
        self.assertNotRegex(page, r'name="photos"[^>]+type="text"')
        self.assertNotIn("http://", page)
        self.assertNotIn("https://", page)

    def test_landing_uses_semantic_grouping_and_a_real_submit_action(self) -> None:
        page = render_landing(action="/sessions")

        self.assertGreaterEqual(page.count("<fieldset"), 2)
        self.assertGreaterEqual(page.count("<legend"), 2)
        self.assertIn('<form class="intake-form" method="post" action="/sessions"', page)
        self.assertIn("Create local analysis", page)


class ResultRenderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cohort = CohortUse(
            requested_id="f_east_asian_30_39",
            used_id="f_all_ethnicities_30_39",
            fallback_level="same_age_sex_broader_ethnicity",
            sample_size=1_204,
            requested_label="East Asian women aged 30-39",
            used_label="Women of all available ethnicities aged 30-39",
            member_ids=("f_white_30_39", "f_black_30_39"),
        )
        self.request = PhotoRequest(
            view="straight-on at eye level",
            expression="neutral, lips relaxed",
            lighting="soft, even light across the whole face",
            framing="head and shoulders; full face visible",
            reason="Symmetry needs 1 more eligible photo looking directly at the lens.",
        )
        self.report = AnalysisReport(
            session_id="safe-session-id-abcdefghijklmnopqrstuvwxyz",
            cohort=self.cohort,
            axes={
                "harmony": AxisResult.measured_profile(
                    axis="harmony",
                    summary="Six geometric proportions were measured.",
                    cohort=self.cohort,
                    photo_count=7,
                    details={
                        "profile": {
                            "horizontal_balance": {
                                "label": "Horizontal balance",
                                "value": 0.97,
                                "percentile": 54.0,
                                "interval": [46.0, 62.0],
                            }
                        },
                        "evidence_photo_ids": ["p1", "p2"],
                        "overall_index": {
                            "value": 37.52,
                            "debug_path": "/private/stale-score.json",
                        },
                    },
                ),
                "dimorphism": AxisResult.measured(
                    axis="dimorphism",
                    summary="Sex-aligned facial geometry is at the 61st percentile.",
                    value=72.0,
                    unit="% alignment probability",
                    percentile=61.0,
                    interval=(54.0, 68.0),
                    cohort=self.cohort,
                    photo_count=7,
                    details={"p_male": 0.28, "features_used": ["jaw_width"]},
                ),
                "symmetry": AxisResult.needs_photo(
                    axis="symmetry",
                    request=self.request,
                    cohort=self.cohort,
                    photo_count=3,
                    details={"eligible_photos": 3},
                ),
                "averageness": AxisResult.processing_error(
                    axis="averageness",
                    error_code="reference_unreadable",
                    error_message="The local averageness reference could not be read.",
                    retryable=True,
                    cohort=self.cohort,
                    photo_count=7,
                ),
            },
            accepted_photos=8,
            instrument_fingerprint="instrument-v1",
            photo_fingerprint="photos-v1",
        )

    def test_results_always_render_four_accessible_cards(self) -> None:
        page = render_results(self.report, csrf_token="csrf-token")

        self.assertEqual(page.count("<article class=\"axis-card"), 4)
        for axis in ("Harmony", "Dimorphism", "Symmetry", "Averageness"):
            with self.subTest(axis=axis):
                slug = axis.lower()
                self.assertIn(f'id="{slug}-title"', page)
                self.assertIn(f'aria-labelledby="{slug}-title"', page)

    def test_packaged_results_change_only_the_end_button(self) -> None:
        normal = render_results(self.report, csrf_token="token")
        packaged = render_results(
            self.report,
            csrf_token="token",
            quit_after_end=True,
        )

        self.assertEqual(normal.count("End session and delete files"), 1)
        self.assertEqual(
            packaged,
            normal.replace("End session and delete files", "End session and quit"),
        )
        self.assertIn("End session and delete files", normal)
        self.assertNotIn("End session and quit", normal)
        self.assertIn("End session and quit", packaged)
        self.assertNotIn("End session and delete files", packaged)

    def test_measured_card_shows_uncertainty_evidence_and_fallback(self) -> None:
        page = render_results(self.report, csrf_token="csrf-token")

        self.assertIn("61st percentile", page)
        self.assertIn("54th–68th percentile", page)
        self.assertIn("7 eligible photos", page)
        self.assertEqual(page.count("1,204 reference identities"), 1)
        self.assertIn("Broader ethnicity fallback", page)
        self.assertIn(self.cohort.requested_label, page)
        self.assertIn(self.cohort.used_label, page)
        self.assertIn("Method and limits", page)
        self.assertIn("How the reference labels were assigned", page)
        self.assertIn("annotator judgement from a photograph", page)

    def test_results_begin_with_an_evidence_led_four_pillar_summary(self) -> None:
        page = render_results(self.report, csrf_token="csrf-token")
        normalized = re.sub(r"\s+", " ", page)

        for text in (
            "Four-pillar facial analysis",
            "Your results at a glance",
            "Harmony profile",
            "Proportion balance contributes to the overall score",
            "The overall score uses an explicit geometric formula",
        ):
            with self.subTest(text=text):
                self.assertIn(text, page)
        self.assertLess(page.index("Your results at a glance"), page.index("axis-grid"))
        self.assertNotIn("38 / 100", page)
        self.assertNotIn("37.52", page)
        self.assertNotIn("debug_path", page)
        self.assertIn("only nose–mouth–eye", normalized.lower())
        self.assertIn("60 Chinese volunteers", normalized)

    def test_visible_percentile_ranges_have_accessible_tracks(self) -> None:
        page = render_results(self.report, csrf_token="csrf-token")

        self.assertGreaterEqual(page.count('class="measurement-track"'), 2)
        tracks = re.findall(r'<svg class="measurement-track"[^>]+>', page)
        self.assertTrue(tracks)
        for track in tracks:
            self.assertIn('role="img"', track)
            self.assertRegex(track, r'aria-label="[^"]+"')

    def test_profile_photo_request_and_processing_error_each_render_honestly(self) -> None:
        page = render_results(self.report, csrf_token="csrf-token")

        self.assertIn("Profile measured", page)
        self.assertIn("Proportion balance contributes to the overall score", page)
        self.assertIn("available component profile", page)
        self.assertNotIn("six-part profile", page)
        self.assertIn("Horizontal balance", page)
        self.assertIn("46th–62nd", page)
        for text in (
            self.request.view,
            self.request.expression,
            self.request.lighting,
            self.request.framing,
            self.request.reason,
        ):
            self.assertIn(text, page)
        self.assertIn("The local averageness reference could not be read.", page)
        self.assertIn("Retry averageness", page)

    def test_available_harmony_scalar_keeps_its_underlying_profile_visible(self) -> None:
        axes = dict(self.report.axes)
        axes["harmony"] = AxisResult.measured(
            axis="harmony",
            summary="Cohort-relative geometric harmony is at the 72nd percentile.",
            value=72.0,
            unit="geometric harmony percentile",
            percentile=72.0,
            interval=(64.0, 80.0),
            cohort=self.cohort,
            photo_count=7,
            details={
                "profile": {
                    "horizontal_balance": {
                        "label": "Horizontal balance",
                        "value": 0.97,
                        "percentile": 54.0,
                        "interval": [46.0, 62.0],
                    }
                }
            },
        )
        report = AnalysisReport(
            session_id=self.report.session_id,
            cohort=self.report.cohort,
            axes=axes,
            accepted_photos=self.report.accepted_photos,
            instrument_fingerprint=self.report.instrument_fingerprint,
            photo_fingerprint=self.report.photo_fingerprint,
        )

        page = render_results(report, csrf_token="csrf-token")

        self.assertIn("72nd percentile", page)
        self.assertIn("Underlying proportions", page)
        self.assertIn("Horizontal balance", page)
        self.assertIn("46th–62nd", page)
        self.assertNotIn("Harmony remains a component profile", page)
        self.assertIn("The harmony headline is a validated geometric measure", page)

    def test_unavailable_harmony_does_not_claim_a_component_profile(self) -> None:
        states = (
            AxisResult.needs_photo(
                axis="harmony",
                request=self.request,
                cohort=self.cohort,
                photo_count=3,
            ),
            AxisResult.processing_error(
                axis="harmony",
                error_code="measurement_failed",
                error_message="Harmony could not be measured.",
                retryable=True,
                cohort=self.cohort,
                photo_count=7,
            ),
        )
        for harmony in states:
            with self.subTest(status=harmony.status):
                axes = dict(self.report.axes)
                axes["harmony"] = harmony
                report = AnalysisReport(
                    session_id=self.report.session_id,
                    cohort=self.report.cohort,
                    axes=axes,
                    accepted_photos=self.report.accepted_photos,
                    instrument_fingerprint=self.report.instrument_fingerprint,
                    photo_fingerprint=self.report.photo_fingerprint,
                )

                page = render_results(report, csrf_token="csrf-token")

                self.assertNotIn("Harmony remains a component profile", page)
                self.assertIn("A harmony result is not available yet", page)

    def test_unvalidated_harmony_profile_suppresses_legacy_score_summary(self) -> None:
        axes = dict(self.report.axes)
        axes["harmony"] = AxisResult.measured_profile(
            axis="harmony",
            summary="Geometric harmony index: 38/100.",
            cohort=self.cohort,
            photo_count=7,
            details=axes["harmony"].details,
        )
        report = AnalysisReport(
            session_id=self.report.session_id,
            cohort=self.report.cohort,
            axes=axes,
            accepted_photos=self.report.accepted_photos,
            instrument_fingerprint=self.report.instrument_fingerprint,
            photo_fingerprint=self.report.photo_fingerprint,
        )

        page = render_results(report, csrf_token="csrf-token")

        self.assertNotIn("38/100", page)
        self.assertNotIn("Geometric harmony index", page)
        self.assertIn("Measured component geometry is shown below", page)

    def test_unresolved_averageness_leads_with_range_and_demotes_point(self) -> None:
        axes = dict(self.report.axes)
        axes["averageness"] = AxisResult.measured(
            axis="averageness",
            summary=(
                "Measured range: 15th–37th percentile. "
                "The range crosses a placement boundary, so placement is unresolved."
            ),
            value=25.7,
            unit="typicality percentile",
            percentile=25.7,
            interval=(14.8, 36.7),
            cohort=self.cohort,
            photo_count=5,
            details={
                "placement": "withheld — the interval straddles a placement cutoff"
            },
        )
        report = AnalysisReport(
            session_id=self.report.session_id,
            cohort=self.report.cohort,
            axes=axes,
            accepted_photos=self.report.accepted_photos,
            instrument_fingerprint=self.report.instrument_fingerprint,
            photo_fingerprint=self.report.photo_fingerprint,
        )

        page = render_results(report, csrf_token="csrf-token")

        self.assertIn("Estimated range: 15th–37th percentile", page)
        self.assertIn("Placement unresolved", page)
        self.assertIn("Point estimate: 26th percentile", page)
        self.assertLess(
            page.index("Estimated range: 15th–37th percentile"),
            page.index("Point estimate: 26th percentile"),
        )

    def test_results_offer_more_photos_without_a_maximum(self) -> None:
        page = render_results(self.report, csrf_token="csrf-token")

        self.assertIn("8 photos in this session", page)
        self.assertIn("Add more photos", page)
        self.assertRegex(page, r'<input[^>]+type="file"[^>]+multiple')
        self.assertNotIn("maximum 7", page.lower())

    def test_results_have_composite_without_clinical_assessment(self) -> None:
        page = render_results(self.report, csrf_token="csrf-token")
        visible = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", page)).lower()

        disclaimer = "the overall score uses an explicit geometric formula"
        self.assertIn(disclaimer, visible)
        self.assertIn("overall geometry score", visible)

        for phrase in (
            "beauty score",
            "acne",
            "skin quality",
            "style advice",
            "medical diagnosis",
        ):
            with self.subTest(phrase=phrase):
                self.assertNotIn(phrase, visible)

    def test_every_user_controlled_field_is_escaped(self) -> None:
        attack = '<img src=x onerror="alert(1)">'
        cohort = CohortUse(
            requested_id="requested",
            used_id="used",
            fallback_level="exact",
            sample_size=200,
            requested_label=attack,
            used_label=attack,
            member_ids=("used",),
        )
        axes = {
            axis: AxisResult.measured(
                axis=axis,
                summary=attack,
                value=1.0,
                unit=attack,
                percentile=50.0,
                interval=(40.0, 60.0),
                cohort=cohort,
                photo_count=7,
                details={"label": attack},
            )
            for axis in ("harmony", "dimorphism", "symmetry", "averageness")
        }
        report = AnalysisReport(
            session_id=attack,
            cohort=cohort,
            axes=axes,
            accepted_photos=7,
            instrument_fingerprint="instrument",
            photo_fingerprint="photos",
        )

        page = render_results(report, csrf_token=attack)

        self.assertNotIn(attack, page)
        self.assertIn(html.escape(attack), page)
        self.assertNotIn("onerror=\"alert(1)\"", page)

    def test_live_technical_details_allowlist_excludes_private_fields(self) -> None:
        private_path = "<local-path>"
        axes = dict(self.report.axes)
        axes["dimorphism"] = AxisResult.measured(
            axis="dimorphism",
            summary="Dimorphism measured.",
            value=72.0,
            unit="% alignment probability",
            percentile=61.0,
            interval=(54.0, 68.0),
            cohort=self.cohort,
            photo_count=7,
            details={
                "features_used": ["jaw_width"],
                "source_path": private_path,
                "display_name": "original-portrait.jpg",
                "sha256": "a" * 64,
                "debug_path": private_path,
                "future_internal_field": "must-not-render",
            },
        )
        report = AnalysisReport(
            session_id=self.report.session_id,
            cohort=self.report.cohort,
            axes=axes,
            accepted_photos=self.report.accepted_photos,
            instrument_fingerprint=self.report.instrument_fingerprint,
            photo_fingerprint=self.report.photo_fingerprint,
        )

        page = render_results(report, csrf_token="csrf-token")

        self.assertIn("features_used", page)
        for forbidden in (
            private_path,
            "original-portrait.jpg",
            "a" * 64,
            "debug_path",
            "must-not-render",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, page)

    def test_static_assets_have_accessible_responsive_basics(self) -> None:
        static = Path(__file__).resolve().parents[1] / "local_app" / "static"
        css = (static / "app.css").read_text(encoding="utf-8")
        script = (static / "app.js").read_text(encoding="utf-8")

        self.assertIn(":focus-visible", css)
        self.assertIn("prefers-reduced-motion", css)
        self.assertIn("@media", css)
        self.assertIn(".results-overview", css)
        self.assertIn(".overview-spine::before", css)
        self.assertIn(".measurement-track", css)
        self.assertIn(".axis-card--harmony", css)
        self.assertIn(".method-and-limits", css)
        self.assertRegex(
            css,
            r"(?s)@media \(max-width: 430px\).*?\.overview-item \{.*?grid-template-columns: 1fr;",
        )
        self.assertIn("files.length", script)
        self.assertNotIn("fetch(", script)


if __name__ == "__main__":
    unittest.main()
