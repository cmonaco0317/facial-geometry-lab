import unittest

from local_app.contracts import AxisResult, CohortUse, Demographics, PhotoRequest


class AxisResultTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cohort = CohortUse(
            requested_id="f_white_20_29",
            used_id="f_white_20_29",
            fallback_level="exact",
            sample_size=532,
        )

    def test_measured_axis_serializes_required_evidence(self) -> None:
        result = AxisResult.measured(
            axis="symmetry",
            summary="More symmetric than 61% of the comparison cohort.",
            value=2.4,
            unit="% IPD asymmetry",
            percentile=61.0,
            interval=(54.0, 68.0),
            cohort=self.cohort,
            photo_count=5,
            details={"eligible_photos": 5},
        )

        payload = result.to_dict()

        self.assertEqual(payload["status"], "measured")
        self.assertEqual(payload["interval"], [54.0, 68.0])
        self.assertEqual(payload["cohort"]["fallback_level"], "exact")
        self.assertIs(payload["cohort"]["exact"], True)
        self.assertEqual(payload["photo_count"], 5)

    def test_needs_photo_requires_actionable_request(self) -> None:
        request = PhotoRequest(
            view="front",
            expression="neutral",
            lighting="even",
            framing="head and shoulders",
            reason="Symmetry needs a straight-on view.",
        )

        result = AxisResult.needs_photo(
            axis="symmetry",
            request=request,
            photo_count=0,
            details={"eligible_photos": 0},
        )

        self.assertEqual(result.to_dict()["photo_request"]["view"], "front")
        self.assertIsNone(result.percentile)

    def test_measured_result_rejects_missing_percentile(self) -> None:
        with self.assertRaisesRegex(ValueError, "percentile"):
            AxisResult(
                axis="averageness",
                status="measured",
                summary="Measured.",
                value=0.12,
                unit="distance",
                percentile=None,
                interval=(40.0, 55.0),
                cohort=self.cohort,
                photo_count=4,
            )

    def test_profile_measurement_does_not_invent_a_headline(self) -> None:
        result = AxisResult.measured_profile(
            axis="harmony",
            summary="Five geometric proportions were measured.",
            cohort=self.cohort,
            photo_count=6,
            details={
                "profile": {
                    "horizontal_balance": {
                        "value": 0.97,
                        "percentile": 54.0,
                        "interval": [46.0, 62.0],
                    }
                }
            },
        )

        payload = result.to_dict()

        self.assertEqual(payload["status"], "measured")
        self.assertIs(payload["headline_available"], False)
        self.assertIsNone(payload["value"])
        self.assertIsNone(payload["percentile"])
        self.assertIn("profile", payload["details"])
        self.assertEqual(AxisResult.from_dict(payload), result)

    def test_profile_measurement_requires_nonempty_profile(self) -> None:
        with self.assertRaisesRegex(ValueError, "profile"):
            AxisResult.measured_profile(
                axis="harmony",
                summary="Measured.",
                cohort=self.cohort,
                photo_count=4,
                details={},
            )

    def test_measured_result_rejects_reversed_interval(self) -> None:
        with self.assertRaisesRegex(ValueError, "interval"):
            AxisResult.measured(
                axis="dimorphism",
                summary="Measured.",
                value=0.7,
                unit="probability",
                percentile=63.0,
                interval=(70.0, 55.0),
                cohort=self.cohort,
                photo_count=4,
            )

    def test_processing_error_requires_safe_message(self) -> None:
        with self.assertRaisesRegex(ValueError, "error"):
            AxisResult(axis="harmony", status="processing_error", summary="")

    def test_every_axis_state_round_trips_through_persisted_json_shape(self) -> None:
        request = PhotoRequest(
            view="front",
            expression="neutral",
            lighting="even",
            framing="head and shoulders",
            reason="Add a straight-on photo.",
        )
        results = (
            AxisResult.measured(
                axis="symmetry",
                summary="Measured.",
                value=2.2,
                unit="% IPD",
                percentile=64.0,
                interval=(58.0, 70.0),
                cohort=self.cohort,
                photo_count=5,
            ),
            AxisResult.needs_photo(
                axis="harmony",
                request=request,
                cohort=self.cohort,
                photo_count=3,
            ),
            AxisResult.processing_error(
                axis="dimorphism",
                error_code="reference_unreadable",
                error_message="The local reference could not be read.",
                retryable=True,
                cohort=self.cohort,
                photo_count=4,
            ),
        )

        for result in results:
            with self.subTest(status=result.status):
                self.assertEqual(AxisResult.from_dict(result.to_dict()), result)


class DemographicsTests(unittest.TestCase):
    def test_subject_must_be_an_adult(self) -> None:
        with self.assertRaisesRegex(ValueError, "18"):
            Demographics(age=17, sex="female", ethnicities=("white",))

    def test_ethnicity_selection_is_normalized_and_deduplicated(self) -> None:
        demographics = Demographics(
            age=31,
            sex="female",
            ethnicities=("White", "white", "Indian"),
        )
        self.assertEqual(demographics.ethnicities, ("white", "indian"))


if __name__ == "__main__":
    unittest.main()
