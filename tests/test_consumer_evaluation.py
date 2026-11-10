"""Consumer repeatability includes estimates without pooling incompatible scores."""

import json
import unittest
from copy import deepcopy
from dataclasses import replace

from local_app.analysis import AnalysisReport
from local_app.consumer_evaluation import snapshot, summarize_consumer
from local_app.contracts import AxisResult, CohortUse


def report(status="estimated", ratio_ids=("vertical_balance", "lip_balance")):
    cohort = CohortUse("m_white_20_29", "m_white_20_29", "exact", 684)
    axes = {
        a: AxisResult(
            axis=a,
            status=status,
            cohort=cohort,
            photo_count=1,
            value=2 if a == "symmetry" else 60,
            unit="% IPD asymmetry"
            if a == "symmetry"
            else "typicality percentile"
            if a == "averageness"
            else "% alignment probability",
            percentile=60,
            interval=(50, 70) if status == "measured" else None,
            summary="Fixture",
            details={"private_path": "/private/person/photo.jpg"},
        )
        for a in ("harmony", "dimorphism", "symmetry", "averageness")
    }
    axes["harmony"] = AxisResult(
        axis="harmony",
        status=status,
        cohort=cohort,
        photo_count=1,
        headline_available=False,
        summary="Fixture",
        details={
            "profile": {
                k: dict(
                    value=0.5,
                    percentile=50,
                    interval=(40, 60) if status == "measured" else None,
                    label="private",
                )
                for k in ratio_ids
            },
            "regional_measurements": [
                dict(
                    id="facial_index",
                    label="private",
                    value=1.4,
                    percentile=70,
                    unit="ratio",
                )
            ],
            "capture_issues": {"yaw": 1, "/private/person": 1},
        },
    )
    return AnalysisReport(
        "private-session", cohort, axes, 1, "instrument-fixture", "private-photo-digest"
    )


def pair(a, b, status="evaluated"):
    return {
        "status": status,
        "sets": [
            {"status": "analyzed", "consumer": a},
            {"status": "analyzed", "consumer": b},
        ],
    }


class ConsumerSnapshotTests(unittest.TestCase):
    def test_corrected_symmetry_unit_survives_snapshot_and_keeps_legacy_readable(self):
        source = report()
        previous = snapshot(source, reference="a" * 64)
        corrected = replace(source, axes={
            **source.axes,
            "symmetry": replace(source.axes["symmetry"], unit="% outer-eye span asymmetry"),
        })
        current = snapshot(corrected, reference="a" * 64)
        self.assertEqual(current["axes"]["symmetry"]["unit"], "% outer-eye span asymmetry")
        self.assertEqual(previous["axes"]["symmetry"]["unit"], "% IPD asymmetry")
        self.assertEqual(current["composite"], previous["composite"])

    def test_snapshot_exact_inputs_and_private_allowlist(self):
        value = snapshot(report(), reference="a" * 64)
        self.assertEqual(
            value["composite"]["inputs"]["harmony_ratios"],
            ["lip_balance", "vertical_balance"],
        )
        self.assertEqual(value["axes"]["symmetry"]["status"], "estimated")
        self.assertEqual(value["axes"]["harmony"]["capture_issues"], {"yaw": 1})
        for private in (
            "private-session",
            "private-photo",
            "/private/person",
            "photo.jpg",
        ):
            self.assertNotIn(private, json.dumps(value))

    def test_estimated_pairs_count_and_missing_stays_missing(self):
        a = snapshot(report(), reference="a" * 64)
        b = deepcopy(a)
        b["axes"]["symmetry"]["value"] = 3
        summary = summarize_consumer(
            [pair(a, b), {"status": "duplicate_evidence", "sets": []}]
        )
        axis = summary["axes"]["symmetry"]
        self.assertEqual(axis["pairs"], 1)
        self.assertEqual(axis["coverage"], 0.5)
        self.assertEqual(
            axis["strata"]["estimated/estimated"]["value_movement"]["median"], 1
        )
        self.assertEqual(summary["composite"]["movement"]["n_identities"], 1)
        a["axes"]["symmetry"]["value"] = None
        b["axes"]["symmetry"]["value"] = None
        self.assertIsNone(
            summarize_consumer([pair(a, b)])["axes"]["symmetry"]["value_movement"][
                "median"
            ]
        )

    def test_ratio_subset_and_reference_changes_block_score_comparison(self):
        a = snapshot(report(), reference="a" * 64)
        for b, reason in (
            (
                snapshot(report(ratio_ids=("vertical_balance",)), reference="a" * 64),
                "harmony_ratios_changed",
            ),
            (snapshot(report(), reference="b" * 64), "reference_changed"),
        ):
            summary = summarize_consumer([pair(a, b)])
            self.assertEqual(summary["composite"]["pairs"], 0)
            self.assertEqual(summary["composite"]["incompatibilities"][reason], 1)

    def test_mixed_route_has_its_own_stratum_and_not_a_progress_score(self):
        a = snapshot(report(), reference="a" * 64)
        b = snapshot(report("measured"), reference="a" * 64)
        summary = summarize_consumer([pair(a, b)])
        self.assertEqual(
            summary["axes"]["symmetry"]["strata"]["estimated/measured"]["pairs"], 1
        )
        self.assertEqual(summary["composite"]["pairs"], 0)
        self.assertEqual(summary["composite"]["route_transition"]["n_identities"], 1)

    def test_annotation_study_does_not_mislabel_automatic_zero_movement(self):
        a = snapshot(report(), reference="a" * 64)
        summary = summarize_consumer([pair(a, a)], study_kind="annotation_repeat")
        self.assertEqual(summary["composite"]["pairs"], 0)
        self.assertEqual(summary["axes"]["symmetry"]["pairs"], 0)
        self.assertEqual(summary["scope"], "annotation_repeat")

    def test_bankless_raw_values_can_still_be_compared_without_score(self):
        a = snapshot(report(), reference=None)
        b = deepcopy(a)
        b["regions"][0]["value"] = 1.5
        summary = summarize_consumer([pair(a, b)])
        self.assertEqual(summary["composite"]["pairs"], 0)
        self.assertAlmostEqual(
            summary["regions"]["facial_index"]["value_movement"]["median"], 0.1
        )


class ComparisonGuardTests(unittest.TestCase):
    def test_reference_dependent_values_are_not_raw_geometry(self):
        a = snapshot(report(), reference="a" * 64)
        b = snapshot(report(), reference="b" * 64)
        b["axes"]["averageness"]["value"] = 90
        summary = summarize_consumer([pair(a, b)])
        self.assertEqual(
            summary["axes"]["averageness"]["value_movement"]["n_identities"], 0
        )
        self.assertEqual(
            summary["axes"]["symmetry"]["value_movement"]["n_identities"], 1
        )

    def test_measured_harmony_scalar_unit_survives(self):
        source = report("measured")
        axes = {
            **source.axes,
            "harmony": AxisResult.measured(
                axis="harmony",
                summary="Measured fixture",
                value=60,
                unit="geometric harmony percentile",
                percentile=60,
                interval=(50, 70),
                cohort=source.cohort,
                photo_count=4,
            ),
        }
        measured = AnalysisReport(
            source.session_id,
            source.cohort,
            axes,
            4,
            source.instrument_fingerprint,
            source.photo_fingerprint,
        )
        a = snapshot(measured, reference="a" * 64)
        self.assertEqual(a["axes"]["harmony"]["unit"], "geometric harmony percentile")
        self.assertEqual(
            summarize_consumer([pair(a, a)])["axes"]["harmony"]["value_movement"][
                "n_identities"
            ],
            1,
        )


class AuditSnapshotTests(unittest.TestCase):
    def test_score_components_reconstruct_reported_total_and_keep_ratio_values(self):
        result = snapshot(report(), reference="a" * 64)
        self.assertAlmostEqual(
            sum(c["contribution"] for c in result["score_components"]),
            result["composite"]["value"],
            places=1,
        )
        self.assertEqual(
            {r["id"] for r in result["harmony_components"]},
            {"vertical_balance", "lip_balance"},
        )
        self.assertEqual(result["harmony_components"][0]["value"], 0.5)
        self.assertEqual(
            summarize_consumer([pair(result, result)])["axes"]["harmony"][
                "scalar_pairs"
            ],
            0,
        )

    def test_legacy_validator_rejects_consumer_input_before_source_verification(self):
        from unittest.mock import patch

        from local_app.score_validation import validate_scores

        with patch("local_app.score_validation.verify_source_instrument") as verifier:
            with self.assertRaises(ValueError):
                validate_scores({"version": 2, "consumer_version": 1}, {}, "fixed")
            verifier.assert_not_called()


class RecommendationMovementTests(unittest.TestCase):
    def test_frozen_preferences_record_selected_ids_and_changes(self):
        a = snapshot(
            report(), reference="a" * 64, settings={"goal": "balance", "hair": "yes"}
        )
        b = deepcopy(a)
        self.assertEqual(a["recommendations"]["ids"], ["hair.part.compare"])
        b["recommendations"]["ids"] = ["hair.volume.height"]
        result = summarize_consumer([pair(a, b)])["recommendations"]
        self.assertEqual(result["changed_identity_pairs"], 1)
        self.assertEqual(result["pairs"], 1)
        b["recommendations"]["preferences"]["goal"] = "emphasize"
        self.assertEqual(
            summarize_consumer([pair(a, b)])["recommendations"]["pairs"], 0
        )
