"""Deterministic opt-in styling: actual facts, bounded rules, no score prediction."""

import unittest
from copy import deepcopy

from local_app.consultation import (
    consultation_payload,
    preferences,
    render_consultation,
)
from local_app.recommendations import RULES, SOURCES, select_recommendations


def observations(face=90, forehead=85, jaw=85):
    return [
        dict(id=k, label=k, value=v, percentile=p, unit="ratio")
        for k, v, p in (
            ("facial_index", 1.4, face),
            ("forehead_ratio", 0.9, forehead),
            ("jaw_ratio", 0.8, jaw),
            ("eye_spacing", 0.3, 60),
        )
    ]


class RecommendationTests(unittest.TestCase):
    def settings(self, **fields):
        return preferences(
            {"goal": "balance", "hair": "yes", "eyewear": "yes", **fields}
        )

    def test_catalogue_is_versioned_and_every_source_resolves(self):
        self.assertEqual(len({r.id for r in RULES}), len(RULES))
        for rule in RULES:
            self.assertTrue(rule.version)
            self.assertTrue(rule.required)
            self.assertTrue(rule.alternative)
            self.assertTrue(all(s in SOURCES for s in rule.sources))

    def test_identical_input_deterministic_no_measurement_mutation(self):
        rows = observations()
        before = deepcopy(rows)
        settings = self.settings()
        a = select_recommendations(rows, settings)
        self.assertEqual(a, select_recommendations(rows, settings))
        self.assertEqual(rows, before)
        self.assertLessEqual(len(a["suggestions"]), 5)
        for suggestion in a["suggestions"]:
            self.assertIn("observations", suggestion)
            self.assertIn("alternative", suggestion)
            self.assertIn("styling convention", suggestion["evidence"].lower())
            self.assertNotIn("score_gain", suggestion)

    def test_opt_out_preserve_and_no_evidence_are_respected(self):
        self.assertEqual(select_recommendations([], self.settings())["suggestions"], [])
        self.assertEqual(
            select_recommendations(observations(), preferences({}))["suggestions"], []
        )
        result = select_recommendations(
            observations(), self.settings(hair="no", eyewear="no")
        )
        self.assertEqual(result["suggestions"], [])
        result = select_recommendations(
            observations(), self.settings(preserve="outline")
        )
        self.assertFalse(any(s["category"] == "hair" for s in result["suggestions"]))
        self.assertIn("preserved_region", {r["reason"] for r in result["excluded"]})

    def test_boundary_is_two_way_experiment_instead_of_opposing_advice(self):
        for p in (24, 26, 74, 76):
            selected = select_recommendations(
                observations(face=p), self.settings(eyewear="no")
            )
            self.assertIn(
                "hair.part.compare", {r["id"] for r in selected["suggestions"]}
            )
            self.assertFalse(
                any(r["id"].startswith("hair.volume.") for r in selected["suggestions"])
            )

    def test_goal_changes_direction_without_changing_measurement(self):
        rows = observations()
        before = deepcopy(rows)
        a = select_recommendations(rows, self.settings())
        b = select_recommendations(rows, self.settings(goal="emphasize"))
        self.assertNotEqual(
            a["suggestions"][0]["action"], b["suggestions"][0]["action"]
        )
        self.assertEqual(rows, before)

    def test_constraints_do_not_promote_infeasible_cut(self):
        settings = self.settings(
            budget="no_purchase", maintenance="low", hair_change="keep"
        )
        self.assertNotIn(
            "hair.fringe.consult",
            {
                r["id"]
                for r in select_recommendations(observations(), settings)["suggestions"]
            },
        )
        settings = self.settings(
            budget="flexible",
            maintenance="flexible",
            hair_change="open",
            hair_length="long",
            hair_texture="wavy",
        )
        result = select_recommendations(observations(), settings)
        suggestion = next(
            r for r in result["suggestions"] if r["id"] == "hair.fringe.consult"
        )
        self.assertEqual(len(suggestion["observations"]), 2)
        self.assertIn("wavy", suggestion["basis"])

    def test_unknown_methods_nonfinite_and_duplicate_inputs_not_used(self):
        for patch in (
            {"value": float("nan")},
            {"percentile": float("inf")},
            {"method": "unrelated3D"},
        ):
            rows = observations()
            rows[0].update(patch)
            self.assertFalse(
                any(
                    r["id"].startswith("hair.")
                    for r in select_recommendations(rows, self.settings())[
                        "suggestions"
                    ]
                )
            )
        rows = observations()
        rows.append(dict(rows[0]))
        self.assertFalse(
            any(
                r["id"].startswith("hair.")
                for r in select_recommendations(rows, self.settings())["suggestions"]
            )
        )

    def test_facial_hair_never_inferred_and_sources_escape(self):
        a = consultation_payload(observations(), self.settings())
        self.assertFalse(any(s["category"] == "facial_hair" for s in a["suggestions"]))
        a = consultation_payload(observations(), self.settings(facial_hair="yes"))
        self.assertTrue(any(s["category"] == "facial_hair" for s in a["suggestions"]))
        a["suggestions"][0]["alternative"] = "<img src=x onerror=bad()>"
        page = render_consultation(a)
        self.assertNotIn("<img", page)
        self.assertIn("&lt;img", page)
        self.assertIn("Why this was selected", page)

    def test_preferences_reject_unbounded_or_unknown_values(self):
        for settings in (
            {"maintenance": []},
            {"preserve": "ethnicity"},
            {"budget": "luxury"},
            {"hair_texture": "<script>"},
        ):
            with self.assertRaises(ValueError):
                preferences(settings)

    def test_bankless_reading_offers_comparison_without_inventing_a_rank(self):
        rows = [dict(id="facial_index", value=1.4, percentile=None)]
        result = select_recommendations(rows, self.settings())
        self.assertEqual([r["id"] for r in result["suggestions"]], ["hair.raw.compare"])
        self.assertIn(
            "reference comparison unavailable", result["suggestions"][0]["basis"]
        )
        self.assertNotIn("percentile 0", result["suggestions"][0]["basis"])

    def test_wrong_units_or_method_cannot_be_relabelled_by_report_boundary(self):
        from local_app.product_report import public_regions
        from tests.test_consumer_evaluation import report

        for patch in ({"unit": "mm"}, {"method": "unrelated3D"}):
            source = report()
            source.axes["harmony"].details["regional_measurements"][0].update(patch)
            rows = public_regions(source.axes)
            self.assertFalse(consultation_payload(rows, self.settings())["suggestions"])
