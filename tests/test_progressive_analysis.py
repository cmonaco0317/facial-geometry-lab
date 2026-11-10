from collections import Counter
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from local_app.analysis import AnalysisNotReady, ProgressiveAnalysisService
from local_app.contracts import AxisResult, CohortUse, Demographics, PhotoRequest
from local_app.photos import PhotoCatalog
from local_app.sessions import SessionStore


def jpeg_bytes(value: int) -> bytes:
    image = np.full((20, 24, 3), value, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError("test JPEG encoding failed")
    return encoded.tobytes()


class FixedResolver:
    def __init__(self, cohort: CohortUse) -> None:
        self.cohort = cohort

    def resolve(self, demographics: Demographics) -> CohortUse:
        return self.cohort


class FakeMeasurementAdapter:
    def __init__(self, instrument: str = "fake-instrument-v1") -> None:
        self.instrument = instrument
        self.measure_calls: Counter[str] = Counter()
        self.percentiles = {
            "harmony": 51.0,
            "dimorphism": 62.0,
            "symmetry": 58.0,
            "averageness": 55.0,
        }
        self.intervals = {
            axis: (value - 7.0, value + 7.0)
            for axis, value in self.percentiles.items()
        }
        self.forced_results: dict[str, AxisResult] = {}

    def fingerprint(self) -> str:
        return self.instrument

    def route(self, photos, cohort):
        routes = {axis: [] for axis in self.percentiles}
        for photo in photos:
            if photo.entry.display_name.startswith("unusable"):
                continue
            if photo.entry.display_name.startswith("symmetry-only"):
                routes["symmetry"].append(photo.entry.photo_id)
                continue
            for axis in routes:
                routes[axis].append(photo.entry.photo_id)
        return {axis: tuple(photo_ids) for axis, photo_ids in routes.items()}

    def measure_axis(self, axis, photos, cohort):
        self.measure_calls[axis] += 1
        if axis in self.forced_results:
            return self.forced_results[axis]
        percentile = self.percentiles[axis]
        interval = self.intervals[axis]
        return AxisResult.measured(
            axis=axis,
            summary=f"{axis} measured",
            value=percentile,
            unit="percentile points",
            percentile=percentile,
            interval=interval,
            cohort=cohort,
            photo_count=len(photos),
            details={"adapter_photo_ids": [photo.entry.photo_id for photo in photos]},
        )


class ProgressiveAnalysisServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.store = SessionStore(Path(self.temp_dir.name) / "sessions")
        self.session = self.store.create(
            Demographics(age=31, sex="female", ethnicities=("white",))
        )
        self.catalog = PhotoCatalog(self.store, self.session.session_id)
        self.cohort = CohortUse(
            requested_id="f_white_30_39",
            used_id="f_white_30_39",
            fallback_level="exact",
            sample_size=412,
            requested_label="White women aged 30-39",
            used_label="White women aged 30-39",
            member_ids=("f_white_30_39",),
        )
        self.resolver = FixedResolver(self.cohort)
        self.adapter = FakeMeasurementAdapter()
        self.service = ProgressiveAnalysisService(
            self.store,
            adapter=self.adapter,
            resolver=self.resolver,
        )

    def add_photos(self, count: int, *, start: int = 0, prefix: str = "casual") -> None:
        for index in range(start, start + count):
            self.catalog.add_bytes(
                f"{prefix}-{index}.jpg",
                jpeg_bytes((index * 23 + 17) % 256),
            )

    def test_first_run_waits_for_one_then_measures_four_axes(self) -> None:
        with self.assertRaisesRegex(AnalysisNotReady, "1 more photo"):
            self.service.analyze(self.session.session_id)

        self.add_photos(1)
        report = self.service.analyze(self.session.session_id)

        self.assertEqual(
            set(report.axes),
            {"harmony", "dimorphism", "symmetry", "averageness"},
        )
        self.assertTrue(all(result.status == "measured" for result in report.axes.values()))
        self.assertEqual(report.accepted_photos, 1)
        self.assertEqual(self.adapter.measure_calls, Counter({axis: 1 for axis in report.axes}))

    def test_repeat_without_changes_uses_persisted_content_cache(self) -> None:
        self.add_photos(7)
        first = self.service.analyze(self.session.session_id)
        calls_after_first = self.adapter.measure_calls.copy()

        second = self.service.analyze(self.session.session_id)

        self.assertEqual(second.to_dict(), first.to_dict())
        self.assertEqual(self.adapter.measure_calls, calls_after_first)

    def test_unusable_addition_cannot_regress_measured_axes(self) -> None:
        self.add_photos(7)
        first = self.service.analyze(self.session.session_id)
        request = PhotoRequest(
            view="straight-on",
            expression="neutral",
            lighting="even",
            framing="head and shoulders",
            reason="Symmetry needs another photo.",
        )
        self.adapter.forced_results["symmetry"] = AxisResult.needs_photo(
            axis="symmetry",
            request=request,
            cohort=self.cohort,
            photo_count=0,
        )
        self.catalog.add_bytes("unusable-blurred.jpg", jpeg_bytes(251))
        calls_after_first = self.adapter.measure_calls.copy()

        second = self.service.analyze(self.session.session_id)

        self.assertEqual(second.axes["symmetry"], first.axes["symmetry"])
        self.assertEqual(self.adapter.measure_calls, calls_after_first)
        self.assertEqual(second.accepted_photos, 8)

    def test_valid_addition_updates_value_and_uncertainty(self) -> None:
        self.add_photos(7)
        first = self.service.analyze(self.session.session_id)
        self.adapter.percentiles["symmetry"] = 72.0
        self.adapter.intervals["symmetry"] = (65.0, 79.0)
        self.catalog.add_bytes("valid-extra.jpg", jpeg_bytes(252))

        second = self.service.analyze(self.session.session_id)

        self.assertEqual(first.axes["symmetry"].percentile, 58.0)
        self.assertEqual(second.axes["symmetry"].percentile, 72.0)
        self.assertEqual(second.axes["symmetry"].interval, (65.0, 79.0))
        self.assertEqual(second.axes["symmetry"].photo_count, 8)

    def test_only_axis_receiving_new_evidence_is_recomputed(self) -> None:
        self.add_photos(7)
        self.service.analyze(self.session.session_id)
        calls_after_first = self.adapter.measure_calls.copy()
        self.catalog.add_bytes("symmetry-only-7.jpg", jpeg_bytes(253))

        self.service.analyze(self.session.session_id)

        self.assertEqual(
            self.adapter.measure_calls["symmetry"],
            calls_after_first["symmetry"] + 1,
        )
        for axis in ("harmony", "dimorphism", "averageness"):
            self.assertEqual(self.adapter.measure_calls[axis], calls_after_first[axis])

    def test_fallback_cohort_is_persisted_on_every_axis(self) -> None:
        fallback = CohortUse(
            requested_id="f_southeast_asian_30_39",
            used_id="f_all_ethnicities_30_39",
            fallback_level="same_age_sex_broader_ethnicity",
            sample_size=1_204,
            requested_label="Southeast Asian women aged 30-39",
            used_label="Women of all available ethnicities aged 30-39",
            member_ids=("f_white_30_39", "f_black_30_39"),
        )
        service = ProgressiveAnalysisService(
            self.store,
            adapter=self.adapter,
            resolver=FixedResolver(fallback),
        )
        self.add_photos(7)

        report = service.analyze(self.session.session_id)

        self.assertEqual(report.cohort, fallback)
        self.assertTrue(
            all(result.cohort == fallback for result in report.axes.values())
        )

    def test_restart_reads_stable_report_without_remeasurement(self) -> None:
        self.add_photos(7)
        first = self.service.analyze(self.session.session_id)
        restarted_adapter = FakeMeasurementAdapter()
        restarted = ProgressiveAnalysisService(
            self.store,
            adapter=restarted_adapter,
            resolver=self.resolver,
        )

        loaded = restarted.analyze(self.session.session_id)

        self.assertEqual(loaded.to_dict(), first.to_dict())
        self.assertEqual(restarted_adapter.measure_calls, Counter())

    def test_instrument_change_recomputes_same_evidence(self) -> None:
        self.add_photos(7)
        self.service.analyze(self.session.session_id)
        calls_after_first = self.adapter.measure_calls.copy()
        self.adapter.instrument = "fake-instrument-v2"

        report = self.service.analyze(self.session.session_id)

        self.assertEqual(report.instrument_fingerprint, "fake-instrument-v2")
        for axis in report.axes:
            self.assertEqual(self.adapter.measure_calls[axis], calls_after_first[axis] + 1)


if __name__ == "__main__":
    unittest.main()
