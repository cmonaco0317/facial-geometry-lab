import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import public_reference as reference


class PublicReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.patch = patch.object(reference, "ROOT", self.root)
        self.patch.start()
        reference.metadata.cache_clear()
        reference._arrays.cache_clear()
        self.cohort = "m_example_20_29"

    def tearDown(self):
        reference.metadata.cache_clear()
        reference._arrays.cache_clear()
        self.patch.stop()
        self.temp.cleanup()

    def write(self, arrays=None, *, count=150, status="available"):
        if arrays is None:
            arrays = {self.cohort + "|ratio|x": np.array([1., 2.]),
                      self.cohort + "|ratio|n": np.array([25, 25])}
        data = self.root / "derived.npz"
        np.savez_compressed(data, **arrays)
        info = {"status": status, "distinct_identity_clusters": count,
                "source_rows": 150, "mean_fit_rows": 75}
        record = {"schema": reference.SCHEMA, "minimum_distinct_identity_clusters": 150,
                  "minimum_cdf_count": 50, "cohorts": {self.cohort: info},
                  "sha256": hashlib.sha256(data.read_bytes()).hexdigest()}
        (self.root / "public_reference.json").write_text(json.dumps(record))

    def test_ecdf_reproduces_strict_ties_and_counts(self):
        self.write()
        pool = reference.pool(self.cohort, "ratio")
        self.assertEqual(len(pool), 50)
        self.assertEqual(float(np.mean(pool < 1)), 0)
        self.assertEqual(float(np.mean(pool < 2)), .5)
        self.assertEqual(reference.metric_names(self.cohort), ["ratio"])
        self.assertIsNone(reference.pool(self.cohort, "unknown"))

    def test_tampering_rejected(self):
        self.write()
        with (self.root / "derived.npz").open("ab") as stream:
            stream.write(b"tamper")
        with self.assertRaisesRegex(ValueError, "checksum"):
            reference.pool(self.cohort, "ratio")

    def test_individual_geometry_and_row_records_rejected(self):
        for name, data in [("landmarks", np.zeros((150, 468, 2))),
                           ("pose", np.zeros((150, 3))),
                           ("identity_cluster_id", np.arange(150))]:
            with self.subTest(name=name):
                self.write({self.cohort + "|" + name: data})
                reference.metadata.cache_clear()
                reference._arrays.cache_clear()
                with self.assertRaises(ValueError):
                    reference._arrays()

    def test_small_and_suppressed_cohorts_rejected(self):
        for count, status in [(149, "available"), (150, "suppressed-small-cohort")]:
            with self.subTest(count=count, status=status):
                self.write(count=count, status=status)
                reference.metadata.cache_clear()
                reference._arrays.cache_clear()
                with self.assertRaises(ValueError):
                    reference._arrays()

    def test_invalid_cdf_pairs_rejected(self):
        cases = [([2., 1.], [25, 25]), ([1., 1.], [25, 25]),
                 ([1., float("nan")], [25, 25]), ([1., 2.], [0, 50]),
                 ([1., 2.], [24, 25]), ([1., 2.], [25., 25.])]
        for values, counts in cases:
            with self.subTest(values=values, counts=counts):
                self.write({self.cohort + "|ratio|x": np.array(values),
                            self.cohort + "|ratio|n": np.array(counts)})
                reference.metadata.cache_clear()
                reference._arrays.cache_clear()
                with self.assertRaises(ValueError):
                    reference._arrays()

    def test_mean_is_cohort_aggregate_and_read_only(self):
        self.write({self.cohort + "|_mean": np.zeros((468, 2)),
                    self.cohort + "|_typicality|x": np.array([.1, .2]),
                    self.cohort + "|_typicality|n": np.array([25, 50])})
        mean, distances, count = reference.average_reference(self.cohort)
        self.assertEqual(count, 150)
        self.assertEqual(len(distances), 75)
        self.assertFalse(mean.flags.writeable)
        self.assertEqual(reference.metric_names(self.cohort), [])

    def test_unknown_typicality_never_substitutes_another_cohort(self):
        self.write()
        with self.assertRaisesRegex(ValueError, "No eligible"):
            reference.average_reference("unknown")

    def test_symlink_metadata_rejected(self):
        self.write()
        target = self.root / "public_reference.json"
        target.rename(self.root / "other.json")
        target.symlink_to(self.root / "other.json")
        with self.assertRaises(ValueError):
            reference.metadata()

    def test_schema_drift_rejected(self):
        self.write()
        target = self.root / "public_reference.json"
        record = json.loads(target.read_text())
        record["minimum_cdf_count"] = 1
        target.write_text(json.dumps(record))
        with self.assertRaises(ValueError):
            reference.metadata()

    def test_alternate_root_is_checked_even_with_cached_valid_checkout(self):
        self.write()
        reference._arrays()
        with tempfile.TemporaryDirectory() as other:
            path = Path(other)
            (path / "public_reference.json").write_text("{}")
            (path / "derived.npz").write_bytes(b"invalid")
            with self.assertRaises(ValueError):
                reference.validate_directory(path)

    def test_overlapping_published_member_sets_rejected(self):
        self.write()
        path = self.root / "public_reference.json"
        record = json.loads(path.read_text())
        record["cohorts"][self.cohort]["members"] = [self.cohort]
        record["cohorts"]["m_example_20_29_to_30_39"] = {
            **record["cohorts"][self.cohort],
            "members": [self.cohort, "m_example_30_39"],
        }
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "overlapping"):
            reference._arrays()

    def test_raw_geometry_sibling_directory_rejected(self):
        self.write()
        path = self.root / self.cohort
        path.mkdir()
        (path / "measurements.npz").write_bytes(b"not for publication")
        with self.assertRaisesRegex(ValueError, "Individual"):
            reference.validate_directory(self.root)


if __name__ == "__main__":
    unittest.main()
