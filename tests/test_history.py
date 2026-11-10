"""Encrypted opt-in history, authenticated backups and contract-aware comparisons."""

from pathlib import Path
import os
import tempfile
import unittest
from local_app.history import ArchiveStore, ArchiveError, MAX_RECORD_BYTES
from local_app.archive_schema import Snapshot, freeze_report, compare
from tests.test_consumer_evaluation import report
from local_app.contracts import Demographics


class Keys:
    def __init__(self):
        self.key = None
        self.locked = False

    def get(self):
        if self.locked:
            raise ArchiveError("Keychain unavailable.")
        return self.key

    def create(self, key):
        if self.key is not None:
            raise ArchiveError("Already initialized.")
        self.key = key


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.keys = Keys()
        self.store = ArchiveStore(Path(self.tmp.name) / "history", self.keys)
        self.snapshot = freeze_report(
            report(),
            Demographics(25, "male", ("white",)),
            profile="Test",
            protocol="controlled",
            camera="Test camera fixed setup",
            reference="a" * 64,
        )

    def test_opt_in_encrypted_immutable_roundtrip_rename_evidence_and_delete(self):
        self.assertEqual(self.store.list(), [])
        self.assertIsNone(self.keys.key)
        record = self.store.save(self.snapshot)
        rid = record.id
        raw = (self.store.root / (rid + ".fae")).read_bytes()
        self.assertNotIn(b"Test", raw)
        self.assertNotIn(b"instrument-fixture", raw)
        self.assertEqual(self.store.read(rid).snapshot, self.snapshot)
        self.store.rename(rid, "New name")
        self.assertEqual(self.store.read(rid).snapshot, self.snapshot)
        self.assertEqual(self.store.read(rid).name, "New name")
        self.assertEqual(os.stat(self.store.root).st_mode & 0o777, 0o700)
        self.assertEqual(
            os.stat(self.store.root / (rid + ".fae")).st_mode & 0o777, 0o600
        )
        self.store.delete(rid)
        self.assertEqual(self.store.list(), [])

    def test_locked_missing_wrong_key_tamper_truncated_path_and_symlink_fail_closed(
        self,
    ):
        rid = self.store.save(self.snapshot).id
        path = self.store.root / (rid + ".fae")
        original = path.read_bytes()
        for key in (None, b"0" * 32):
            self.keys.key = key
            with self.assertRaises(ArchiveError):
                self.store.read(rid)
        self.assertEqual(path.read_bytes(), original)
        self.keys.key = None
        with self.assertRaises(ArchiveError):
            self.store.save(self.snapshot)
        self.keys.key = b"1" * 32
        for raw in (original[:-1], b"bad", original[:-1] + bytes([original[-1] ^ 1])):
            path.write_bytes(raw)
            with self.assertRaises(ArchiveError):
                self.store.read(rid)
        for bad in ("../outside", "", rid + "/x"):
            with self.assertRaises(ArchiveError):
                self.store.read(bad)
        path.unlink()
        target = Path(self.tmp.name) / "outside"
        target.write_bytes(original)
        path.symlink_to(target)
        with self.assertRaises(ArchiveError):
            self.store.read(rid)

    def test_backup_recovers_on_different_key_and_rejects_password_tamper_schema_size(
        self,
    ):
        rid = self.store.save(self.snapshot).id
        backup = self.store.backup(rid, "a sufficiently long password")
        other = ArchiveStore(Path(self.tmp.name) / "other", Keys())
        restored = other.import_backup(backup, "a sufficiently long password")
        self.assertNotEqual(restored.id, rid)
        self.assertEqual(restored.snapshot, self.snapshot)
        self.assertTrue(restored.imported_at)
        for value, password in (
            (backup, "wrong password long"),
            (backup[:-1], "a sufficiently long password"),
            (b"x" * (MAX_RECORD_BYTES + 1000), "a sufficiently long password"),
        ):
            with self.assertRaises(ArchiveError):
                other.import_backup(value, password)
        with self.assertRaises(ArchiveError):
            self.store.backup(rid, "short")
        value = self.snapshot.model_dump()
        value["html"] = "<script>alert(1)</script>"
        with self.assertRaises(ValueError):
            Snapshot.model_validate(value)

    def test_quota_and_failed_replace_preserve_original(self):
        from unittest.mock import patch

        rid = self.store.save(self.snapshot).id
        original = (self.store.root / (rid + ".fae")).read_bytes()
        with patch("local_app.history.os.replace", side_effect=OSError("interrupted")):
            with self.assertRaises(ArchiveError):
                self.store.rename(rid, "No")
        self.assertEqual((self.store.root / (rid + ".fae")).read_bytes(), original)
        self.assertEqual(list(self.store.root.glob("*.tmp")), [])
        with patch("local_app.history.MAX_RECORDS", 1):
            with self.assertRaises(ArchiveError):
                self.store.save(self.snapshot)

    def test_compare_exact_dependencies_unknown_capture_duplicates_and_partial_rows(
        self,
    ):
        a = self.snapshot
        b = a.model_copy(update={"input_digest": "b" * 64})
        self.assertEqual(compare(a, b)["status"], "compatible")
        self.assertEqual(compare(a, b)["score_delta"], 0)
        self.assertTrue(compare(a, a)["same_inputs"])
        for field, value in (
            ("formula_contract", "changed"),
            ("reference", "b" * 64),
            ("cohort", "changed"),
            ("harmony_ids", []),
            ("instrument", "new"),
            ("routes", []),
        ):
            c = b.model_copy(update={field: value})
            r = compare(a, c)
            self.assertIsNone(r["score_delta"])
            self.assertIn(field, r["differences"])
        self.assertEqual(
            compare(a, b.model_copy(update={"protocol": "unspecified"}))["status"],
            "insufficient_evidence",
        )
        self.assertIsNone(
            compare(a, b.model_copy(update={"profile": "Another person"}))[
                "score_delta"
            ]
        )

    def test_import_arithmetic_provenance_preferences_and_formula_guard(self):
        for field, value in (
            ("score", 100.0),
            ("reference", ""),
            ("preferences", {"goal": "invalid"}),
        ):
            changed = self.snapshot.model_dump()
            changed[field] = value
            with self.assertRaises(ValueError):
                Snapshot.model_validate(changed)
        changed = self.snapshot.model_dump()
        changed.update(formula="unsupported-formula", score=100.0)
        unknown = Snapshot.model_validate(changed)
        self.assertIsNone(compare(unknown, unknown)["score_delta"])
        self.assertEqual(compare(unknown, unknown)["status"], "method_changed")
        a = self.snapshot.model_copy(
            update={"captured_at": "2026-01-01T10:00:00+02:00"}
        )
        b = self.snapshot.model_copy(
            update={"captured_at": "2026-01-01T09:00:00+00:00"}
        )
        self.assertFalse(any("reversed" in s for s in compare(a, b)["warnings"]))
        from local_app.contracts import SEXES

        for sex in SEXES:
            value = freeze_report(
                report(), Demographics(25, sex, ("white",)), profile="Test"
            )
            self.assertEqual(value.demographics.sex, sex)
