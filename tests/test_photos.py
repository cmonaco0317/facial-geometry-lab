import os
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from local_app.contracts import Demographics
from local_app.photos import PhotoCatalog, PhotoLimits, PhotoRejected
from local_app.sessions import SessionStore


def jpeg_bytes(value: int, *, width: int = 24, height: int = 20) -> bytes:
    image = np.full((height, width, 3), value, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError("test JPEG encoding failed")
    return encoded.tobytes()


def png_bytes(value: int, *, width: int = 24, height: int = 20) -> bytes:
    image = np.full((height, width, 3), value, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("test PNG encoding failed")
    return encoded.tobytes()


class PhotoCatalogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.store = SessionStore(Path(self.temp_dir.name) / "sessions")
        self.session = self.store.create(
            Demographics(age=31, sex="female", ethnicities=("white",))
        )
        self.catalog = PhotoCatalog(self.store, self.session.session_id)

    def test_content_is_decoded_instead_of_trusting_extension(self) -> None:
        entry = self.catalog.add_bytes("not-really-text.txt", png_bytes(30))

        self.assertEqual(entry.format, "png")
        self.assertRegex(entry.stored_name, r"^photo-[a-f0-9]{16}\.jpg$")
        self.assertNotIn("not-really-text", entry.stored_name)
        stored = self.store.path_for(
            self.session.session_id,
            "uploads",
            entry.stored_name,
        )
        self.assertTrue(stored.exists())
        self.assertEqual(stored.stat().st_mode & 0o777, 0o600)

    def test_identical_bytes_are_deduplicated(self) -> None:
        payload = jpeg_bytes(40)
        first = self.catalog.add_bytes("first.jpg", payload)

        with self.assertRaisesRegex(PhotoRejected, "duplicate"):
            self.catalog.add_bytes("second.jpg", payload)

        self.assertEqual(len(self.catalog.entries()), 1)
        self.assertEqual(self.catalog.entries()[0].photo_id, first.photo_id)
        self.assertEqual(self.catalog.rejections()[-1].code, "duplicate")

    def test_more_than_seven_photos_remains_ready(self) -> None:
        for index in range(12):
            self.catalog.add_bytes(f"casual-{index}.jpg", jpeg_bytes(index * 10))

        readiness = self.catalog.readiness()
        self.assertTrue(readiness.ready)
        self.assertEqual(readiness.accepted, 12)
        self.assertEqual(readiness.minimum, 1)

    def test_one_photo_starts_analysis_and_empty_session_waits(self) -> None:
        self.assertFalse(self.catalog.readiness().ready)
        self.assertIn("1 more", self.catalog.readiness().message)
        self.catalog.add_bytes("first.jpg", jpeg_bytes(60))
        self.assertTrue(self.catalog.readiness().ready)

    def test_file_byte_limit_is_enforced_before_decode(self) -> None:
        payload = jpeg_bytes(20)
        catalog = PhotoCatalog(
            self.store,
            self.session.session_id,
            limits=PhotoLimits(max_file_bytes=len(payload) - 1),
        )

        with self.assertRaisesRegex(PhotoRejected, "40 MiB|file size"):
            catalog.add_bytes("large.jpg", payload)

        self.assertEqual(catalog.entries(), ())

    def test_decoded_pixel_limit_is_enforced(self) -> None:
        catalog = PhotoCatalog(
            self.store,
            self.session.session_id,
            limits=PhotoLimits(max_pixels=399),
        )

        with self.assertRaisesRegex(PhotoRejected, "80 megapixels|pixel"):
            catalog.add_bytes("too-many-pixels.jpg", jpeg_bytes(10, width=20, height=20))

    def test_declared_dimensions_are_bounded_before_full_decode(self) -> None:
        forged_png = (
            b"\x89PNG\r\n\x1a\n"
            + (13).to_bytes(4, "big")
            + b"IHDR"
            + (100_000).to_bytes(4, "big")
            + (100_000).to_bytes(4, "big")
            + b"\x08\x02\x00\x00\x00"
            + b"\x00\x00\x00\x00"
        )
        catalog = PhotoCatalog(
            self.store,
            self.session.session_id,
            limits=PhotoLimits(max_pixels=80_000_000),
        )

        with self.assertRaises(PhotoRejected) as caught:
            catalog.add_bytes("bomb.png", forged_png)

        self.assertEqual(caught.exception.code, "too_many_pixels")

    def test_malformed_image_is_rejected_without_losing_good_photos(self) -> None:
        accepted = self.catalog.add_bytes("good.jpg", jpeg_bytes(80))

        with self.assertRaisesRegex(PhotoRejected, "decode"):
            self.catalog.add_bytes("broken.jpg", b"this is not an image")

        self.assertEqual([entry.photo_id for entry in self.catalog.entries()], [accepted.photo_id])

    def test_session_byte_limit_is_cumulative(self) -> None:
        first = jpeg_bytes(10)
        second = jpeg_bytes(220)
        catalog = PhotoCatalog(
            self.store,
            self.session.session_id,
            limits=PhotoLimits(max_session_bytes=len(first) + len(second) - 1),
        )
        catalog.add_bytes("first.jpg", first)

        with self.assertRaisesRegex(PhotoRejected, "2 GiB|session"):
            catalog.add_bytes("second.jpg", second)

        self.assertEqual(len(catalog.entries()), 1)

    @unittest.skipIf(os.name == "nt", "POSIX permission semantics")
    def test_normalized_file_is_private(self) -> None:
        entry = self.catalog.add_bytes("photo.jpg", jpeg_bytes(100))
        path = self.store.path_for(self.session.session_id, "uploads", entry.stored_name)
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)


if __name__ == "__main__":
    unittest.main()
