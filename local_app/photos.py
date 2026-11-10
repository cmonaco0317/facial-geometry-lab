"""Bounded local photo normalization and per-session cataloging."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import os
from pathlib import Path
import secrets
from typing import Any

import cv2
import numpy as np

from .sessions import Session, SessionStore

MIN_PHOTOS = 1
MAX_FILE_BYTES = 10_000_000
MAX_PIXELS = 80_000_000
MAX_SESSION_BYTES = 2 * 1024 * 1024 * 1024


@dataclass(frozen=True)
class PhotoLimits:
    max_file_bytes: int = MAX_FILE_BYTES
    max_pixels: int = MAX_PIXELS
    max_session_bytes: int = MAX_SESSION_BYTES
    minimum_photos: int = MIN_PHOTOS

    def __post_init__(self) -> None:
        for field_name in (
            "max_file_bytes",
            "max_pixels",
            "max_session_bytes",
            "minimum_photos",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive whole number")


@dataclass(frozen=True)
class PhotoEntry:
    photo_id: str
    stored_name: str
    display_name: str
    sha256: str
    format: str
    width: int
    height: int
    source_bytes: int
    stored_bytes: int

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PhotoEntry":
        return cls(**{name: payload[name] for name in cls.__dataclass_fields__})


@dataclass(frozen=True)
class PhotoRejection:
    display_name: str
    code: str
    message: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PhotoRejection":
        return cls(**{name: payload[name] for name in cls.__dataclass_fields__})


@dataclass(frozen=True)
class PhotoReadiness:
    ready: bool
    accepted: int
    minimum: int
    message: str


class PhotoRejected(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class PhotoCatalog:
    """Decode once, normalize once, and keep user names out of filesystem paths."""

    def __init__(
        self,
        store: SessionStore,
        session_id: str,
        *,
        limits: PhotoLimits | None = None,
    ) -> None:
        self.store = store
        self.session_id = session_id
        self.limits = limits or PhotoLimits()

    def add_bytes(self, original_name: str, content: bytes) -> PhotoEntry:
        display_name = self._display_name(original_name)
        try:
            raw = bytes(content)
        except (TypeError, ValueError) as error:
            return self._reject(display_name, "malformed", "file data is not readable", error)

        if not raw:
            return self._reject(display_name, "empty", "could not decode an empty image file")
        if len(raw) > self.limits.max_file_bytes:
            return self._reject(
                display_name,
                "file_too_large",
                f"file size exceeds the {self.limits.max_file_bytes:,}-byte per-image limit",
            )

        session = self.store.load(self.session_id)
        photos = self._photo_payloads(session)
        source_bytes = sum(int(item["source_bytes"]) for item in photos)
        if source_bytes + len(raw) > self.limits.max_session_bytes:
            return self._reject(
                display_name,
                "session_too_large",
                "session storage would exceed the 2 GiB limit",
                session=session,
            )

        digest = hashlib.sha256(raw).hexdigest()
        if any(item["sha256"] == digest for item in photos):
            return self._reject(
                display_name,
                "duplicate",
                "duplicate image bytes were already accepted",
                session=session,
            )

        source_format = self._sniff_format(raw)
        dimensions = self._declared_dimensions(raw, source_format)
        if dimensions is None:
            return self._reject(
                display_name,
                "decode_failed",
                "could not decode this file as a supported JPEG or PNG image",
                session=session,
            )
        declared_width, declared_height = dimensions
        if declared_height * declared_width > self.limits.max_pixels:
            return self._reject(
                display_name,
                "too_many_pixels",
                "decoded image exceeds the 80 megapixels per-image limit",
                session=session,
            )

        encoded_input = np.frombuffer(raw, dtype=np.uint8)
        image = cv2.imdecode(encoded_input, cv2.IMREAD_COLOR)
        if image is None or image.ndim != 3:
            return self._reject(
                display_name,
                "decode_failed",
                "could not decode this file as an image",
                session=session,
            )

        height, width = (int(value) for value in image.shape[:2])
        if height <= 0 or width <= 0:
            return self._reject(
                display_name,
                "decode_failed",
                "could not decode this file as an image",
                session=session,
            )
        if height * width > self.limits.max_pixels:
            return self._reject(
                display_name,
                "too_many_pixels",
                "decoded image exceeds the 80 megapixels per-image limit",
                session=session,
            )

        ok, normalized = cv2.imencode(
            ".jpg",
            image,
            [int(cv2.IMWRITE_JPEG_QUALITY), 95],
        )
        if not ok:
            return self._reject(
                display_name,
                "normalize_failed",
                "the image decoded but could not be normalized",
                session=session,
            )

        photo_id = digest[:16]
        stored_name = f"photo-{photo_id}.jpg"
        stored_path = self.store.path_for(self.session_id, "uploads", stored_name)
        if stored_path.exists():
            return self._reject(
                display_name,
                "identifier_collision",
                "the image could not be assigned a unique local identifier",
                session=session,
            )

        normalized_bytes = normalized.tobytes()
        self._write_private(stored_path, normalized_bytes)
        entry = PhotoEntry(
            photo_id=photo_id,
            stored_name=stored_name,
            display_name=display_name,
            sha256=digest,
            format=source_format,
            width=width,
            height=height,
            source_bytes=len(raw),
            stored_bytes=len(normalized_bytes),
        )
        photos.append(asdict(entry))
        session.state["photos"] = photos
        try:
            self.store.save(session)
        except Exception:
            stored_path.unlink(missing_ok=True)
            raise
        return entry

    def entries(self) -> tuple[PhotoEntry, ...]:
        session = self.store.load(self.session_id)
        return tuple(PhotoEntry.from_dict(item) for item in self._photo_payloads(session))

    def rejections(self) -> tuple[PhotoRejection, ...]:
        session = self.store.load(self.session_id)
        payloads = session.state.get("photo_rejections", [])
        if not isinstance(payloads, list):
            raise ValueError("photo rejection state is invalid")
        return tuple(PhotoRejection.from_dict(item) for item in payloads)

    def readiness(self) -> PhotoReadiness:
        accepted = len(self.entries())
        missing = max(0, self.limits.minimum_photos - accepted)
        if not missing:
            message = "Ready to analyze"
        elif missing == 1:
            message = "Add 1 more photo"
        else:
            message = f"Add {missing} more photos"
        return PhotoReadiness(
            ready=accepted >= self.limits.minimum_photos,
            accepted=accepted,
            minimum=self.limits.minimum_photos,
            message=message,
        )

    @staticmethod
    def _photo_payloads(session: Session) -> list[dict[str, Any]]:
        payloads = session.state.get("photos", [])
        if not isinstance(payloads, list):
            raise ValueError("photo catalog state is invalid")
        return list(payloads)

    def _reject(
        self,
        display_name: str,
        code: str,
        message: str,
        cause: Exception | None = None,
        *,
        session: Session | None = None,
    ) -> PhotoEntry:
        current = session or self.store.load(self.session_id)
        payloads = current.state.get("photo_rejections", [])
        if not isinstance(payloads, list):
            payloads = []
        payloads = list(payloads)
        payloads.append(asdict(PhotoRejection(display_name, code, message)))
        current.state["photo_rejections"] = payloads
        self.store.save(current)
        error = PhotoRejected(code, message)
        if cause is not None:
            raise error from cause
        raise error

    @staticmethod
    def _display_name(original_name: str) -> str:
        if not isinstance(original_name, str):
            return "unnamed image"
        normalized = original_name.replace("\\", "/")
        name = Path(normalized).name.strip()
        return name[:120] or "unnamed image"

    @staticmethod
    def _sniff_format(raw: bytes) -> str:
        if raw.startswith(b"\xff\xd8\xff"):
            return "jpeg"
        if raw.startswith(b"\x89PNG\r\n\x1a\n"):
            return "png"
        if raw.startswith((b"II*\x00", b"MM\x00*")):
            return "tiff"
        if raw.startswith(b"RIFF") and raw[8:12] == b"WEBP":
            return "webp"
        return "unknown"

    @classmethod
    def _declared_dimensions(cls, raw: bytes, source_format: str) -> tuple[int, int] | None:
        if source_format == "png":
            if len(raw) < 24 or raw[12:16] != b"IHDR":
                return None
            width = int.from_bytes(raw[16:20], "big")
            height = int.from_bytes(raw[20:24], "big")
            return (width, height) if width > 0 and height > 0 else None
        if source_format == "jpeg":
            return cls._jpeg_dimensions(raw)
        return None

    @staticmethod
    def _jpeg_dimensions(raw: bytes) -> tuple[int, int] | None:
        if len(raw) < 4 or not raw.startswith(b"\xff\xd8"):
            return None
        start_of_frame = {
            0xC0,
            0xC1,
            0xC2,
            0xC3,
            0xC5,
            0xC6,
            0xC7,
            0xC9,
            0xCA,
            0xCB,
            0xCD,
            0xCE,
            0xCF,
        }
        index = 2
        while index + 3 < len(raw):
            if raw[index] != 0xFF:
                index += 1
                continue
            while index < len(raw) and raw[index] == 0xFF:
                index += 1
            if index >= len(raw):
                return None
            marker = raw[index]
            index += 1
            if marker in {0x01, *range(0xD0, 0xDA)}:
                continue
            if index + 2 > len(raw):
                return None
            segment_length = int.from_bytes(raw[index : index + 2], "big")
            if segment_length < 2 or index + segment_length > len(raw):
                return None
            if marker in start_of_frame:
                if segment_length < 7:
                    return None
                height = int.from_bytes(raw[index + 3 : index + 5], "big")
                width = int.from_bytes(raw[index + 5 : index + 7], "big")
                return (width, height) if width > 0 and height > 0 else None
            index += segment_length
        return None

    @staticmethod
    def _write_private(path: Path, content: bytes) -> None:
        temporary = path.with_name(f".{path.name}-{secrets.token_hex(6)}.tmp")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            os.chmod(path, 0o600)
        finally:
            if temporary.exists():
                temporary.unlink()
