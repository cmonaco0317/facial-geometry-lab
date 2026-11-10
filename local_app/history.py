"""Opt-in authenticated archives, atomic storage and password-protected recovery."""

from __future__ import annotations
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import re
import secrets
import stat
from threading import RLock
from uuid import uuid4
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from .archive_schema import Evidence, Record, Snapshot, now, snapshot_hash
from .keychain import KeychainKeys, KeychainUnavailable

MAX_RECORD_BYTES = 6_000_000
MAX_RECORDS = 100
MAX_TOTAL_BYTES = 200_000_000
MAGIC = b"FAH1\0"
BACKUP_MAGIC = b"FAB1\0"
ID = re.compile(r"^[a-f0-9]{32}$")


class ArchiveError(ValueError):
    pass


class ArchiveStore:
    def __init__(self, root: Path, keys=None):
        self.root = Path(root).expanduser().absolute()
        if self.root.is_symlink():
            raise ArchiveError("History directory must not be a symlink.")
        self.root.mkdir(parents=True, mode=0o700, exist_ok=True)
        os.chmod(self.root, 0o700)
        self.keys = keys if keys is not None else KeychainKeys(self.root)
        self._lock = RLock()

    @contextmanager
    def _locked(self):
        with self._lock:
            fd = os.open(
                self.root / ".lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600
            )
            try:
                if not stat.S_ISREG(os.fstat(fd).st_mode):
                    raise ArchiveError("Invalid history lock.")
                fcntl.flock(fd, fcntl.LOCK_EX)
                yield
            finally:
                os.close(fd)

    def _paths(self):
        return sorted(self.root.glob("*.fae"))

    def _key(self, create=False):
        try:
            key = self.keys.get()
            if key is None and create and not self._paths():
                key = secrets.token_bytes(32)
                self.keys.create(key)
        except KeychainUnavailable as error:
            raise ArchiveError(str(error)) from error
        if not isinstance(key, bytes) or len(key) != 32:
            raise ArchiveError(
                "Archive key missing. Existing history cannot be overwritten. Restore a password-protected backup into a new history location."
            )
        return key

    def _path(self, rid):
        if not isinstance(rid, str) or not ID.fullmatch(rid):
            raise ArchiveError("Invalid archive identifier.")
        return self.root / (rid + ".fae")

    def _read(self, rid):
        path = self._path(rid)
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(fd, "rb") as stream:
                size = os.fstat(stream.fileno())
                if (
                    not stat.S_ISREG(size.st_mode)
                    or not 33 <= size.st_size <= MAX_RECORD_BYTES
                ):
                    raise ArchiveError("Archive size or type is invalid.")
                raw = stream.read(MAX_RECORD_BYTES + 1)
            if not raw.startswith(MAGIC):
                raise ArchiveError("Unsupported archive format.")
            plain = AESGCM(self._key()).decrypt(
                raw[5:17], raw[17:], MAGIC + rid.encode()
            )
            record = Record.model_validate_json(plain)
            if record.id != rid:
                raise ArchiveError("Archive identifier mismatch.")
            return record
        except ArchiveError:
            raise
        except (OSError, InvalidTag, ValueError) as error:
            raise ArchiveError(
                "Archive is missing, damaged, unsupported or uses a different key."
            ) from error

    def read(self, rid):
        with self._locked():
            return self._read(rid)

    def list(self):
        with self._locked():
            rows = []
            for path in self._paths():
                try:
                    r = self._read(path.stem)
                    rows.append(
                        dict(
                            id=r.id,
                            name=r.name,
                            profile=r.snapshot.profile,
                            created_at=r.snapshot.created_at,
                            archived_at=r.archived_at,
                            imported_at=r.imported_at,
                            score=r.snapshot.score,
                            evidence_count=len(r.evidence),
                            bytes=path.lstat().st_size,
                        )
                    )
                except ArchiveError:
                    rows.append(dict(id=path.stem, status="locked_or_damaged"))
            return rows

    def _write(self, record, *, new=False):
        committed = False
        try:
            plain = record.model_dump_json().encode()
            if len(plain) + 33 > MAX_RECORD_BYTES:
                raise ArchiveError(
                    "Archive exceeds the 6 MB encrypted-record limit. Save fewer photos or report only."
                )
            existing = self._paths()
            path = self._path(record.id)
            if new and len(existing) >= MAX_RECORDS:
                raise ArchiveError("History has reached its 100-report limit.")
            current = path.lstat().st_size if path.exists() else 0
            total = sum(p.lstat().st_size for p in existing) - current + len(plain) + 33
            if total > MAX_TOTAL_BYTES:
                raise ArchiveError("History exceeds its 200 MB limit.")
            nonce = secrets.token_bytes(12)
            raw = (
                MAGIC
                + nonce
                + AESGCM(self._key(create=True)).encrypt(
                    nonce, plain, MAGIC + record.id.encode()
                )
            )
            temporary = self.root / (record.id + "." + secrets.token_hex(6) + ".tmp")
            try:
                fd = os.open(
                    temporary,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600,
                )
                with os.fdopen(fd, "wb") as stream:
                    stream.write(raw)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
                committed = True
                directory = os.open(self.root, os.O_RDONLY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            finally:
                temporary.unlink(missing_ok=True)
        except OSError as error:
            raise ArchiveError(
                "The new encrypted record was committed but its durability could not be confirmed."
                if committed
                else "The encrypted archive could not be written; existing files were preserved."
            ) from error

    def save(self, snapshot: Snapshot, *, evidence=None, name=None):
        snapshot = Snapshot.model_validate(snapshot.model_dump())
        record = Record(
            id=uuid4().hex,
            name=name or snapshot.profile,
            archived_at=now(),
            snapshot=snapshot,
            snapshot_hash=snapshot_hash(snapshot),
            evidence=[Evidence.model_validate(e.model_dump()) for e in evidence or []],
        )
        with self._locked():
            self._write(record, new=True)
        return record

    def rename(self, rid, name):
        with self._locked():
            record = self._read(rid)
            data = record.model_dump()
            data["name"] = name
            self._write(Record.model_validate(data))

    def remove_evidence(self, rid):
        with self._locked():
            record = self._read(rid)
            self._write(record.model_copy(update={"evidence": []}))

    def delete(self, rid):
        with self._locked():
            path = self._path(rid)
            if path.is_symlink() or not path.is_file():
                raise ArchiveError("Cannot delete this archive path.")
            try:
                path.unlink()
            except OSError as error:
                raise ArchiveError("Archive could not be deleted.") from error

    @staticmethod
    def _password_key(password, salt):
        if not isinstance(password, str) or not 12 <= len(password) <= 1024:
            raise ArchiveError(
                "Use a backup password of 12–1024 characters. It cannot be recovered."
            )
        return Scrypt(salt=salt, length=32, n=2**17, r=8, p=1).derive(password.encode())

    def backup(self, rid, password):
        record = self.read(rid)
        salt = secrets.token_bytes(16)
        nonce = secrets.token_bytes(12)
        key = self._password_key(password, salt)
        return (
            BACKUP_MAGIC
            + salt
            + nonce
            + AESGCM(key).encrypt(
                nonce, record.model_dump_json().encode(), BACKUP_MAGIC + salt
            )
        )

    def import_backup(self, raw, password):
        if (
            not isinstance(raw, bytes)
            or not 49 <= len(raw) <= MAX_RECORD_BYTES + 16
            or not raw.startswith(BACKUP_MAGIC)
        ):
            raise ArchiveError("Backup is too large, truncated or unsupported.")
        try:
            salt = raw[5:21]
            nonce = raw[21:33]
            key = self._password_key(password, salt)
            data = AESGCM(key).decrypt(nonce, raw[33:], BACKUP_MAGIC + salt)
            old = Record.model_validate_json(data)
            new = old.model_copy(
                update={"id": uuid4().hex, "archived_at": now(), "imported_at": now()}
            )
            with self._locked():
                self._write(new, new=True)
            return new
        except ArchiveError:
            raise
        except (ValueError, InvalidTag) as error:
            raise ArchiveError(
                "Password incorrect, backup damaged or schema unsupported."
            ) from error
