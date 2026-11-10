"""Private, expiring workspaces for one local analysis subject at a time."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
import re
import secrets
import shutil
import time
from typing import Any, Iterator

from .contracts import Demographics

SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{32,96}$")
SESSION_AREAS = frozenset({"uploads", "derived"})
DEFAULT_TTL_SECONDS = 24 * 60 * 60
LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class Session:
    session_id: str
    csrf_token: str
    root: Path
    demographics: Demographics
    created_at: float
    last_activity: float
    state: dict[str, Any] = field(default_factory=dict)


class SessionStore:
    """Owns session paths and is the only code allowed to construct them."""

    def __init__(
        self,
        base_dir: Path,
        *,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self.base_dir = Path(base_dir).expanduser().resolve()
        self.ttl_seconds = int(ttl_seconds)
        self._clock = clock
        self._owned_session_ids: set[str] = set()
        self.base_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self.base_dir, 0o700)

    def create(self, demographics: Demographics) -> Session:
        while True:
            session_id = secrets.token_urlsafe(32)
            root = self.base_dir / session_id
            try:
                root.mkdir(mode=0o700)
                break
            except FileExistsError:
                continue

        for area in sorted(SESSION_AREAS):
            (root / area).mkdir(mode=0o700)

        now = float(self._clock())
        session = Session(
            session_id=session_id,
            csrf_token=secrets.token_urlsafe(24),
            root=root,
            demographics=demographics,
            created_at=now,
            last_activity=now,
        )
        self.save(session)
        self._owned_session_ids.add(session_id)
        return session

    def save(self, session: Session) -> None:
        root = self._session_root(session.session_id)
        if session.root.resolve() != root.resolve():
            raise ValueError("session root does not belong to this store")
        if not root.is_dir() or root.is_symlink():
            raise FileNotFoundError("session workspace is missing")

        session.last_activity = float(self._clock())
        payload = {
            "version": 1,
            "session_id": session.session_id,
            "csrf_token": session.csrf_token,
            "demographics": session.demographics.to_dict(),
            "created_at": session.created_at,
            "last_activity": session.last_activity,
            "state": session.state,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        state_path = root / "state.json"
        temporary = root / f".state-{secrets.token_hex(8)}.tmp"
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, state_path)
            os.chmod(state_path, 0o600)
        finally:
            if temporary.exists():
                temporary.unlink()

    def load(self, session_id: str) -> Session:
        root = self._session_root(session_id)
        if not root.is_dir() or root.is_symlink():
            raise FileNotFoundError("session not found")
        state_path = root / "state.json"
        with state_path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        if payload.get("version") != 1 or payload.get("session_id") != session_id:
            raise ValueError("session state is invalid")

        raw_demographics = payload.get("demographics", {})
        demographics = Demographics(
            age=raw_demographics.get("age"),
            sex=raw_demographics.get("sex"),
            ethnicities=tuple(raw_demographics.get("ethnicities", ())),
        )
        state = payload.get("state", {})
        if not isinstance(state, dict):
            raise ValueError("session state payload must be an object")

        return Session(
            session_id=session_id,
            csrf_token=self._required_text(payload, "csrf_token"),
            root=root,
            demographics=demographics,
            created_at=self._required_timestamp(payload, "created_at"),
            last_activity=self._required_timestamp(payload, "last_activity"),
            state=state,
        )

    def touch(self, session_id: str) -> Session:
        session = self.load(session_id)
        self.save(session)
        return session

    def path_for(self, session_id: str, area: str, filename: str) -> Path:
        root = self._session_root(session_id)
        if area not in SESSION_AREAS:
            raise ValueError("unknown session area")
        if not isinstance(filename, str) or not filename or len(filename) > 255:
            raise ValueError("invalid server-assigned filename")
        if filename in {".", ".."} or Path(filename).name != filename:
            raise ValueError("filename must not contain a path")
        if "/" in filename or "\\" in filename or "\x00" in filename:
            raise ValueError("filename must not contain a path")

        directory = root / area
        if not directory.is_dir() or directory.is_symlink():
            raise FileNotFoundError("session area is missing")
        return directory / filename

    def end(self, session_id: str) -> None:
        root = self._session_root(session_id)
        if not root.exists():
            self._owned_session_ids.discard(session_id)
            return
        if root.is_symlink():
            raise ValueError("refusing to follow a session symlink")
        shutil.rmtree(root)
        self._owned_session_ids.discard(session_id)

    @property
    def has_owned_sessions(self) -> bool:
        return bool(self._owned_session_ids)

    def claim(self, session_id: str) -> None:
        """Mark a validated existing session as owned by this server process."""
        root = self._session_root(session_id)
        if not root.is_dir() or root.is_symlink():
            raise FileNotFoundError("session not found")
        self._owned_session_ids.add(session_id)

    @contextmanager
    def analysis_scope(self, session_id: str) -> Iterator[Path]:
        """Provide a private scratch directory and erase the session on failure."""
        self.claim(session_id)
        workspace = self.path_for(
            session_id,
            "derived",
            f".analysis-{secrets.token_hex(12)}",
        )
        workspace.mkdir(mode=0o700)
        try:
            yield workspace
        except Exception:
            try:
                self.end(session_id)
            except (OSError, ValueError):
                LOGGER.error("session_cleanup event=analysis_failure count=0")
            else:
                LOGGER.info("session_cleanup event=analysis_failure count=1")
            raise
        finally:
            if workspace.is_symlink():
                workspace.unlink(missing_ok=True)
            elif workspace.exists():
                shutil.rmtree(workspace)

    def cleanup_all(self) -> int:
        """Erase sessions owned or claimed by this store; safe to call repeatedly."""
        removed = 0
        failures = 0
        for session_id in tuple(sorted(self._owned_session_ids)):
            root = self._session_root(session_id)
            existed = root.exists() and not root.is_symlink()
            try:
                self.end(session_id)
            except (OSError, ValueError):
                failures += 1
                continue
            if existed:
                removed += 1
        LOGGER.info("session_cleanup event=shutdown count=%d failures=%d", removed, failures)
        return removed

    def expire_stale(self) -> list[str]:
        now = float(self._clock())
        expired: list[str] = []
        for root in sorted(self.base_dir.iterdir(), key=lambda path: path.name):
            if root.is_symlink() or not root.is_dir() or not SESSION_ID_PATTERN.fullmatch(root.name):
                continue
            try:
                session = self.load(root.name)
            except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
                continue
            if now - session.last_activity > self.ttl_seconds:
                self.end(session.session_id)
                expired.append(session.session_id)
        if expired:
            LOGGER.info("session_cleanup event=ttl_expiry count=%d", len(expired))
        return expired

    def _session_root(self, session_id: str) -> Path:
        if not isinstance(session_id, str) or not SESSION_ID_PATTERN.fullmatch(session_id):
            raise ValueError("invalid session identifier")
        return self.base_dir / session_id

    @staticmethod
    def _required_text(payload: dict[str, Any], key: str) -> str:
        value = payload.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"session {key} is invalid")
        return value

    @staticmethod
    def _required_timestamp(payload: dict[str, Any], key: str) -> float:
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"session {key} is invalid")
        return float(value)
