"""Deterministic assembly primitives for the private macOS application."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path, PurePosixPath
import plistlib
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from urllib.request import Request, urlopen
from urllib.parse import quote
import zipfile

from . import inventory
from .config import (
    APP_VERSION,
    ARCHITECTURES,
    MEDIAPIPE_VERSION,
    MINIMUM_MACOS,
    MODEL_ASSETS,
    PYTHON_BUILDS,
)

APP_NAME = "Facial Analysis.app"
EXECUTABLE_NAME = "Facial Analysis"
FIXED_ZIP_TIME = (2026, 1, 1, 0, 0, 0)
BANK_SHA256 = "938f02dfaff4d32c914a89bd303f824ca7ced12a6a1b5a0b37280623f7d40540"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BUILD_ROOT = REPOSITORY_ROOT / ".build" / "macos"
RELEASE_OUTPUT = REPOSITORY_ROOT / "dist" / "macos"
DOWNLOAD_CHUNK_BYTES = 1024 * 1024
LOCKED_RUNTIME_REQUIREMENTS = (
    "cryptography==50.0.2",
    "pydantic==2.14.1",
    "pydantic-core==2.50.1",
    "annotated-types==0.8.0",
    "typing-inspection==0.4.4",
    "typing-extensions==4.16.0",
    "limits==5.8.0",
    "deprecated==1.3.1",
    "wrapt==2.5.1",
    "absl-py==2.5.0",
    "certifi==2026.7.22",
    "cffi==2.1.1",
    "contourpy==1.3.3",
    "cycler==0.12.1",
    "flatbuffers==25.12.19",
    "fonttools==4.65.0",
    "kiwisolver==1.5.1",
    "matplotlib==3.11.0",
    "numpy==2.4.6",
    "opencv-contrib-python==4.13.0.92",
    "packaging==26.3",
    "pillow==12.3.0",
    "pycparser==3.0",
    "pyparsing==3.3.3",
    "python-dateutil==2.9.0.post0",
    "scipy==1.17.1",
    "six==1.17.0",
    "sounddevice==0.5.6",
)


class UnsafeArchiveError(RuntimeError):
    """A downloaded runtime archive contains an unsafe member."""


def _safe_member_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise UnsafeArchiveError("runtime archive contains an unsafe path")
    return path


def safe_extract_tar(archive: Path, destination: Path) -> Path:
    """Extract regular files, directories, and contained relative links only."""
    archive = Path(archive).resolve()
    destination = Path(destination).resolve()
    destination.mkdir(mode=0o700, parents=True, exist_ok=True)
    with tarfile.open(archive, "r:*") as stream:
        members = stream.getmembers()
        for member in members:
            relative = _safe_member_path(member.name)
            if member.ischr() or member.isblk() or member.isfifo():
                raise UnsafeArchiveError("runtime archive contains a special file")
            if member.issym() or member.islnk():
                target = PurePosixPath(member.linkname)
                if target.is_absolute():
                    raise UnsafeArchiveError("runtime archive contains an unsafe link")
                base = relative.parent if member.issym() else PurePosixPath(".")
                parts: list[str] = []
                for part in (base / target).parts:
                    if part == "..":
                        if not parts:
                            raise UnsafeArchiveError(
                                "runtime archive link escapes its root"
                            )
                        parts.pop()
                    elif part not in {"", "."}:
                        parts.append(part)
            elif not (member.isdir() or member.isfile()):
                raise UnsafeArchiveError(
                    "runtime archive contains an unsupported member"
                )
        stream.extractall(destination)
    return destination


def _write_info_plist(destination: Path, architecture: str) -> None:
    template_path = Path(__file__).with_name("Info.plist")
    with template_path.open("rb") as stream:
        payload = plistlib.load(stream)
    payload["CFBundleShortVersionString"] = APP_VERSION
    payload["LSMinimumSystemVersion"] = MINIMUM_MACOS
    payload["FAArchitecture"] = architecture
    with destination.open("wb") as stream:
        plistlib.dump(payload, stream, sort_keys=True)


def stage_application(
    *,
    architecture: str,
    python_root: Path,
    payload_root: Path,
    bootstrap_path: Path,
    launcher_path: Path,
    destination: Path,
) -> Path:
    """Create an unsigned app layout from already verified inputs."""
    destination = Path(destination).resolve()
    app = destination / APP_NAME
    if app.exists():
        shutil.rmtree(app)
    macos = app / "Contents" / "MacOS"
    resources = app / "Contents" / "Resources"
    macos.mkdir(mode=0o755, parents=True)
    resources.mkdir(mode=0o755)
    shutil.copy2(launcher_path, macos / EXECUTABLE_NAME, follow_symlinks=False)
    os.chmod(macos / EXECUTABLE_NAME, 0o755)
    shutil.copytree(python_root, resources / "python", symlinks=True)
    shutil.copytree(payload_root, resources / "payload", symlinks=False)
    shutil.copy2(bootstrap_path, resources / "bootstrap.py", follow_symlinks=False)
    os.chmod(resources / "bootstrap.py", 0o600)
    _write_info_plist(app / "Contents" / "Info.plist", architecture)
    # Standalone runtimes may preserve group-writable build permissions.
    # Tighten only the new bundle, preserving execute bits and private modes.
    for path in (app, *app.rglob("*")):
        if not path.is_symlink():
            os.chmod(path, stat.S_IMODE(path.stat().st_mode) & ~0o022)
    return app


def _zip_mode(path: Path) -> int:
    if path.is_symlink():
        return stat.S_IFLNK | 0o777
    if path.is_dir():
        return stat.S_IFDIR | 0o755
    return stat.S_IFREG | (0o755 if os.access(path, os.X_OK) else 0o644)


def write_deterministic_zip(app: Path, destination: Path) -> Path:
    """Write a byte-stable archive while preserving executable bits and links."""
    app = Path(app).resolve()
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    entries = [app, *sorted(app.rglob("*"), key=lambda item: item.as_posix())]
    with zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as stream:
        for path in entries:
            relative = path.relative_to(app.parent).as_posix()
            is_directory = path.is_dir() and not path.is_symlink()
            name = f"{relative}/" if is_directory else relative
            info = zipfile.ZipInfo(name, FIXED_ZIP_TIME)
            info.create_system = 3
            info.external_attr = _zip_mode(path) << 16
            if is_directory:
                payload = b""
            elif path.is_symlink():
                payload = os.readlink(path).encode("utf-8")
            else:
                payload = path.read_bytes()
            stream.writestr(
                info, payload, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9
            )
    return destination


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _download_runtime(architecture: str, destination: Path) -> Path:
    specification = PYTHON_BUILDS[architecture]
    destination.mkdir(mode=0o700, parents=True, exist_ok=True)
    archive = destination / specification.name
    if archive.is_file() and sha256(archive) == specification.sha256:
        return archive
    partial = archive.with_suffix(f"{archive.suffix}.partial")
    partial.unlink(missing_ok=True)
    request = Request(
        specification.url, headers={"User-Agent": "facial-analysis-builder"}
    )
    try:
        with urlopen(request, timeout=60) as response, partial.open("wb") as stream:
            while chunk := response.read(DOWNLOAD_CHUNK_BYTES):
                stream.write(chunk)
        if sha256(partial) != specification.sha256:
            raise RuntimeError("downloaded Python runtime checksum differs")
        partial.replace(archive)
    finally:
        partial.unlink(missing_ok=True)
    return archive


def _replace_directory(path: Path) -> None:
    resolved_parent = path.parent.resolve()
    if BUILD_ROOT.resolve() not in (resolved_parent, *resolved_parent.parents):
        raise RuntimeError("refusing to replace a directory outside the build root")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(mode=0o700, parents=True)


def _prepare_python(architecture: str) -> Path:
    archive = _download_runtime(architecture, BUILD_ROOT / "downloads")
    destination = BUILD_ROOT / "python" / architecture
    _replace_directory(destination)
    safe_extract_tar(archive, destination)
    python_root = destination / "python"
    interpreter = python_root / "bin" / "python3"
    if not interpreter.is_file():
        raise RuntimeError("downloaded Python runtime is incomplete")
    return python_root


def _architecture_command(architecture: str, command: list[str]) -> list[str]:
    if architecture == "x86_64":
        return ["/usr/bin/arch", "-x86_64", *command]
    return command


def _run(command: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def _install_dependencies(
    architecture: str,
    python_root: Path,
    *,
    intel_wheel: Path | None,
) -> None:
    interpreter = python_root / "bin" / "python3"
    base = [str(interpreter), "-m", "pip", "install", "--disable-pip-version-check"]
    _run(
        _architecture_command(
            architecture,
            [*base, "--only-binary=:all:", *LOCKED_RUNTIME_REQUIREMENTS],
        )
    )
    if architecture == "x86_64":
        if intel_wheel is None or not intel_wheel.is_file():
            raise RuntimeError("verified Intel MediaPipe wheel is missing")
        _run(
            _architecture_command(
                architecture,
                [*base, "--only-binary=:all:", "--no-deps", str(intel_wheel)],
            )
        )
    else:
        _run(
            [
                *base,
                "--only-binary=:all:",
                "--no-deps",
                f"mediapipe=={MEDIAPIPE_VERSION}",
            ]
        )


def _remove_build_residue(root: Path) -> None:
    executable_directory = root / "bin"
    if executable_directory.is_dir():
        keep = {"python", "python3", "python3.11"}
        for path in executable_directory.iterdir():
            if path.name not in keep:
                if path.is_dir() and not path.is_symlink():
                    shutil.rmtree(path)
                else:
                    path.unlink()
    for path in sorted(root.rglob("__pycache__"), reverse=True):
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
    for pattern in ("*.pyc", "*.pyo"):
        for path in root.rglob(pattern):
            path.unlink()
    for path in root.rglob("direct_url.json"):
        path.unlink()
    removable = (
        root / "lib" / "python3.11" / "idlelib",
        root / "lib" / "tk9.0" / "images",
        root
        / "lib"
        / "python3.11"
        / "site-packages"
        / "matplotlib"
        / "mpl-data"
        / "sample_data",
    )
    for path in removable:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
    site_packages = root / "lib" / "python3.11" / "site-packages"
    if site_packages.is_dir():
        for path in sorted(site_packages.rglob("tests"), reverse=True):
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)


def _assert_no_local_build_paths(root: Path) -> None:
    home = Path.home().expanduser().resolve()
    sensitive = {os.fspath(REPOSITORY_ROOT), os.fspath(BUILD_ROOT), os.fspath(home)}
    needles = {
        representation.encode("utf-8")
        for value in sensitive
        for representation in (value, quote(value), quote(value, safe=""))
    }
    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        if path.name == "direct_url.json":
            raise RuntimeError("application contains local installation metadata")
        with path.open("rb") as stream:
            previous = b""
            overlap = max(map(len, needles)) - 1
            while chunk := stream.read(DOWNLOAD_CHUNK_BYTES):
                combined = previous + chunk
                if any(needle in combined for needle in needles):
                    raise RuntimeError("application contains a local build path")
                previous = combined[-overlap:]


def _stage_payload() -> Path:
    bank = REPOSITORY_ROOT / "bank" / "derived.npz"
    if not bank.is_file() or sha256(bank) != BANK_SHA256:
        raise RuntimeError("reference bank is absent or differs from the release bank")
    files = inventory.audited_payload(REPOSITORY_ROOT)
    measurement_modules = inventory.bank_measurement_modules(REPOSITORY_ROOT)
    payload = BUILD_ROOT / "payload"
    _replace_directory(payload)
    for relative in files:
        source = REPOSITORY_ROOT / relative
        target = payload / relative
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        shutil.copy2(source, target, follow_symlinks=False)
    inventory.write_manifest(
        REPOSITORY_ROOT,
        files,
        payload / "manifest.json",
        measurement_modules=measurement_modules,
        runtime_assets={
            name: download.sha256 for name, download in MODEL_ASSETS.items()
        },
    )
    return payload


def _compile_launcher(architecture: str) -> Path:
    destination = BUILD_ROOT / "launchers" / architecture / EXECUTABLE_NAME
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    _run(
        [
            "swiftc",
            "-O",
            "-target",
            f"{architecture}-apple-macos{MINIMUM_MACOS}",
            str(Path(__file__).with_name("launcher.swift")),
            "-o",
            str(destination),
        ]
    )
    return destination


def _codesign(app: Path) -> None:
    _run(["codesign", "--force", "--deep", "--sign", "-", str(app)])
    _run(["codesign", "--verify", "--deep", "--strict", str(app)])


def _find_intel_wheel() -> Path | None:
    candidates = sorted((BUILD_ROOT / "wheels" / "x86_64").glob("*.whl"))
    if len(candidates) > 1:
        raise RuntimeError("Intel MediaPipe wheel inventory is ambiguous")
    if not candidates:
        return None
    from .mediapipe_wheel import verify_wheel_metadata

    verify_wheel_metadata(candidates[0])
    return candidates[0]


def _promote_release(staged: Path, output: Path) -> None:
    """Atomically expose a complete release directory, restoring on failure."""
    staged = Path(staged).resolve()
    output = Path(output).expanduser().absolute()
    output.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    if output.is_symlink():
        raise RuntimeError("release output must not be a symlink")
    backup = output.parent / f".{output.name}.previous"
    if backup.exists() or backup.is_symlink():
        raise RuntimeError("stale release backup must be removed")
    had_output = output.exists()
    if had_output:
        output.replace(backup)
    try:
        staged.replace(output)
    except BaseException:
        if had_output and backup.exists() and not output.exists():
            backup.replace(output)
        raise
    if had_output:
        if backup.is_dir() and not backup.is_symlink():
            shutil.rmtree(backup)
        else:
            backup.unlink()


def _validate_release_output(output: Path) -> Path:
    """Restrict destructive promotion to the repository's release directory."""
    candidate = Path(os.path.abspath(Path(output).expanduser()))
    expected = Path(os.path.abspath(RELEASE_OUTPUT))
    if candidate != expected:
        raise ValueError(
            f"release output must be the dedicated release directory: {expected}"
        )
    if expected.parent.is_symlink() or expected.is_symlink():
        raise ValueError("dedicated release directory must not traverse a symlink")
    return expected


def build_distributions(architectures: list[str], output: Path) -> tuple[Path, ...]:
    """Build, sign, and archive the requested native application bundles."""
    if len(architectures) != len(ARCHITECTURES) or set(architectures) != set(
        ARCHITECTURES
    ):
        raise ValueError(
            "release builds require exactly one arm64 and one x86_64 target"
        )
    output = _validate_release_output(output)
    payload = _stage_payload()
    intel_wheel = _find_intel_wheel()
    BUILD_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="release-stage-", dir=BUILD_ROOT
    ) as temporary:
        staged_output = Path(temporary) / "release"
        staged_output.mkdir(mode=0o700)
        staged_apps = Path(temporary) / "apps"
        archives: list[Path] = []
        for architecture in architectures:
            python_root = _prepare_python(architecture)
            _install_dependencies(
                architecture,
                python_root,
                intel_wheel=intel_wheel,
            )
            _remove_build_residue(python_root)
            launcher = _compile_launcher(architecture)
            app_parent = staged_apps / architecture
            app = stage_application(
                architecture=architecture,
                python_root=python_root,
                payload_root=payload,
                bootstrap_path=Path(__file__).with_name("bootstrap.py"),
                launcher_path=launcher,
                destination=app_parent,
            )
            _remove_build_residue(app)
            _assert_no_local_build_paths(app)
            _codesign(app)
            archive = staged_output / f"Facial-Analysis-macOS-{architecture}.zip"
            write_deterministic_zip(app, archive)
            archives.append(archive)
        checksums = staged_output / "SHA256SUMS.txt"
        checksums.write_text(
            "".join(f"{sha256(path)}  {path.name}\n" for path in archives),
            encoding="utf-8",
        )
        names = tuple(path.name for path in archives)
        _promote_release(staged_output, output)
    return tuple(output / name for name in names)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architectures", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args(argv)
    try:
        archives = build_distributions(options.architectures, options.output)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        parser.exit(1, f"macOS application build failed: {error}\n")
    sys.stdout.write(f"built architectures: {len(archives)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "UnsafeArchiveError",
    "safe_extract_tar",
    "sha256",
    "stage_application",
    "build_distributions",
    "write_deterministic_zip",
]
