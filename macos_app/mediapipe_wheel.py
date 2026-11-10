"""Build and verify the pinned official MediaPipe wheel for Intel macOS."""

from __future__ import annotations

import argparse
import base64
from collections.abc import Callable, Mapping, Sequence
import csv
from dataclasses import dataclass
from email.parser import Parser
import hashlib
import io
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from typing import Protocol
import zipfile

from .config import MEDIAPIPE_COMMIT, MEDIAPIPE_VERSION

OFFICIAL_REPOSITORY = "https://github.com/google-ai-edge/mediapipe.git"
PATCH_PATH = Path(__file__).with_name("mediapipe-x86_64.patch")
PATCH_SHA256 = "7aa543d610cde326855d19125b3550864353ad780d73aedd712878509beede0f"
PATCH_BASE_SETUP_BLOB = "2c5a77c4d69c6e00677fba565f2b7da20339c42a"
PATCHED_SETUP_BLOB = "efc1de35284b981c334f67a58eaa2b0c78709faa"
PATCH_BASE_THIRD_PARTY_BLOB = "2ab0a7d00e85a0a562b76e420f3a49f8199c63f8"
PATCHED_THIRD_PARTY_BLOB = "5258b714cfdfc135b08f23263dd1f606eaeb3227"
DEFAULT_PACKAGED_PYTHON = (
    Path(__file__).resolve().parents[1]
    / ".build/macos/python/x86_64/python/bin/python3"
)
ARCH_COMMAND = "/usr/bin/arch"
IMPORT_CHECK = (
    "import pathlib, mediapipe; "
    f"assert mediapipe.__version__ == '{MEDIAPIPE_VERSION}'; "
    "print(mediapipe.__version__); "
    "print(pathlib.Path(mediapipe.__file__).resolve())"
)
NATIVE_LIBRARY_QUERY = (
    "import mediapipe, pathlib; "
    "root = pathlib.Path(mediapipe.__file__).resolve().parent; "
    "print('\\n'.join(str(path) for path in sorted(root.rglob('*')) "
    "if path.suffix in {'.so', '.dylib'}))"
)


# Upstream's setup.py builds //mediapipe/tasks/c:libmediapipe.so on every
# platform, but the 0.10.35 Python bindings dlopen libmediapipe.dylib on Darwin,
# and the shared OpenCV that the Darwin source build links stays behind in
# Bazel's output tree behind absolute rpaths. A source-built Intel wheel is
# therefore not loadable as built: 2026-09-21 the first real Intel journey
# answered 500 because the library could not be found, and renamed it could
# not be loaded. The repair below makes the wheel self-contained and the
# verification below it actually loads the library under Rosetta.
TASKS_LIBRARY_DIR = "mediapipe/tasks/c"
BUILT_TASKS_LIBRARY = "libmediapipe.so"
DARWIN_TASKS_LIBRARY = "libmediapipe.dylib"
OPENCV_LIBRARY_DIR = Path("bazel-bin/third_party/opencv_cmake/lib")
INSTALL_NAME_TOOL = "/usr/bin/install_name_tool"
OTOOL = "/usr/bin/otool"
CODESIGN = "/usr/bin/codesign"
TASKS_LIBRARY_LOAD_CHECK = (
    "import ctypes, importlib.resources as resources; "
    f"path = resources.files('mediapipe.tasks.c') / '{DARWIN_TASKS_LIBRARY}'; "
    "library = ctypes.CDLL(str(path)); "
    "assert hasattr(library, 'MpErrorFree'), 'MpErrorFree missing'; "
    "print('tasks library loaded')"
)


class MediaPipeWheelError(RuntimeError):
    """Base error for the pinned Intel wheel workflow."""


class BuildPrerequisiteError(MediaPipeWheelError): ...


class SourceIntegrityError(MediaPipeWheelError): ...


class WheelVerificationError(MediaPipeWheelError): ...


class WheelUnavailableError(MediaPipeWheelError): ...


class CommandRunner(Protocol):
    def __call__(
        self,
        command: Sequence[str | os.PathLike[str]],
        *,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
        check: bool = False,
        capture_output: bool = True,
    ) -> subprocess.CompletedProcess[str]: ...


Which = Callable[[str], str | None]


@dataclass(frozen=True)
class WheelSelection:
    architecture: str
    version: str
    route: str
    repository: str | None = None
    commit: str | None = None
    build_command: tuple[str, ...] = ()


@dataclass(frozen=True)
class BuildTools:
    git: str
    bazelisk: str
    cmake: str
    protoc: str
    xcode_select: str
    arch: str
    lipo: str
    file: str


@dataclass(frozen=True)
class LockedRequirement:
    name: str
    version: str
    sha256: str


def _locked_requirements(text: str) -> tuple[LockedRequirement, ...]:
    return tuple(LockedRequirement(*line.split()) for line in text.splitlines())


BUILD_REQUIREMENTS = _locked_requirements(
    """setuptools 82.0.1 a59e362652f08dcd477c78bb6e7bd9d80a7995bc73ce773050228a348ce2e5bb
wheel 0.48.0 3217dcc807155e45db462d7ef2431f5ddda0d7273b700d05a67b271ceb1287ab
packaging 26.2 5fc45236b9446107ff2415ce77c807cee2862cb6fac22b8a73826d0693b0980e"""
)

RUNTIME_REQUIREMENTS = _locked_requirements(
    """packaging 26.2 5fc45236b9446107ff2415ce77c807cee2862cb6fac22b8a73826d0693b0980e
absl-py 2.4.0 88476fd881ca8aab94ffa78b7b6c632a782ab3ba1cd19c9bd423abc4fb4cd28d
certifi 2026.5.20 3c52e209ba0a4ad7aebe60436a4ab349c39e1e602e8c134221e546902ad25897
numpy 2.4.6 8155154c7c691289fe18f510b5d4657c68c67989f293f0535a91360392ff6538
sounddevice 0.5.5 05eb9fd6c54c38d67741441c19164c0dae8ce80453af2d8c4ad2e7823d15b722
flatbuffers 25.12.19 7634f50c427838bb021c2d66a3d1168e9d199b0607e6329399f04846d42e20b4
opencv-contrib-python 4.13.0.92 1973d0fc773873f9d1b5bf0d1b65895da2f47b06ba033b7d58393f5c28ba0778
matplotlib 3.11.0 f857524b442f0f36e641868ce2171aafa88cb0bc0644f4e1d8a5df9b32649fef
cffi 2.0.0 b4c854ef3adc177950a8dfc81a86f5115d2abd545751a304c5bcf2c2c7283cfe
contourpy 1.3.3 709a48ef9a690e1343202916450bc48b9e51c049b089c7f79a267b46cffcdaa1
cycler 0.12.1 85cef7cff222d8644161529808465972e51340599459b8ac3ccbac5a854e0d30
fonttools 4.63.0 79cdc9f567aec74a72918fd060283911406750cbc9fd28c1316023deb6ce31a9
kiwisolver 1.5.0 3c4923e404d6bcd91b6779c009542e5647fef32e4a5d75e115e3bbac6f2335eb
pillow 12.2.0 8be29e59487a79f173507c30ddf57e733a357f67881430449bb32614075a40ab
pyparsing 3.3.2 850ba148bd908d7e2411587e247a1e4f0327839c40e2e5e6d05a007ecc69911d
python-dateutil 2.9.0.post0 a8b2bc7bffae282281c8140a97d3aa9c14da0b136dfe83f850eea9a5f7470427
six 1.17.0 4721f391ed90541fddacab5acf947aa0d3dc7d27b2e1e8eda2be8970586c3274
pycparser 3.0 b727414169a36b7d524c1c3e31839a521725078d7b2ff038656844266160a992"""
)


def _run(
    command: Sequence[str | os.PathLike[str]],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    check: bool = False,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        tuple(str(part) for part in command),
        cwd=cwd,
        env=None if env is None else dict(env),
        check=check,
        text=True,
        capture_output=capture_output,
    )


def _validate_architecture(architecture: str) -> None:
    if architecture not in {"arm64", "x86_64"}:
        raise ValueError(f"unsupported macOS architecture: {architecture}")


def _sanitized_environment(
    additions: Mapping[str, str] | None = None,
) -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("PYTHON") and key != "__PYVENV_LAUNCHER__"
    }
    if additions:
        environment.update(additions)
    return environment


def available_binary_wheel(
    architecture: str,
    *,
    runner: CommandRunner = _run,
) -> bool:
    """Return whether PyPI resolves the exact pinned binary for ``architecture``."""
    _validate_architecture(architecture)
    platform_tag = f"macosx_13_0_{architecture}"
    with tempfile.TemporaryDirectory(prefix="mediapipe-wheel-probe-") as directory:
        result = runner(
            (
                sys.executable,
                "-m",
                "pip",
                "--isolated",
                "download",
                "--disable-pip-version-check",
                "--index-url",
                "https://pypi.org/simple",
                "--only-binary=:all:",
                "--no-deps",
                "--implementation",
                "cp",
                "--python-version",
                "3.11",
                "--abi",
                "cp311",
                "--platform",
                platform_tag,
                "--dest",
                directory,
                f"mediapipe=={MEDIAPIPE_VERSION}",
            )
        )
    return result.returncode == 0


def wheel_for(
    architecture: str,
    *,
    runner: CommandRunner = _run,
) -> WheelSelection:
    """Choose a pinned wheel route without substituting a MediaPipe version."""
    _validate_architecture(architecture)
    if available_binary_wheel(architecture, runner=runner):
        return WheelSelection(
            architecture=architecture,
            version=MEDIAPIPE_VERSION,
            route="binary",
        )
    if architecture == "x86_64":
        return WheelSelection(
            architecture=architecture,
            version=MEDIAPIPE_VERSION,
            route="official-source",
            repository=OFFICIAL_REPOSITORY,
            commit=MEDIAPIPE_COMMIT,
            build_command=(
                sys.executable,
                "-m",
                "macos_app.mediapipe_wheel",
                "--output",
            ),
        )
    raise WheelUnavailableError(
        f"official mediapipe {MEDIAPIPE_VERSION} binary is unavailable for {architecture}"
    )


def _required_tools(which: Which) -> BuildTools:
    names = (
        "git",
        "bazelisk",
        "cmake",
        "protoc",
        "xcode-select",
        "arch",
        "lipo",
        "file",
    )
    resolved = {name: which(name) for name in names}
    missing = sorted(name for name, path in resolved.items() if path is None)
    if missing:
        raise BuildPrerequisiteError(
            "missing required build tools: " + ", ".join(missing)
        )
    return BuildTools(
        **{name.replace("-", "_"): path for name, path in resolved.items()}
    )


def require_build_environment(
    packaged_python: Path,
    *,
    runner: CommandRunner = _run,
    which: Which = shutil.which,
) -> BuildTools:
    """Require Xcode tools, Bazelisk, CMake, protobuf, Rosetta, and x86 Python."""
    if sys.platform != "darwin":
        raise BuildPrerequisiteError("the Intel MediaPipe wheel must be built on macOS")
    tools = _required_tools(which)
    packaged_python = Path(packaged_python).expanduser().resolve()
    if not packaged_python.is_file():
        raise BuildPrerequisiteError(
            f"packaged x86_64 Python is missing: {packaged_python}"
        )

    xcode = runner((tools.xcode_select, "-p"))
    if xcode.returncode != 0 or not (xcode.stdout or "").strip():
        raise BuildPrerequisiteError("Xcode command-line tools are unavailable")

    file_result = runner((tools.file, "-b", str(packaged_python)))
    if file_result.returncode != 0 or "x86_64" not in (file_result.stdout or ""):
        raise BuildPrerequisiteError("packaged Python does not contain x86_64 code")

    bazelisk_result = runner((tools.file, "-b", tools.bazelisk))
    if bazelisk_result.returncode != 0 or "x86_64" not in (
        bazelisk_result.stdout or ""
    ):
        raise BuildPrerequisiteError(
            "an x86_64 Bazelisk binary is required for the Intel toolchain"
        )

    rosetta = runner((tools.arch, "-x86_64", "/usr/bin/true"))
    if rosetta.returncode != 0:
        raise BuildPrerequisiteError(
            "Rosetta 2 is unavailable; install it with softwareupdate before building"
        )

    interpreter = runner(
        (
            tools.arch,
            "-x86_64",
            str(packaged_python),
            "-I",
            "-c",
            "import platform,sys; print(platform.machine()); "
            "print(f'{sys.version_info.major}.{sys.version_info.minor}')",
        ),
        env=_sanitized_environment(),
    )
    reported = (interpreter.stdout or "").splitlines()
    if interpreter.returncode != 0 or reported != ["x86_64", "3.11"]:
        raise BuildPrerequisiteError(
            "packaged Python must run as CPython 3.11 x86_64 under Rosetta"
        )
    return tools


def verify_patch_integrity(patch_path: Path = PATCH_PATH) -> None:
    """Require the reviewed patch bytes and their exact Git blob identities."""
    try:
        payload = Path(patch_path).read_bytes()
    except OSError as error:
        raise SourceIntegrityError("MediaPipe patch is unreadable") from error
    if hashlib.sha256(payload).hexdigest() != PATCH_SHA256:
        raise SourceIntegrityError("MediaPipe patch checksum does not match review")
    expected_indices = (
        b"index 2c5a77c..efc1de3 100644",
        b"index 2ab0a7d..5258b71 100644",
    )
    if any(index not in payload for index in expected_indices):
        raise SourceIntegrityError("MediaPipe patch blob identity is invalid")


def checkout_official_source(
    destination: Path,
    *,
    git: str,
    runner: CommandRunner = _run,
) -> Path:
    """Clone, authenticate the exact official commit, then apply the local patch."""
    verify_patch_integrity()
    destination = Path(destination)
    runner(
        (
            git,
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            OFFICIAL_REPOSITORY,
            str(destination),
        ),
        check=True,
        capture_output=False,
    )
    runner(
        (
            git,
            "-C",
            str(destination),
            "fetch",
            "--depth=1",
            "origin",
            MEDIAPIPE_COMMIT,
        ),
        check=True,
        capture_output=False,
    )
    runner(
        (git, "-C", str(destination), "checkout", "--detach", MEDIAPIPE_COMMIT),
        check=True,
        capture_output=False,
    )
    head = runner(
        (git, "-C", str(destination), "rev-parse", "HEAD"),
        check=True,
    )
    if (head.stdout or "").strip() != MEDIAPIPE_COMMIT:
        raise SourceIntegrityError("unexpected MediaPipe commit; refusing to patch")
    for path, expected_blob in (
        ("setup.py", PATCH_BASE_SETUP_BLOB),
        ("third_party/BUILD", PATCH_BASE_THIRD_PARTY_BLOB),
    ):
        base_blob = runner(
            (git, "-C", str(destination), "rev-parse", f"HEAD:{path}"),
            check=True,
        )
        if (base_blob.stdout or "").strip() != expected_blob:
            raise SourceIntegrityError(f"unexpected upstream {path} blob")
    pristine = runner(
        (
            git,
            "-C",
            str(destination),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ),
        check=True,
    )
    if (pristine.stdout or "").strip():
        raise SourceIntegrityError("MediaPipe checkout is not pristine before patch")
    runner(
        (
            git,
            "-C",
            str(destination),
            "apply",
            "--check",
            "--index",
            str(PATCH_PATH),
        ),
        check=True,
    )
    runner(
        (git, "-C", str(destination), "apply", "--index", str(PATCH_PATH)),
        check=True,
    )
    staged_blobs = {}
    for path in ("setup.py", "third_party/BUILD"):
        staged_blobs[path] = (
            runner(
                (git, "-C", str(destination), "rev-parse", f":{path}"),
                check=True,
            ).stdout
            or ""
        ).strip()
    staged_stat = runner(
        (git, "-C", str(destination), "diff", "--cached", "--numstat", "--"),
        check=True,
    )
    unstaged = runner(
        (git, "-C", str(destination), "diff", "--quiet", "--"),
    )
    patched_status = runner(
        (
            git,
            "-C",
            str(destination),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ),
        check=True,
    )
    if (
        staged_blobs
        != {
            "setup.py": PATCHED_SETUP_BLOB,
            "third_party/BUILD": PATCHED_THIRD_PARTY_BLOB,
        }
        or (staged_stat.stdout or "").strip()
        != "1\t0\tthird_party/BUILD\n3\t4\tsetup.py"
        or unstaged.returncode != 0
        or (patched_status.stdout or "").strip() != "M  setup.py\nM  third_party/BUILD"
    ):
        raise SourceIntegrityError(
            "patched checkout differs from the exact approved diff"
        )
    return destination


def _x86_command(python: Path, *arguments: str) -> tuple[str, ...]:
    return (ARCH_COMMAND, "-x86_64", str(python), "-I", *arguments)


def _create_venv(
    packaged_python: Path,
    destination: Path,
    *,
    runner: CommandRunner,
) -> Path:
    runner(
        _x86_command(packaged_python, "-m", "venv", "--copies", str(destination)),
        env=_sanitized_environment(),
        check=True,
        capture_output=False,
    )
    return destination / "bin/python"


def _write_requirement_lock(
    path: Path,
    requirements: Sequence[LockedRequirement],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{item.name}=={item.version} --hash=sha256:{item.sha256}"
        for item in requirements
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _install_locked_requirements(
    python: Path,
    requirements: Sequence[LockedRequirement],
    lock_path: Path,
    *,
    runner: CommandRunner,
) -> None:
    lock = _write_requirement_lock(lock_path, requirements)
    runner(
        _x86_command(
            python,
            "-m",
            "pip",
            "install",
            "--only-binary=:all:",
            "--require-hashes",
            "-r",
            str(lock),
        ),
        env=_sanitized_environment(),
        check=True,
        capture_output=False,
    )


def prepare_build_environment(
    source: Path,
    packaged_python: Path,
    destination: Path,
    *,
    runner: CommandRunner = _run,
) -> Path:
    """Create an isolated x86_64 build environment with upstream requirements."""
    python = _create_venv(packaged_python, destination, runner=runner)
    _install_locked_requirements(
        python,
        BUILD_REQUIREMENTS,
        destination / "build-requirements.lock",
        runner=runner,
    )
    _install_locked_requirements(
        python,
        RUNTIME_REQUIREMENTS,
        destination / "runtime-requirements.lock",
        runner=runner,
    )
    return python


def _verify_source_build_contract(source: Path) -> None:
    try:
        setup = (source / "setup.py").read_text(encoding="utf-8")
    except OSError as error:
        raise SourceIntegrityError(
            "patched MediaPipe setup.py is unreadable"
        ) from error
    required = {
        "__version__ = '0.10.35'": "pinned MediaPipe version",
        "if not link_opencv and not IS_WINDOWS and not IS_MAC:": "shared OpenCV",
        "['--cpu=darwin_x86_64', '--ios_multi_cpus=x86_64']": "x86 build flags",
        "GPU_OPTIONS_DISABLED = ['--define=MEDIAPIPE_DISABLE_GPU=1']": "GPU off",
        "bazel_command.append('--define=OPENCV=source')": "source-built OpenCV path",
    }
    for marker, description in required.items():
        if marker not in setup:
            raise SourceIntegrityError(
                f"patched MediaPipe source lacks the {description}"
            )
    if "--define=ENABLE_ODML_CONVERTER=1" in setup:
        raise SourceIntegrityError(
            "patched MediaPipe source enables the unavailable ODML converter"
        )


def build_source_wheel(
    source: Path,
    output: Path,
    *,
    python: Path,
    bazelisk: Path,
    runner: CommandRunner = _run,
) -> Path:
    """Run upstream's source-OpenCV wheel build as x86_64 with GPU disabled."""
    source = Path(source)
    _verify_source_build_contract(source)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    before = set(output.glob("*.whl"))
    with tempfile.TemporaryDirectory(prefix="mediapipe-bazel-") as directory:
        tool_directory = Path(directory)
        bazelrc = tool_directory / "clang-compat.bazelrc"
        bazelrc.write_text(
            "build --per_file_copt=external/zlib/.*@-UTARGET_OS_MAC\n"
            "build --host_per_file_copt=external/zlib/.*@-UTARGET_OS_MAC\n",
            encoding="utf-8",
        )
        bazel = tool_directory / "bazel"
        bazel.write_text(
            f'#!/bin/sh\nif [ "$1" = "--version" ]; then\n'
            f'  exec {shlex.quote(str(Path(bazelisk).resolve()))} "$@"\nfi\n'
            f"exec {shlex.quote(str(Path(bazelisk).resolve()))} "
            f'--bazelrc={shlex.quote(str(bazelrc))} "$@"\n',
            encoding="utf-8",
        )
        bazel.chmod(0o700)
        environment = _sanitized_environment(
            {
                "PATH": os.pathsep.join(
                    (str(tool_directory), os.environ.get("PATH", ""))
                ),
                "MEDIAPIPE_DISABLE_GPU": "1",
                "CMAKE_OSX_ARCHITECTURES": "x86_64",
                "ARCHFLAGS": "-arch x86_64",
            }
        )
        runner(
            _x86_command(
                python,
                "setup.py",
                "bdist_wheel",
                "--dist-dir",
                str(output),
            ),
            cwd=source,
            env=environment,
            check=True,
            capture_output=False,
        )
    created = set(output.glob("*.whl")) - before
    if len(created) != 1:
        raise WheelVerificationError("source build did not produce exactly one wheel")
    return created.pop()


def _install_name(binary: Path, *, runner: CommandRunner) -> str:
    """A dylib's own LC_ID_DYLIB (otool -D), or '' for a bundle without one."""
    listing = runner((OTOOL, "-D", str(binary)), check=True)
    lines = [
        line.strip() for line in (listing.stdout or "").splitlines() if line.strip()
    ]
    return lines[1] if len(lines) > 1 else ""


def _linked_libraries(binary: Path, *, runner: CommandRunner) -> tuple[str, ...]:
    """The install names a Mach-O links (otool -L, minus its own id line)."""
    own = _install_name(binary, runner=runner)
    listing = runner((OTOOL, "-L", str(binary)), check=True)
    names: list[str] = []
    for line in (listing.stdout or "").splitlines()[1:]:
        name = line.strip().split(" (", 1)[0]
        if not name or name == own:
            continue
        if name.startswith("/System/Library/") or name.startswith("/usr/lib/"):
            continue
        names.append(name)
    return tuple(names)


def _rpaths(binary: Path, *, runner: CommandRunner) -> tuple[str, ...]:
    """Every LC_RPATH a Mach-O carries."""
    listing = runner((OTOOL, "-l", str(binary)), check=True)
    paths: list[str] = []
    lines = (listing.stdout or "").splitlines()
    for index, line in enumerate(lines):
        if line.strip() == "cmd LC_RPATH":
            for follow in lines[index + 1 : index + 4]:
                match = re.match(r"\s*path (.+?) \(offset \d+\)$", follow)
                if match:
                    paths.append(match.group(1))
                    break
    return tuple(paths)


def _relink_beside_loader(
    binary: Path, *, identity: str, runner: CommandRunner
) -> None:
    """Point a Mach-O's rpath at its own directory, name it, and ad-hoc sign it."""
    for rpath in _rpaths(binary, runner=runner):
        runner((INSTALL_NAME_TOOL, "-delete_rpath", rpath, str(binary)), check=True)
    runner((INSTALL_NAME_TOOL, "-add_rpath", "@loader_path", str(binary)), check=True)
    runner((INSTALL_NAME_TOOL, "-id", identity, str(binary)), check=True)
    runner((CODESIGN, "--force", "--sign", "-", str(binary)), check=True)


def _record_entry(root: Path, relative: str) -> list[str]:
    data = (root / relative).read_bytes()
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
    return [relative, f"sha256={digest.decode('ascii')}", str(len(data))]


def repair_darwin_wheel(
    wheel: Path,
    opencv_libraries: Path,
    *,
    output: Path,
    lipo: str = "/usr/bin/lipo",
    runner: CommandRunner = _run,
) -> Path:
    """Make a source-built Darwin wheel self-contained and loadable.

    Moves the Tasks C library to the name the Darwin bindings look for, copies
    the OpenCV dylibs it links beside it, points every rpath at ``@loader_path``,
    re-signs each library, and rewrites RECORD. Returns the repaired wheel.
    """
    wheel = Path(wheel)
    opencv_libraries = Path(opencv_libraries)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / wheel.name
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing wheel: {destination}")
    with tempfile.TemporaryDirectory(prefix="mediapipe-repair-") as directory:
        root = Path(directory)
        with zipfile.ZipFile(wheel) as archive:
            names = archive.namelist()
            if any(".." in name or name.startswith("/") for name in names):
                raise WheelVerificationError("wheel contains an unsafe path")
            archive.extractall(root)
        record_names = [name for name in names if name.endswith(".dist-info/RECORD")]
        if len(record_names) != 1:
            raise WheelVerificationError("wheel RECORD inventory is ambiguous")
        library_dir = root / TASKS_LIBRARY_DIR
        built = library_dir / BUILT_TASKS_LIBRARY
        if not built.is_file() or built.is_symlink():
            raise WheelVerificationError("wheel does not carry the built Tasks library")
        library = library_dir / DARWIN_TASKS_LIBRARY
        built.rename(library)

        bundled: dict[str, Path] = {}
        pending = [library]
        while pending:
            binary = pending.pop()
            for name in _linked_libraries(binary, runner=runner):
                if not name.startswith("@rpath/"):
                    raise WheelVerificationError(
                        f"Tasks library links outside the wheel: {name}"
                    )
                filename = name.removeprefix("@rpath/")
                if filename == DARWIN_TASKS_LIBRARY or filename in bundled:
                    continue
                source = opencv_libraries / filename
                if not source.is_file() or source.is_symlink():
                    raise WheelVerificationError(
                        f"linked library is unavailable: {filename}"
                    )
                architectures = runner((lipo, "-archs", str(source)), check=True)
                if "x86_64" not in (architectures.stdout or "").split():
                    raise WheelVerificationError(
                        f"linked library is not x86_64: {filename}"
                    )
                copied = library_dir / filename
                shutil.copyfile(source, copied)
                copied.chmod(0o755)
                bundled[filename] = copied
                pending.append(copied)
        if not bundled:
            raise WheelVerificationError("Tasks library links no OpenCV to bundle")

        _relink_beside_loader(
            library, identity=f"@rpath/{DARWIN_TASKS_LIBRARY}", runner=runner
        )
        for filename, copied in bundled.items():
            _relink_beside_loader(copied, identity=f"@rpath/{filename}", runner=runner)

        record_path = root / record_names[0]
        rows = list(csv.reader(io.StringIO(record_path.read_text(encoding="utf-8"))))
        replaced = f"{TASKS_LIBRARY_DIR}/{BUILT_TASKS_LIBRARY}"
        rows = [row for row in rows if row and row[0] != replaced]
        rows.append(_record_entry(root, f"{TASKS_LIBRARY_DIR}/{DARWIN_TASKS_LIBRARY}"))
        rows.extend(
            _record_entry(root, f"{TASKS_LIBRARY_DIR}/{filename}")
            for filename in sorted(bundled)
        )
        rows = [row for row in rows if row[0] != record_names[0]]
        rows.append([record_names[0], "", ""])
        buffer = io.StringIO()
        csv.writer(buffer, lineterminator="\n").writerows(rows)
        record_path.write_text(buffer.getvalue(), encoding="utf-8")

        members = sorted(path for path in root.rglob("*") if path.is_file())
        with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in members:
                archive.write(path, path.relative_to(root).as_posix())
    return destination


def verify_wheel_metadata(wheel: Path) -> None:
    """Require exact project/version metadata and a CPython 3.11 Intel tag."""
    try:
        with zipfile.ZipFile(wheel) as archive:
            names = archive.namelist()
            metadata_names = [
                name for name in names if name.endswith(".dist-info/METADATA")
            ]
            wheel_names = [name for name in names if name.endswith(".dist-info/WHEEL")]
            if len(metadata_names) != 1 or len(wheel_names) != 1:
                raise WheelVerificationError("wheel metadata inventory is ambiguous")
            metadata = Parser().parsestr(
                archive.read(metadata_names[0]).decode("utf-8")
            )
            wheel_metadata = Parser().parsestr(
                archive.read(wheel_names[0]).decode("utf-8")
            )
    except (OSError, UnicodeDecodeError, zipfile.BadZipFile) as error:
        raise WheelVerificationError("wheel metadata is unreadable") from error

    name = (metadata.get("Name") or "").lower().replace("_", "-")
    version = metadata.get("Version")
    if name != "mediapipe" or version != MEDIAPIPE_VERSION:
        raise WheelVerificationError(
            f"wheel must contain mediapipe {MEDIAPIPE_VERSION}, found {name} {version}"
        )
    tags = wheel_metadata.get_all("Tag", [])
    if not any(
        tag.startswith("cp311-cp311-macosx_") and tag.endswith("_x86_64")
        for tag in tags
    ):
        raise WheelVerificationError(
            "wheel must declare CPython 3.11 x86_64 compatibility"
        )


def verify_isolated_install(
    wheel: Path,
    *,
    packaged_python: Path,
    runtime: Path,
    lipo: str,
    runner: CommandRunner = _run,
) -> None:
    """Install only the local wheel, import under Rosetta, and inspect native code."""
    runtime_python = _create_venv(packaged_python, runtime, runner=runner)
    _install_locked_requirements(
        runtime_python,
        RUNTIME_REQUIREMENTS,
        runtime / "runtime-requirements.lock",
        runner=runner,
    )
    isolated_environment = _sanitized_environment()
    runner(
        _x86_command(
            runtime_python,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            "--no-cache-dir",
            str(wheel),
        ),
        env=isolated_environment,
        check=True,
        capture_output=False,
    )
    imported = runner(
        _x86_command(runtime_python, "-c", IMPORT_CHECK),
        env=isolated_environment,
        check=True,
    )
    import_lines = (imported.stdout or "").splitlines()
    if not import_lines or import_lines[0] != MEDIAPIPE_VERSION:
        raise WheelVerificationError(
            "isolated MediaPipe import reported the wrong version"
        )
    if len(import_lines) != 2:
        raise WheelVerificationError("isolated MediaPipe import path is missing")
    imported_file = Path(import_lines[1]).resolve()
    try:
        imported_file.relative_to(runtime.resolve())
    except ValueError as error:
        raise WheelVerificationError(
            "MediaPipe imported from outside the isolated runtime"
        ) from error
    libraries = runner(
        _x86_command(runtime_python, "-c", NATIVE_LIBRARY_QUERY),
        env=isolated_environment,
        check=True,
    )
    native_paths = [
        Path(line) for line in (libraries.stdout or "").splitlines() if line.strip()
    ]
    if not native_paths:
        raise WheelVerificationError("installed MediaPipe contains no native library")
    for native_path in native_paths:
        architectures = runner((lipo, "-archs", str(native_path)), check=True)
        if "x86_64" not in (architectures.stdout or "").split():
            raise WheelVerificationError(
                f"MediaPipe native library is not x86_64: {native_path}"
            )
    if not any(path.name == DARWIN_TASKS_LIBRARY for path in native_paths):
        raise WheelVerificationError(
            f"installed MediaPipe has no {DARWIN_TASKS_LIBRARY} for the Darwin bindings"
        )
    loaded = runner(
        _x86_command(runtime_python, "-c", TASKS_LIBRARY_LOAD_CHECK),
        env=isolated_environment,
        check=False,
    )
    if loaded.returncode != 0 or "tasks library loaded" not in (loaded.stdout or ""):
        raise WheelVerificationError(
            "the MediaPipe Tasks library does not load under Rosetta: "
            + (loaded.stderr or "").strip()[-400:]
        )


def build_wheel(
    output: Path,
    *,
    packaged_python: Path = DEFAULT_PACKAGED_PYTHON,
    runner: CommandRunner = _run,
    which: Which = shutil.which,
) -> Path:
    """Build and independently verify the official pinned Intel wheel."""
    tools = require_build_environment(packaged_python, runner=runner, which=which)
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mediapipe-x86_64-") as directory:
        workspace = Path(directory)
        source = checkout_official_source(
            workspace / "source", git=tools.git, runner=runner
        )
        build_python = prepare_build_environment(
            source,
            packaged_python,
            workspace / "build-environment",
            runner=runner,
        )
        built = build_source_wheel(
            source,
            workspace / "wheel",
            python=build_python,
            bazelisk=Path(tools.bazelisk),
            runner=runner,
        )
        wheel = repair_darwin_wheel(
            built,
            source / OPENCV_LIBRARY_DIR,
            output=workspace / "repaired",
            lipo=tools.lipo,
            runner=runner,
        )
        verify_wheel_metadata(wheel)
        verify_isolated_install(
            wheel,
            packaged_python=packaged_python,
            runtime=workspace / "verification-runtime",
            lipo=tools.lipo,
            runner=runner,
        )
        destination = output / wheel.name
        if destination.exists():
            raise FileExistsError(
                f"refusing to overwrite existing wheel: {destination}"
            )
        shutil.copy2(wheel, destination)
    return destination


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the pinned official MediaPipe 0.10.35 Intel macOS wheel."
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="directory that will receive the verified x86_64 wheel",
    )
    parser.add_argument(
        "--repair",
        type=Path,
        metavar="WHEEL",
        help="repair and re-verify this already-built source wheel instead of building",
    )
    parser.add_argument(
        "--opencv-libraries",
        type=Path,
        metavar="DIR",
        help="with --repair: the build's bazel-bin/third_party/opencv_cmake/lib",
    )
    return parser


def repair_built_wheel(
    wheel: Path,
    opencv_libraries: Path,
    output: Path,
    *,
    packaged_python: Path = DEFAULT_PACKAGED_PYTHON,
    runner: CommandRunner = _run,
    which: Which = shutil.which,
) -> Path:
    """Repair a wheel built earlier and put it through the same verification."""
    lipo = which("lipo")
    if lipo is None:
        raise BuildPrerequisiteError("missing required build tools: lipo")
    output = Path(output).expanduser().resolve()
    with tempfile.TemporaryDirectory(prefix="mediapipe-x86_64-repair-") as directory:
        workspace = Path(directory)
        repaired = repair_darwin_wheel(
            wheel,
            opencv_libraries,
            output=workspace / "repaired",
            lipo=lipo,
            runner=runner,
        )
        verify_wheel_metadata(repaired)
        verify_isolated_install(
            repaired,
            packaged_python=packaged_python,
            runtime=workspace / "verification-runtime",
            lipo=lipo,
            runner=runner,
        )
        output.mkdir(parents=True, exist_ok=True)
        destination = output / repaired.name
        if destination.exists():
            raise FileExistsError(
                f"refusing to overwrite existing wheel: {destination}"
            )
        shutil.copy2(repaired, destination)
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    parser = argument_parser()
    arguments = parser.parse_args(argv)
    if (arguments.repair is None) != (arguments.opencv_libraries is None):
        parser.error("--repair and --opencv-libraries go together")
    try:
        if arguments.repair is not None:
            wheel = repair_built_wheel(
                arguments.repair, arguments.opencv_libraries, arguments.output
            )
        else:
            wheel = build_wheel(arguments.output)
    except (
        MediaPipeWheelError,
        FileExistsError,
        OSError,
        subprocess.SubprocessError,
    ) as error:
        parser.exit(1, f"mediapipe wheel build failed: {error}\n")
    sys.stdout.write(f"{wheel}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
