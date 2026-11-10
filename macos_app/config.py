from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


APP_VERSION = "0.4.7"
RELEASE_TAG = "v0.4.7"
MINIMUM_MACOS = "13.0"
PYTHON_VERSION = "3.11.15"
MEDIAPIPE_VERSION = "0.10.35"
MEDIAPIPE_COMMIT = "f8ef212d5c962c0e853db7e59d217056b187084b"
ARCHITECTURES = ("arm64", "x86_64")


@dataclass(frozen=True)
class Download:
    name: str
    url: str
    sha256: str


PYTHON_BUILDS: Mapping[str, Download] = MappingProxyType(
    {
        "arm64": Download(
            name="cpython-3.11.15+20260807-aarch64-apple-darwin-install_only.tar.gz",
            url=(
                "https://github.com/astral-sh/python-build-standalone/releases/download/"
                "20260807/cpython-3.11.15%2B20260807-aarch64-apple-darwin-install_only.tar.gz"
            ),
            sha256="f7783d9a56a75d91554c08409a674e8678d56d42555fdbb365d8190d5cc93659",
        ),
        "x86_64": Download(
            name="cpython-3.11.15+20260807-x86_64-apple-darwin-install_only.tar.gz",
            url=(
                "https://github.com/astral-sh/python-build-standalone/releases/download/"
                "20260807/cpython-3.11.15%2B20260807-x86_64-apple-darwin-install_only.tar.gz"
            ),
            sha256="d1842391797a9478fc1de67c8575f275c1cdf2e93a8d1deb991939dec3ec9bbf",
        ),
    }
)

MODEL_ASSETS: Mapping[str, Download] = MappingProxyType(
    {
        "forehead_context.bin": Download(
            name="forehead_context.bin",
            url=("https://storage.googleapis.com/mediapipe-models/image_segmenter/"
                 "selfie_multiclass_256x256/float32/1/selfie_multiclass_256x256.tflite"),
            sha256="c6748b1253a99067ef71f7e26ca71096cd449baefa8f101900ea23016507e0e0",
        ),
        "face_landmarker.task": Download(
            name="face_landmarker.task",
            url=(
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                "face_landmarker/float16/1/face_landmarker.task"
            ),
            sha256="64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
        ),
        "selfie_segmenter.tflite": Download(
            name="selfie_segmenter.tflite",
            url=(
                "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
                "selfie_segmenter/float16/latest/selfie_segmenter.tflite"
            ),
            sha256="191ac9529ae506ee0beefa6b2c945a172dab9d07d1e802a290a4e4038226658b",
        ),
        "canonical_face_model.obj": Download(
            name="canonical_face_model.obj",
            url=(
                "https://raw.githubusercontent.com/google-ai-edge/mediapipe/master/"
                "mediapipe/modules/face_geometry/data/canonical_face_model.obj"
            ),
            sha256="8bac80443397e113f41a8b565ea72c59390bc031d9defab289dba7bc0c54e618",
        ),
    }
)
