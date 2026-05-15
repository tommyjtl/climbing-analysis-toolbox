"""Integration tests – exercise actual video processing pipelines.

These tests require the full ML stack (opencv-python, mediapipe, torch, …)
and, for body-trajectory tests, a real video file.  They are skipped
automatically in lightweight CI and in any environment where the heavy
dependencies are not installed.

Running locally
---------------
1.  Install the full package::

        pip install -e ".[vitpose]"

2.  Point at a test video (any short clip works)::

        export CRUXES_TEST_VIDEO=/path/to/clip.mp4

3.  Run only integration tests::

        pytest tests/ -m integration -v

    Or run the whole suite including integration tests::

        pytest tests/ -v

If CRUXES_TEST_VIDEO is not set, all video-processing tests are skipped.
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ── Skip guard ──────────────────────────────────────────────────────────────

# conftest.py mocks cv2 when it is not installed.  A MagicMock is never the
# real cv2 module, so this check reliably detects the lightweight CI env.
_HEAVY_DEPS_AVAILABLE = not isinstance(sys.modules.get("cv2"), MagicMock)

pytestmark = pytest.mark.skipif(
    not _HEAVY_DEPS_AVAILABLE,
    reason=(
        "Integration tests skipped: full ML stack not installed "
        "(cv2 / mediapipe / torch missing). "
        "Install with `pip install -e .` and re-run."
    ),
)

# ── Test data ────────────────────────────────────────────────────────────────

_ENV_VIDEO = os.environ.get("CRUXES_TEST_VIDEO", "")
_TEST_VIDEO_PATH: str | None = (
    _ENV_VIDEO if _ENV_VIDEO and Path(_ENV_VIDEO).is_file() else None
)

_REPO_ROOT = Path(__file__).parent.parent
_EXAMPLES_LANDMARKS_DIR = _REPO_ROOT / "examples" / "videos" / "tests"

# ── Helpers ──────────────────────────────────────────────────────────────────


def _require_test_video():
    """Skip the calling test if no test video is available."""
    if _TEST_VIDEO_PATH is None:
        pytest.skip(
            "No test video available. "
            "Set CRUXES_TEST_VIDEO=/path/to/clip.mp4 to enable this test."
        )


# ── Warp tests ────────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_warp_image_small_synthetic(tmp_path):
    """warp_image() completes (or fails gracefully) on synthetic images.

    The image-matching model is unlikely to find keypoints in random noise;
    this test verifies that the error path is handled without an unhandled
    exception.
    """
    import cv2
    import numpy as np
    from cruxes import Cruxes

    # Create two tiny random-noise images (10×10 – no real keypoints)
    ref_path = tmp_path / "ref.png"
    src_path = tmp_path / "src.png"
    cv2.imwrite(str(ref_path), np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8))
    cv2.imwrite(str(src_path), np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8))

    c = Cruxes()
    # Must not raise; result may be False if no homography found
    try:
        c.warp_image(ref_img=str(ref_path), src_img_path=str(src_path))
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"warp_image raised an unexpected exception: {exc}")


# ── Body-trajectory tests ─────────────────────────────────────────────────────


@pytest.mark.integration
def test_body_trajectory_json_only(tmp_path):
    """body_trajectory(json_only=True) exports landmarks without rendering video."""
    _require_test_video()

    import shutil
    from cruxes import Cruxes

    assert _TEST_VIDEO_PATH is not None  # narrowed by _require_test_video
    # Copy into tmp so the landmark JSON ends up there too
    video_copy = tmp_path / Path(_TEST_VIDEO_PATH).name
    shutil.copy2(_TEST_VIDEO_PATH, video_copy)

    c = Cruxes()
    c.body_trajectory(
        str(video_copy),
        json_only=True,
        export_landmarks=True,
        smoothing="gaussian",
        pose_backend="mediapipe",
    )

    landmarks_json = tmp_path / (video_copy.stem + "_landmarks.json")
    assert landmarks_json.exists(), "Landmarks JSON was not created"
    assert landmarks_json.stat().st_size > 0, "Landmarks JSON is empty"


@pytest.mark.integration
def test_body_trajectory_cached_landmarks(tmp_path):
    """body_trajectory() can render using a pre-existing landmarks JSON cache."""
    _require_test_video()

    import json
    import shutil
    from cruxes import Cruxes

    assert _TEST_VIDEO_PATH is not None
    video_copy = tmp_path / Path(_TEST_VIDEO_PATH).name
    shutil.copy2(_TEST_VIDEO_PATH, video_copy)

    # First pass: extract landmarks to cache
    c = Cruxes()
    c.body_trajectory(
        str(video_copy),
        json_only=True,
        export_landmarks=True,
        smoothing="gaussian",
        pose_backend="mediapipe",
    )

    landmarks_json = tmp_path / (video_copy.stem + "_landmarks.json")
    assert landmarks_json.exists()

    # Second pass: reuse cache – should be faster and produce identical data
    c.body_trajectory(
        str(video_copy),
        json_only=True,
        use_cached_landmarks=True,
        landmarks_json_path=str(landmarks_json),
        smoothing="gaussian",
        pose_backend="mediapipe",
    )


@pytest.mark.integration
def test_body_trajectory_smoothing_methods(tmp_path):
    """All documented smoothing methods run without error."""
    _require_test_video()

    import shutil
    from cruxes import Cruxes

    assert _TEST_VIDEO_PATH is not None
    video_copy = tmp_path / Path(_TEST_VIDEO_PATH).name
    shutil.copy2(_TEST_VIDEO_PATH, video_copy)

    c = Cruxes()
    for method in ("gaussian", "savgol"):
        c.body_trajectory(
            str(video_copy),
            json_only=True,
            smoothing=method,
            pose_backend="mediapipe",
        )


@pytest.mark.integration
def test_body_trajectory_invalid_smoothing_raises():
    """Passing an unsupported smoothing method raises ValueError before any I/O."""
    _require_test_video()

    from cruxes import Cruxes

    assert _TEST_VIDEO_PATH is not None
    c = Cruxes()
    with pytest.raises(ValueError, match="Unsupported smoothing method"):
        c.body_trajectory(
            _TEST_VIDEO_PATH,
            smoothing="not_a_real_method",
            pose_backend="mediapipe",
        )
