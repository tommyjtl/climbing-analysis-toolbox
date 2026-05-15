"""Tests for the Cruxes Python class interface."""

import sys
import pytest
from unittest.mock import MagicMock


def test_cruxes_default_instantiation():
    from cruxes import Cruxes

    c = Cruxes()
    assert c.matcher_model_name == "superpoint-lightglue"
    assert c.matcher_device == "auto"


def test_cruxes_custom_instantiation():
    from cruxes import Cruxes

    c = Cruxes(matcher_model_name="disk-lightglue", matcher_device="cpu")
    assert c.matcher_model_name == "disk-lightglue"
    assert c.matcher_device == "cpu"


def test_set_matcher_model_name():
    from cruxes import Cruxes

    c = Cruxes()
    result = c.set_matcher_model_name("disk-lightglue")
    assert result == "disk-lightglue"
    assert c.matcher_model_name == "disk-lightglue"


def test_set_matcher_device():
    from cruxes import Cruxes

    c = Cruxes()
    result = c.set_matcher_device("cpu")
    assert result == "cpu"
    assert c.matcher_device == "cpu"


def test_get_default_matcher_device_explicit():
    """When matcher_device is set to a real value it is returned as-is."""
    from cruxes import Cruxes

    for device in ("cpu", "cuda", "mps"):
        c = Cruxes(matcher_device=device)
        assert c._get_default_matcher_device() == device


def test_get_default_matcher_device_auto_fallback_to_cpu(monkeypatch):
    """With neither CUDA nor MPS available the device should fall back to 'cpu'."""
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.backends.mps.is_available.return_value = False
    monkeypatch.setitem(sys.modules, "torch", mock_torch)

    from cruxes import Cruxes

    c = Cruxes(matcher_device="auto")
    assert c._get_default_matcher_device() == "cpu"


def test_body_trajectory_method_exists():
    from cruxes import Cruxes

    assert callable(getattr(Cruxes, "body_trajectory", None))


def test_warp_video_method_exists():
    from cruxes import Cruxes

    assert callable(getattr(Cruxes, "warp_video", None))


def test_warp_image_method_exists():
    from cruxes import Cruxes

    assert callable(getattr(Cruxes, "warp_image", None))


def test_body_trajectory_default_smoothing():
    """body_trajectory() default keyword argument for smoothing is 'gaussian'."""
    import inspect
    from cruxes import Cruxes

    sig = inspect.signature(Cruxes.body_trajectory)
    assert sig.parameters["smoothing"].default == "gaussian"


def test_body_trajectory_default_draw_pose():
    """body_trajectory() should draw pose by default."""
    import inspect
    from cruxes import Cruxes

    sig = inspect.signature(Cruxes.body_trajectory)
    assert sig.parameters["draw_pose"].default is True


def test_body_trajectory_default_show_trajectory():
    """body_trajectory() should show trajectory by default."""
    import inspect
    from cruxes import Cruxes

    sig = inspect.signature(Cruxes.body_trajectory)
    assert sig.parameters["show_trajectory"].default is True


# ---------------------------------------------------------------------------
# File-existence guards – no ML inference, no heavy deps needed
# ---------------------------------------------------------------------------

def test_warp_image_missing_ref_returns_false(tmp_path):
    """warp_image() returns False and prints a warning when ref image is missing."""
    from cruxes import Cruxes

    c = Cruxes()
    result = c.warp_image(
        ref_img=str(tmp_path / "nonexistent_ref.png"),
        src_img_path=str(tmp_path / "nonexistent_src.png"),
    )
    assert result is False


def test_warp_image_missing_src_returns_false(tmp_path):
    """warp_image() returns False when only the source image is missing."""
    import numpy as np
    import cv2 as cv2_mod  # cv2 may be real or a mock – both support this test

    from cruxes import Cruxes

    # Write a tiny real-looking PNG for the reference
    ref_path = tmp_path / "ref.png"
    try:
        real_write = cv2_mod.imwrite(str(ref_path), np.zeros((10, 10, 3), dtype=np.uint8))
        if not real_write:
            raise RuntimeError("cv2.imwrite failed (may be a mock)")
    except Exception:
        # cv2 is mocked – just create an empty file so os.path.exists passes
        ref_path.write_bytes(b"\x00")

    c = Cruxes()
    result = c.warp_image(
        ref_img=str(ref_path),
        src_img_path=str(tmp_path / "nonexistent_src.png"),
    )
    assert result is False


def test_warp_video_missing_ref_returns_early(tmp_path, capsys):
    """warp_video() prints a warning and returns early when ref image is missing."""
    from cruxes import Cruxes

    c = Cruxes()
    # Should not raise – just warn and return
    c.warp_video(
        ref_img=str(tmp_path / "nonexistent_ref.png"),
        src_video_path=str(tmp_path / "nonexistent.mp4"),
    )
    captured = capsys.readouterr()
    assert "Warning" in captured.out
