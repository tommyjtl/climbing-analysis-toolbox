"""Tests for the CLI argument parser (cruxes command).

All tests are in-process: sys.argv is patched and heavy downstream work is
mocked so no real video processing or model inference takes place.
"""

import sys
import pytest
from unittest.mock import patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_body_trajectory(*extra_args):
    """Invoke the body-trajectory subcommand and return the kwargs that would
    be forwarded to Cruxes.body_trajectory, without actually running it."""
    from cruxes import Cruxes
    from cruxes.cli import main

    captured: dict = {}

    def _fake(self, video_path, **kwargs):
        captured["video_path"] = video_path
        captured.update(kwargs)

    argv = ["cruxes", "body-trajectory", "--video_path", "test.mp4"] + list(extra_args)
    with patch.object(Cruxes, "body_trajectory", _fake):
        with patch.object(sys, "argv", argv):
            main()

    return captured


# ---------------------------------------------------------------------------
# Smoke tests – help / usage
# ---------------------------------------------------------------------------


def test_cli_help_exits_zero():
    from cruxes.cli import main

    with patch.object(sys, "argv", ["cruxes", "--help"]):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 0


def test_cli_body_trajectory_help_exits_zero():
    from cruxes.cli import main

    with patch.object(sys, "argv", ["cruxes", "body-trajectory", "--help"]):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 0


def test_cli_warp_help_exits_zero():
    from cruxes.cli import main

    with patch.object(sys, "argv", ["cruxes", "warp", "--help"]):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 0


def test_cli_warp_image_help_exits_zero():
    from cruxes.cli import main

    with patch.object(sys, "argv", ["cruxes", "warp-image", "--help"]):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 0


def test_cli_no_subcommand_exits_nonzero():
    from cruxes.cli import main

    with patch.object(sys, "argv", ["cruxes"]):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code != 0


def test_cli_missing_video_path_exits_nonzero():
    from cruxes.cli import main

    with patch.object(sys, "argv", ["cruxes", "body-trajectory"]):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code != 0


# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------


def test_cli_default_smoothing_is_gaussian():
    assert _parse_body_trajectory().get("smoothing") == "gaussian"


def test_cli_default_pose_backend_is_mediapipe():
    assert _parse_body_trajectory().get("pose_backend") == "mediapipe"


def test_cli_default_json_only_is_false():
    assert _parse_body_trajectory().get("json_only") is False


def test_cli_default_trajectory_only_is_false():
    assert _parse_body_trajectory().get("trajectory_only") is False


def test_cli_default_draw_pose_is_false():
    # draw_pose defaults to False in CLI (user must opt-in)
    assert _parse_body_trajectory().get("draw_pose") is False


# ---------------------------------------------------------------------------
# Explicit valid choices
# ---------------------------------------------------------------------------


def test_cli_smoothing_gaussian_explicit():
    assert (
        _parse_body_trajectory("--smoothing", "gaussian").get("smoothing") == "gaussian"
    )


def test_cli_smoothing_savgol():
    assert _parse_body_trajectory("--smoothing", "savgol").get("smoothing") == "savgol"


def test_cli_smoothing_smoothnet():
    assert (
        _parse_body_trajectory("--smoothing", "smoothnet").get("smoothing")
        == "smoothnet"
    )


def test_cli_pose_backend_vitpose():
    assert (
        _parse_body_trajectory("--pose_backend", "vitpose").get("pose_backend")
        == "vitpose"
    )


def test_cli_pose_backend_mediapipe_explicit():
    assert (
        _parse_body_trajectory("--pose_backend", "mediapipe").get("pose_backend")
        == "mediapipe"
    )


# ---------------------------------------------------------------------------
# Invalid choices are rejected by argparse
# ---------------------------------------------------------------------------


def test_cli_invalid_smoothing_exits_nonzero():
    from cruxes.cli import main

    with patch.object(
        sys,
        "argv",
        [
            "cruxes",
            "body-trajectory",
            "--video_path",
            "test.mp4",
            "--smoothing",
            "invalid",
        ],
    ):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code != 0


def test_cli_invalid_pose_backend_exits_nonzero():
    from cruxes.cli import main

    with patch.object(
        sys,
        "argv",
        [
            "cruxes",
            "body-trajectory",
            "--video_path",
            "test.mp4",
            "--pose_backend",
            "invalid",
        ],
    ):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code != 0


# ---------------------------------------------------------------------------
# Argument forwarding
# ---------------------------------------------------------------------------


def test_cli_track_points_forwarded_as_list():
    cap = _parse_body_trajectory("--track_point", "hip_mid,left_hand,right_hand")
    assert cap.get("track_point") == ["hip_mid", "left_hand", "right_hand"]


def test_cli_track_point_single():
    cap = _parse_body_trajectory("--track_point", "hip_mid")
    assert cap.get("track_point") == ["hip_mid"]


def test_cli_track_point_whitespace_stripped():
    cap = _parse_body_trajectory("--track_point", " hip_mid , left_hand ")
    assert cap.get("track_point") == ["hip_mid", "left_hand"]


def test_cli_json_only_flag():
    assert _parse_body_trajectory("--json_only").get("json_only") is True


def test_cli_trajectory_only_flag():
    assert _parse_body_trajectory("--trajectory_only").get("trajectory_only") is True


def test_cli_show_trajectory_flag():
    assert _parse_body_trajectory("--show_trajectory").get("show_trajectory") is True


def test_cli_draw_pose_flag():
    assert _parse_body_trajectory("--draw_pose").get("draw_pose") is True


def test_cli_use_cached_landmarks_flag():
    assert (
        _parse_body_trajectory("--use_cached_landmarks").get("use_cached_landmarks")
        is True
    )


def test_cli_export_landmarks_flag():
    assert _parse_body_trajectory("--export_landmarks").get("export_landmarks") is True


def test_cli_kalman_settings_forwarded():
    cap = _parse_body_trajectory("--kalman_settings", "0.5")
    # kalman_settings becomes [True, 0.5] when supplied
    kalman = cap.get("kalman_settings")
    assert kalman is not None
    assert kalman[0] is True
    assert kalman[1] == pytest.approx(0.5)


def test_cli_kalman_settings_default_disabled():
    cap = _parse_body_trajectory()
    kalman = cap.get("kalman_settings")
    assert kalman is not None
    assert kalman[0] is False


def test_cli_trajectory_history_seconds():
    cap = _parse_body_trajectory("--trajectory_history_seconds", "3.5")
    assert cap.get("trajectory_history_seconds") == pytest.approx(3.5)


def test_cli_visibility_threshold_forwarded():
    cap = _parse_body_trajectory("--pose_visibility_threshold", "0.5")
    assert cap.get("pose_visibility_threshold") == pytest.approx(0.5)


def test_cli_presence_threshold_forwarded():
    cap = _parse_body_trajectory("--pose_presence_threshold", "0.3")
    assert cap.get("pose_presence_threshold") == pytest.approx(0.3)
