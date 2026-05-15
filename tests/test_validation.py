"""Tests for module-level constants and input-validation logic."""

import pytest


def test_default_track_point_visibility_threshold():
    from cruxes.utils.body_trajectory import DEFAULT_TRACK_POINT_VISIBILITY_THRESHOLD

    assert isinstance(DEFAULT_TRACK_POINT_VISIBILITY_THRESHOLD, float)
    assert 0.0 <= DEFAULT_TRACK_POINT_VISIBILITY_THRESHOLD <= 1.0
    assert DEFAULT_TRACK_POINT_VISIBILITY_THRESHOLD == 0.6


def test_visibility_threshold():
    from cruxes.utils.pose_backend import VISIBILITY_THRESHOLD

    assert isinstance(VISIBILITY_THRESHOLD, float)
    assert VISIBILITY_THRESHOLD == 0.2


def test_presence_threshold():
    from cruxes.utils.pose_backend import PRESENCE_THRESHOLD

    assert isinstance(PRESENCE_THRESHOLD, float)
    assert PRESENCE_THRESHOLD == 0.2


def test_pose_connections_is_list_of_int_tuples():
    from cruxes.utils.pose_backend import POSE_CONNECTIONS

    assert isinstance(POSE_CONNECTIONS, list)
    assert len(POSE_CONNECTIONS) > 0
    for conn in POSE_CONNECTIONS:
        assert isinstance(conn, tuple)
        assert len(conn) == 2
        assert isinstance(conn[0], int)
        assert isinstance(conn[1], int)


def test_supported_smoothing_methods_are_known():
    """The CLI --smoothing choices must include exactly the documented methods."""
    import argparse
    import sys
    from unittest.mock import patch

    from cruxes.cli import main

    captured_choices: list = []

    _real_add_argument = argparse.ArgumentParser.add_argument

    def _spy_add_argument(self, *args, **kwargs):
        if "--smoothing" in args:
            choices = kwargs.get("choices")
            if choices is not None:
                captured_choices.extend(choices)
        return _real_add_argument(self, *args, **kwargs)

    with patch.object(argparse.ArgumentParser, "add_argument", _spy_add_argument):
        with patch.object(sys, "argv", ["cruxes", "body-trajectory", "--help"]):
            with pytest.raises(SystemExit):
                main()

    # "none" is the sentinel for disabling smoothing; the three real methods are:
    expected_methods = {"gaussian", "savgol", "smoothnet"}
    actual_methods = {c for c in captured_choices if c != "none"}
    assert (
        actual_methods == expected_methods
    ), f"CLI --smoothing choices {actual_methods!r} don't match expected {expected_methods!r}"
