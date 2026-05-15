"""Pytest configuration.

Heavy dependencies (cv2, mediapipe, torch, …) are mocked when they are not
installed so the test suite can run in a lightweight CI environment without
pulling in gigabytes of ML packages.
"""
import sys
from unittest.mock import MagicMock

_MOCK_IF_MISSING = ("cv2", "mediapipe", "torch", "torchvision", "torchaudio")

for _mod in _MOCK_IF_MISSING:
    try:
        __import__(_mod)
    except ImportError:
        sys.modules[_mod] = MagicMock()
