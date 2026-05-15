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
        _mock = MagicMock()
        if _mod == "torch":
            # scipy's import chain calls issubclass(cls, torch.Tensor) at
            # module level (scipy/stats/_new_distributions.py docstring
            # generation).  If torch.Tensor is a MagicMock instance rather
            # than a real class, Python raises:
            #   TypeError: issubclass() arg 2 must be a class, …
            # Making it a real (empty) class avoids this.
            _mock.Tensor = type("Tensor", (), {})
        sys.modules[_mod] = _mock
