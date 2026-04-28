from __future__ import annotations

import pytest

from crforest import _gpu_detect


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires CUDA + cupy + nvidia-cuda-nvrtc")


def pytest_collection_modifyitems(config, items):
    available, reason = _gpu_detect.detect_cuda()
    if available:
        return
    skip_gpu = pytest.mark.skip(reason=f"GPU unavailable: {reason}")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
