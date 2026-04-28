"""
Detect Python / OS / PyTorch CUDA tag for matching third-party wheels (llama-cpp-python).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import FrozenSet, Optional


def pip_cuda_tag_from_torch() -> Optional[str]:
    """Map torch.version.cuda like '12.4' -> 'cu124' for matching wheel filenames."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    raw = (torch.version.cuda or "").strip()
    if not raw:
        return None
    parts = raw.replace(" ", "").split(".")
    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
        return f"cu{int(parts[0]):d}{int(parts[1]):d}"
    return None


def preferred_python_cp_tag() -> str:
    v = sys.version_info
    return f"cp{v.major}{v.minor}"


def compatible_wheel_platform_strings() -> FrozenSet[str]:
    """Platform strings like win_amd64, manylinux_2_17_x86_64 from packaging.tags."""
    try:
        import packaging.tags

        return frozenset({t.platform for t in packaging.tags.sys_tags()})
    except Exception:
        if sys.platform == "win32":
            return frozenset({"win_amd64"})
        if sys.platform == "darwin":
            return frozenset({"macosx_10_13_x86_64", "macosx_11_0_arm64"})
        return frozenset({"manylinux_2_17_x86_64", "linux_x86_64"})


def torch_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


@dataclass(frozen=True)
class InstallProfile:
    """What to match against GitHub release .whl asset names."""

    python_cp_tag: str
    platform_strings: FrozenSet[str]
    cuda_hint_tag: Optional[str]
    prefer_cuda: bool


def build_install_profile() -> InstallProfile:
    return InstallProfile(
        python_cp_tag=preferred_python_cp_tag(),
        platform_strings=compatible_wheel_platform_strings(),
        cuda_hint_tag=pip_cuda_tag_from_torch(),
        prefer_cuda=torch_cuda_available(),
    )
