"""
Scan ComfyUI/models/llm for .gguf files so the node can offer pickers (relative paths).
"""

from __future__ import annotations

import os
from typing import Final

from . import paths_helper

# First COMBO entry: use the STRING path fields only.
SENTINEL: Final[str] = "— use path fields below —"


def iter_llm_gguf_relpaths() -> list[str]:
    """All *.gguf paths relative to each registered llm root, forward slashes, sorted unique."""
    found: set[str] = set()
    for base in paths_helper.get_llm_root_dirs():
        if not base or not os.path.isdir(base):
            continue
        base_abs = os.path.abspath(os.path.normpath(base))
        for dirpath, _, files in os.walk(base_abs):
            for f in files:
                if not f.lower().endswith(".gguf"):
                    continue
                full = os.path.join(dirpath, f)
                try:
                    rel = os.path.relpath(full, base_abs).replace("\\", "/")
                except ValueError:
                    continue
                found.add(rel)
    return sorted(found)


def combo_choices_llm_gguf() -> list[str]:
    """COMBO options: sentinel + every .gguf under models/llm (any subfolder)."""
    items = iter_llm_gguf_relpaths()
    return [SENTINEL] + items


def suggest_default_main_and_mmproj() -> tuple[str, str]:
    """
    Pick sensible defaults for new nodes: main LLM .gguf + mmproj .gguf from models/llm scan.
    Falls back to SENTINEL when unknown or missing files.
    """
    paths = iter_llm_gguf_relpaths()
    choices = combo_choices_llm_gguf()
    if not paths:
        return (SENTINEL, SENTINEL)

    mmproj_pick = ""
    for p in paths:
        if "mmproj" in p.lower():
            mmproj_pick = p
            break

    main_pick = ""
    for p in paths:
        if p == mmproj_pick:
            continue
        base = os.path.basename(p).lower()
        if "mmproj" not in base:
            main_pick = p
            break
    if not main_pick:
        for p in paths:
            if p != mmproj_pick:
                main_pick = p
                break

    main_def = main_pick if main_pick and main_pick in choices else SENTINEL
    mm_def = mmproj_pick if mmproj_pick and mmproj_pick in choices else SENTINEL
    return (main_def, mm_def)
