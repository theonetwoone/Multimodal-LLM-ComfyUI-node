"""
Register ComfyUI/models/llm so local paths can be short names inside that folder.
"""

from __future__ import annotations

import logging
import os


def _register_llm_folder() -> None:
    try:
        import folder_paths

        p = os.path.join(folder_paths.models_dir, "llm")
        os.makedirs(p, exist_ok=True)
        folder_paths.add_model_folder_path("llm", p, is_default=True)
    except Exception as e:
        logging.getLogger(__name__).warning(
            "llm_comfy_multimodal: could not register models/llm (%s)", e
        )


_register_llm_folder()


def get_llm_root_dirs() -> list[str]:
    try:
        import folder_paths

        return folder_paths.get_folder_paths("llm")
    except Exception:
        return []
