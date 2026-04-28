"""Allow: python -m llm_comfy_multimodal (same as install_llama_wheel CLI)."""

from __future__ import annotations

from .install_llama_wheel import main

if __name__ == "__main__":
    raise SystemExit(main())
