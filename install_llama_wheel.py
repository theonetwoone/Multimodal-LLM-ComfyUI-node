"""
CLI: resolve best JamePeng-style llama-cpp-python wheel from GitHub Releases and optionally pip install.

  python -m llm_comfy_multimodal.install_llama_wheel
  python -m llm_comfy_multimodal.install_llama_wheel --install

Environment GITHUB_TOKEN is optional (raises GitHub API rate limits when unset).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

from . import env_probe, wheel_resolver


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Pick a matching llama-cpp-python wheel from GitHub Releases (same rules as the Comfy node)."
    )
    p.add_argument(
        "--repo",
        default=wheel_resolver.DEFAULT_REPO,
        help=f"owner/repo on GitHub (default: {wheel_resolver.DEFAULT_REPO})",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=45.0,
        help="seconds for each GitHub API request",
    )
    p.add_argument(
        "--install",
        action="store_true",
        help="run pip uninstall + pip install URL for the best match (otherwise print only)",
    )
    ns = p.parse_args(argv)

    profile = env_probe.build_install_profile()
    repo = (ns.repo or "").strip() or wheel_resolver.DEFAULT_REPO
    try:
        best, alts = wheel_resolver.resolve_best_wheel(
            profile, repo=repo, timeout=ns.timeout
        )
    except Exception as e:
        print(f"[llm_comfy_multimodal] {e}", file=sys.stderr)
        return 1

    text = wheel_resolver.format_resolution_report(
        sys.executable, profile, repo, best, alts
    )
    print(text)

    if ns.install and best is not None:
        pip_base = [sys.executable, "-m", "pip"]
        subprocess.run([*pip_base, "uninstall", "-y", "llama-cpp-python"], check=False)
        subprocess.run([*pip_base, "install", best.url], check=True)
        print("\nInstalled. Restart ComfyUI before using GGUF vision.", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
