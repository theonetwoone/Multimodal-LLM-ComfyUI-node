"""
Resolve the best-matching llama-cpp-python wheel from GitHub Releases (no extra deps).
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterator, Optional

from packaging.utils import InvalidWheelFilename, parse_wheel_filename
from packaging.version import InvalidVersion, Version

from .env_probe import InstallProfile

GITHUB_API = "https://api.github.com"
DEFAULT_REPO = "JamePeng/llama-cpp-python"
UA = "llm-comfy-multimodal-wheel-resolver/1.0"


@dataclass(frozen=True)
class ResolvedWheel:
    filename: str
    url: str
    score: float
    reason: str
    package_version: str


def _cu_to_int(tag: Optional[str]) -> Optional[int]:
    if not tag or not tag.startswith("cu"):
        return None
    rest = tag[2:]
    if not rest.isdigit():
        return None
    return int(rest)


def _infer_cuda_tokens_from_text(name: str) -> list[str]:
    lower = name.lower()
    return list(dict.fromkeys(re.findall(r"cu\d{3,4}", lower)))


def _wheel_interpreter_tags_ok(filename: str, profile: InstallProfile) -> bool:
    """True if wheel supports this CPython version (e.g. cp312)."""
    want = profile.python_cp_tag.lower()
    try:
        _dist, _ver, _build, tags = parse_wheel_filename(filename)
        for t in tags:
            inter = (getattr(t, "interpreter", "") or "").lower()
            if inter == want or inter.startswith("py3") or inter == "py2.py3":
                return True
        return False
    except (InvalidWheelFilename, ValueError):
        pass
    # Legacy / odd filenames: require cpXYZ substring
    return want in filename.lower()


def _wheel_platform_ok(filename: str, profile: InstallProfile) -> bool:
    try:
        _dist, _ver, _build, tags = parse_wheel_filename(filename)
        plat_strings = {getattr(t, "platform", "") for t in tags}
        plat_strings.discard("")
        if plat_strings & profile.platform_strings:
            return True
    except (InvalidWheelFilename, ValueError):
        pass
    fn_lower = filename.lower()
    for p in profile.platform_strings:
        if p.lower() in fn_lower:
            return True
    return False


def _parse_wheel_version_string(filename: str) -> str:
    try:
        _dist, ver, _build, _tags = parse_wheel_filename(filename)
        return str(ver)
    except Exception:
        m = re.match(r"^[Ll]lama[_-]?cpp[_-]?python-([^-]+(?:-[^-]+)?)", filename)
        return m.group(1) if m else "0"


def _score_wheel(filename: str, profile: InstallProfile) -> tuple[float, str]:
    """
    Higher is better. Filters incompatible wheels via very low scores (caller skips < 0).
    """
    if not filename.endswith(".whl"):
        return -1.0, "not a .whl"

    if not _wheel_interpreter_tags_ok(filename, profile):
        return -1.0, f"no {profile.python_cp_tag} tag"

    if not _wheel_platform_ok(filename, profile):
        return -1.0, "platform mismatch"

    tokens = _infer_cuda_tokens_from_text(filename)
    has_cuda_token = bool(tokens)
    pref = profile.cuda_hint_tag
    pref_i = _cu_to_int(pref)

    parts: list[str] = []

    if profile.prefer_cuda:
        if has_cuda_token:
            best_cuda = 0.0
            for tok in tokens:
                ti = _cu_to_int(tok)
                if pref_i is not None and ti is not None:
                    dist = abs(ti - pref_i)
                    s = 100.0 - min(dist, 50.0) * 0.8
                elif ti is not None:
                    s = 70.0
                else:
                    s = 50.0
                best_cuda = max(best_cuda, s)
            cuda_score = best_cuda
            parts.append(f"CUDA tokens {tokens} score~{cuda_score:.1f} vs pref {pref!r}")
        else:
            cuda_score = 55.0
            parts.append("no cuNNN in name (may still be CUDA build); score 55")
    else:
        if has_cuda_token:
            cuda_score = 40.0
            parts.append("CUDA wheel on CPU-only profile; score 40")
        else:
            cuda_score = 90.0
            parts.append("non-CUDA naming; score 90")

    ver_s = _parse_wheel_version_string(filename)
    try:
        pv = Version(ver_s)
        ver_bonus = min(float(pv.major) * 10 + float(pv.minor), 500.0) * 0.01
    except (InvalidVersion, ValueError):
        ver_bonus = 0.0
    parts.append(f"version {ver_s!r} bonus {ver_bonus:.2f}")

    if "avx2" in filename.lower():
        parts.append("AVX2 hint +2")
        cuda_score += 2.0
    elif "basic" in filename.lower():
        parts.append("Basic +0")

    total = cuda_score + ver_bonus
    return total, "; ".join(parts)


def iter_release_wheel_assets(
    repo: str,
    timeout: float = 45.0,
    max_pages: int = 15,
) -> Iterator[dict[str, Any]]:
    """Yield GitHub release asset dicts with name, browser_download_url for .whl files."""
    headers = {
        "User-Agent": UA,
        "Accept": "application/vnd.github+json",
    }
    ght = (os.environ.get("GITHUB_TOKEN") or "").strip()
    if ght:
        headers["Authorization"] = f"Bearer {ght}"
    for page in range(1, max_pages + 1):
        url = f"{GITHUB_API}/repos/{repo}/releases?page={page}&per_page=100"
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            if e.code == 403:
                raise RuntimeError(
                    "GitHub API returned 403 (rate limit). Retry later or set "
                    "GITHUB_TOKEN for higher limits."
                ) from e
            if e.code == 404:
                raise ValueError(f"GitHub repo not found or not visible: {repo}") from e
            raise RuntimeError(f"GitHub API HTTP {e.code}: {e.reason}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"network error fetching releases: {e}") from e

        page_data = json.loads(body)
        if not isinstance(page_data, list) or not page_data:
            break
        for release in page_data:
            assets = release.get("assets") or []
            for a in assets:
                name = (a.get("name") or "").strip()
                if name.endswith(".whl"):
                    yield a
        if len(page_data) < 100:
            break


def resolve_best_wheel(
    profile: InstallProfile,
    repo: str = DEFAULT_REPO,
    timeout: float = 45.0,
) -> tuple[Optional[ResolvedWheel], list[ResolvedWheel]]:
    """
    Returns (best, top_alternatives_sorted[:5] excluding best).
    """
    ranked: list[ResolvedWheel] = []
    seen: set[str] = set()

    try:
        for asset in iter_release_wheel_assets(repo, timeout=timeout):
            fn = (asset.get("name") or "").strip()
            url = (asset.get("browser_download_url") or "").strip()
            if not fn or not url or fn in seen:
                continue
            seen.add(fn)
            score, reason = _score_wheel(fn, profile)
            if score < 0:
                continue
            ver_s = _parse_wheel_version_string(fn)
            ranked.append(
                ResolvedWheel(
                    filename=fn,
                    url=url,
                    score=score,
                    reason=reason,
                    package_version=ver_s,
                )
            )
    except RuntimeError:
        raise
    except ValueError:
        raise

    def _rank_sort_key(r: ResolvedWheel) -> tuple:
        try:
            v = Version(r.package_version)
            ver_key = (-v.major, -v.minor, -v.micro)
        except Exception:
            ver_key = (0, 0, 0)
        return (-r.score, ver_key, r.filename)

    ranked.sort(key=_rank_sort_key)

    if not ranked:
        return None, []

    best = ranked[0]
    alts = [r for r in ranked[1:6]]
    return best, alts


def format_resolution_report(
    python_exe: str,
    profile: InstallProfile,
    repo: str,
    best: Optional[ResolvedWheel],
    alternatives: list[ResolvedWheel],
) -> str:
    lines = [
        "=== llama-cpp-python - auto-resolved wheel (GitHub) ===",
        "",
        f"github_repo: {repo}",
        f"python_executable:\n  {python_exe}",
        f"match_profile: cp={profile.python_cp_tag!r} "
        f"platforms={sorted(profile.platform_strings)!r} "
        f"prefer_cuda={profile.prefer_cuda} cuda_hint={profile.cuda_hint_tag!r}",
        "",
    ]
    if best is None:
        lines.append("No compatible .whl found (check repo, network, or Python/OS version).")
        lines.append("")
        lines.append(
            "Tip: browse releases manually - "
            f"https://github.com/{repo}/releases"
        )
        return "\n".join(lines)

    lines.extend(
        [
            f"best_match: {best.filename}",
            f"score: {best.score:.2f}",
            f"detail: {best.reason}",
            f"download_url:\n  {best.url}",
            "",
            "-- pip (same environment as Comfy) --",
            f'  "{python_exe}" -m pip uninstall -y llama-cpp-python',
            f'  "{python_exe}" -m pip install "{best.url}"',
            "",
        ]
    )
    if alternatives:
        lines.append("alternatives (next best):")
        for a in alternatives:
            lines.append(f"  score={a.score:.2f}  {a.filename}")
        lines.append("")
    lines.append(
        "After install, verify Qwen3-VL support:\n"
        f'  "{python_exe}" -c "from llama_cpp.llama_chat_format import Qwen3VLChatHandler; print(Qwen3VLChatHandler)"'
    )
    return "\n".join(lines)
