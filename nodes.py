"""
In-ComfyUI multimodal chat: transformers (HF/local) or optional GGUF+mmproj via llama-cpp-python.
No external chat servers. GGUF: set main + mmproj for vision (use_vision+image) or for text-only (use_vision off / no image) when you want llama without loading combined_model.
"""

from __future__ import annotations

import hashlib
import os
import platform
import re
import subprocess
import sys
import warnings
from typing import Any, Callable, Optional, Tuple

import numpy as np
import torch

from . import env_probe, gguf_multimodal, gguf_picker, paths_helper, wheel_resolver

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        pipeline,
    )
except ImportError as e:  # pragma: no cover
    pipeline = None  # type: ignore
    _IMPORT_ERR = e
else:
    _IMPORT_ERR = None

# Used when generation_seed_mode == increment_each_run (global per process).
_GEN_SEED_INCREMENT: int = 0


def _resolve_sampling_seed(mode: str, base_seed: int) -> int:
    """Uint32 seed for HF generator + llama.cpp create_chat_completion."""
    global _GEN_SEED_INCREMENT
    base = int(base_seed) % (2**32)
    m = (mode or "fixed").strip().lower().replace("-", "_")
    if m in ("increment_each_run", "increment", "each_run"):
        out = (base + _GEN_SEED_INCREMENT) % (2**32)
        _GEN_SEED_INCREMENT = (_GEN_SEED_INCREMENT + 1) % (2**32)
        return out
    return base


def _merge_hf_generator_seed(
    gen_kw: dict[str, Any],
    *,
    do_sample: bool,
    device_mode: str,
    sampling_seed: int,
) -> dict[str, Any]:
    if not do_sample:
        return gen_kw
    out = dict(gen_kw)
    dev_name = (
        "cuda"
        if device_mode == "cuda" and torch.cuda.is_available()
        else "cpu"
    )
    g = torch.Generator(device=torch.device(dev_name))
    g.manual_seed(int(sampling_seed) & 0xFFFFFFFF)
    out["generator"] = g
    return out


def _tensor_to_pil_rgb(image: torch.Tensor) -> "Any":
    from PIL import Image

    if image.dim() == 4:
        image = image[0]
    arr = image.detach().cpu().float().numpy()
    mx = float(np.max(arr)) if arr.size else 0.0
    mn = float(np.min(arr)) if arr.size else 0.0
    # Comfy IMAGE is usually float 0–1; some nodes feed ~0–255 float
    if mx <= 1.0 + 1e-3 and mn >= -1e-3:
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    else:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _maybe_tensor_to_pil_list(
    use_vision: bool, *images: Optional[torch.Tensor]
) -> tuple[bool, Optional[Any]]:
    """Returns (image_linked, pil_or_list_or_none). Only decodes when use_vision=True."""
    linked = False
    if not use_vision:
        for im in images:
            if im is not None and isinstance(im, torch.Tensor) and im.numel() > 0:
                linked = True
                break
        return (linked, None)
    pils: list[Any] = []
    for im in images:
        if im is None or not isinstance(im, torch.Tensor) or im.numel() <= 0:
            continue
        linked = True
        pils.append(_tensor_to_pil_rgb(im))
    if not pils:
        return (linked, None)
    if len(pils) == 1:
        return (linked, pils[0])
    return (linked, pils)


def _read_text_file_maybe(path: str) -> str:
    p = (path or "").strip()
    if not p:
        return ""
    try:
        p2 = os.path.abspath(os.path.expanduser(os.path.expandvars(p)))
        if not os.path.isfile(p2):
            return ""
        with open(p2, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception:
        return ""


def _read_preset(rel_path: str, fallback: str) -> str:
    """Read a preset file relative to this module, fallback if missing."""
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        p = os.path.join(here, rel_path)
        if os.path.isfile(p):
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                s = f.read()
            if (s or "").strip():
                return s.strip()
    except Exception:
        pass
    return (fallback or "").strip()


def _preset_abs_path(rel_path: str) -> str:
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(here, rel_path)
    except Exception:
        return rel_path


def _list_preset_files(subdir: str) -> list[str]:
    """
    Returns relative preset file paths under `presets/<subdir>/` (sorted).
    Example entry: `presets/system/image_describer.md`
    """
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.join(here, "presets", subdir)
        if not os.path.isdir(root):
            return []
        out: list[str] = []
        for name in os.listdir(root):
            if not name.lower().endswith(".md"):
                continue
            if name.lower() == "custom.md":
                # Keep editable custom prompt file, but don't show it in preset pickers.
                continue
            out.append(os.path.join("presets", subdir, name))
        out.sort(key=lambda s: os.path.basename(s).lower())
        return out
    except Exception:
        return []


def _basename_no_ext(p: str) -> str:
    b = os.path.basename(p or "")
    if b.lower().endswith(".md"):
        return b[:-3]
    return b


def _format_block(title: str, text: str, *, fenced: bool = False) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    head = (title or "").strip()
    if not head:
        return t
    if fenced:
        return f"## {head}\n\n```text\n{t}\n```".strip()
    return f"## {head}\n\n{t}".strip()


def _qwen_gguf_text_dummy_pil() -> Any:
    """1×1 RGB placeholder so Qwen VL uses the same chat handler path as real vision (force_reasoning off)."""
    from PIL import Image

    return Image.new("RGB", (1, 1), (240, 240, 240))


def _looks_like_local_model_ref(s: str) -> bool:
    """True if this string should be loaded from disk (not the HF hub id)."""
    st = (s or "").strip()
    if not st:
        return False
    e = os.path.expandvars(os.path.expanduser(st))
    if os.path.isabs(e):
        return True
    if len(e) > 2 and e[1] == ":" and e[0].isalpha():  # Windows C:\...
        return True
    if e.startswith(("\\\\", "/", "./", "../", ".\\")):
        return True
    if e.lower().endswith(".gguf"):
        return True
    try:
        if os.path.isfile(e) or os.path.isdir(e):
            return True
    except OSError:
        pass
    for root in paths_helper.get_llm_root_dirs():
        if not root:
            continue
        cand = os.path.join(root, st)
        try:
            if os.path.isfile(cand) or os.path.isdir(cand):
                return True
        except OSError:
            continue
    return False


def _normalize_load_source(load_source: str, model_id: str) -> str:
    """
    auto → disk vs hub from the string shape.
    If the widget is stuck on huggingface but the value is clearly a path, use local anyway.
    """
    ls = (load_source or "auto").strip().lower()
    if ls not in ("auto", "huggingface", "local"):
        ls = "auto"
    if ls == "auto":
        return "local" if _looks_like_local_model_ref(model_id) else "huggingface"
    if ls == "huggingface" and _looks_like_local_model_ref(model_id):
        return "local"
    return ls


def _resolve_model_id(load_source: str, model_id: str) -> str:
    """HF hub id as-is; local = full path, or a name/path resolved under ComfyUI/models/llm/."""
    s = (model_id or "").strip()
    if not s:
        raise ValueError("Model id/path is empty.")
    ls = _normalize_load_source(load_source, s)
    if ls == "local":
        expanded = os.path.abspath(os.path.expanduser(os.path.expandvars(s)))
        if os.path.isfile(expanded) or os.path.isdir(expanded):
            return os.path.realpath(expanded)
        for root in paths_helper.get_llm_root_dirs():
            if not root:
                continue
            cand = os.path.join(root, s)
            if os.path.isfile(cand) or os.path.isdir(cand):
                return os.path.realpath(cand)
        return expanded
    return s


def _resolve_gguf_path_from_picker_or_string(picker: str, path_field: str) -> str:
    """
    If the models/llm COMBO is not the sentinel, resolve relative path under models/llm.
    Otherwise use the STRING path field (expandvars / strip only; same as before).
    """
    p = (picker or "").strip()
    if p and p != gguf_picker.SENTINEL:
        rel = p.replace("/", os.sep)
        for root in paths_helper.get_llm_root_dirs():
            if not root:
                continue
            base = os.path.abspath(os.path.normpath(root))
            cand = os.path.normpath(os.path.join(base, rel))
            if os.path.isfile(cand):
                return cand
        raise ValueError(
            f"GGUF picker file not found: {p!r} (place it under ComfyUI/models/llm or refresh the list)"
        )
    return (path_field or "").strip()


def _coerce_use_vision(raw: Any) -> bool:
    """
    Comfy BOOLEAN widgets normally pass bool; tolerate int/str from reroutes or adapters.
    """
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    s = str(raw).strip().lower()
    if s in ("false", "0", "no", "off", ""):
        return False
    if s in ("true", "1", "yes", "on"):
        return True
    return bool(s)


def _resolve_model_field(load_source: str, text_field: str, slot_label: str) -> str:
    s = (text_field or "").strip()
    if not s:
        raise ValueError(
            f"{slot_label}: empty. Set load_source to auto (recommended), huggingface (Hub id), or local (path); then fill this field."
        )
    return _resolve_model_id(load_source, s)


def _resolve_combined_if_filled(load_source: str, combined_model: str) -> str | None:
    """None when combined_model widget is empty (GGUF-only workflows)."""
    s = (combined_model or "").strip()
    if not s:
        return None
    return _resolve_model_id(load_source, s)


def _resolve_combined_required_for_hf_vlm(load_source: str, combined_model: str) -> str:
    mid = _resolve_combined_if_filled(load_source, combined_model)
    if mid is None:
        raise ValueError(
            "Vision without valid GGUF needs **combined_model**: set a Hugging Face id or local "
            "path to a VLM (transformers), or configure both GGUF main + mmproj."
        )
    return mid


def _hf_pretrained_extra_kw(mid: str) -> dict[str, Any]:
    """
    Local snapshot folders must use local_files_only=True or huggingface_hub may treat
    Windows paths as repo ids and raise HFValidationError.
    """
    exp = os.path.abspath(os.path.expanduser(os.path.expandvars(mid)))
    if os.path.isdir(exp):
        return {"local_files_only": True}
    if os.path.isfile(exp):
        return {"local_files_only": True}
    return {}


def _vision_lm_snapshot_unusable_as_causal_lm(resolved_mid: str) -> bool:
    """HF VLM repos/paths that typically fail AutoModelForCausalLM text-only fallback."""
    ln = resolved_mid.replace("\\", "/").lower()
    needles = (
        "smolvlm",
        "smol-vlm",
        "moondream",
        "llava",
        "internvl",
        "pixtral",
        "videollama",
        "gemma-3-it",
        "gemma3",
        "qgvl",
        "florence",
        "kosmos",
        "idefics",
        "nvila",
        "bunny",
        "minicpm-v",
        "minicpm_v",
    )
    return any(n in ln for n in needles)


def _dtype_from_choice(choice: str) -> torch.dtype:
    c = (choice or "auto").lower()
    if c == "bfloat16":
        return torch.bfloat16
    if c == "float16":
        return torch.float16
    if c == "float32":
        return torch.float32
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def _device_index(device_mode: str) -> int:
    if device_mode == "cuda" and torch.cuda.is_available():
        return 0
    return -1


def _cache_key(parts: Tuple[Any, ...]) -> str:
    h = hashlib.sha256(repr(parts).encode("utf-8")).hexdigest()[:24]
    return h


def _sanitize_sidebar_inputs(
    device: str,
    dtype: Any,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
) -> tuple[str, str, float, int, int]:
    """Tolerate mis-mapped widgets from old workflows (wrong socket order)."""
    d_raw = str(device or "").strip().lower()
    if d_raw == "auto" or d_raw not in ("cuda", "cpu"):
        if d_raw not in ("auto", "cuda", "cpu") and d_raw:
            warnings.warn(
                f"device={device!r} is not cuda/cpu/auto (likely mis-mapped); using auto.",
                UserWarning,
                stacklevel=2,
            )
        d = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        d = d_raw

    dt_raw = str(dtype or "").strip().lower()
    if dt_raw in ("false", "true", "none", ""):
        warnings.warn(
            f"dtype={dtype!r} coerced to auto (often a boolean on the wrong input).",
            UserWarning,
            stacklevel=2,
        )
        dt = "auto"
    elif dt_raw in ("auto", "bfloat16", "float16", "float32"):
        dt = dt_raw
    else:
        warnings.warn(f"dtype={dtype!r} unknown; using auto.", UserWarning, stacklevel=2)
        dt = "auto"

    tp = float(top_p)
    if tp > 1.0:
        warnings.warn(
            f"top_p={tp} > 1.0 (often top_k mis-mapped to top_p); clamping to 1.0.",
            UserWarning,
            stacklevel=2,
        )
        tp = 1.0

    tk = int(top_k)
    mnt = int(max_new_tokens)
    if mnt < 8:
        if mnt <= 0:
            warnings.warn(
                f"max_new_tokens={mnt} invalid or mis-mapped; using 256.",
                UserWarning,
                stacklevel=2,
            )
            mnt = 256
        else:
            warnings.warn(
                f"max_new_tokens={mnt} below 8; using 8.",
                UserWarning,
                stacklevel=2,
            )
            mnt = 8

    return d, dt, tp, tk, mnt


def _torch_cuda_one_liner() -> str:
    if not torch.cuda.is_available():
        return "torch_cuda_available=False"
    tag = env_probe.pip_cuda_tag_from_torch() or "?"
    nv = torch.version.cuda or "?"
    try:
        name = torch.cuda.get_device_name(0)
    except Exception:
        name = "?"
    mem = ""
    try:
        alloc = int(torch.cuda.memory_allocated(0))
        reserv = int(torch.cuda.memory_reserved(0))
        free_b, total_b = torch.cuda.mem_get_info(0)
        gb = 1024**3
        mem = (
            f" cuda_mem_alloc_gb={(alloc/gb):.2f}"
            f" cuda_mem_reserved_gb={(reserv/gb):.2f}"
            f" cuda_mem_free_gb={(free_b/gb):.2f}"
            f" cuda_mem_total_gb={(total_b/gb):.2f}"
        )
    except Exception:
        mem = ""
    return (
        f"torch_cuda_available=True torch.version.cuda={nv!r} "
        f"pip_wheel_cuda_tag_hint={tag!r} gpu0={name!r}{mem}"
    )


def build_llm_environment_report() -> str:
    """
    Human-readable lines for picking a llama-cpp-python wheel (CUDA tag vs CPU)
    and confirming Comfy's Python.
    """
    lines: list[str] = []
    lines.append("=== llm_comfy_multimodal — environment (copy when choosing a wheel) ===")
    lines.append("")
    lines.append(f"python_executable:\n  {sys.executable}")
    lines.append(f"python_version: {sys.version.split()[0]} ({sys.version})")
    lines.append(f"platform: {platform.platform()}")
    lines.append("")
    lines.append("-- PyTorch (this is what Comfy's graphs use) --")
    try:
        lines.append(f"torch.__version__: {torch.__version__}")
        lines.append(f"torch.version.cuda (built with): {torch.version.cuda!r}")
        lines.append(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            lines.append(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                try:
                    lines.append(
                        f"  GPU {i}: {torch.cuda.get_device_name(i)!r} "
                        f"capability={torch.cuda.get_device_capability(i)}"
                    )
                except Exception as e:
                    lines.append(f"  GPU {i}: (error reading: {e})")
            hint = env_probe.pip_cuda_tag_from_torch()
            if hint:
                lines.append("")
                lines.append(
                    f"pip_wheel_cuda_tag_hint: **{hint}**  "
                    f"(prefer llama-cpp-python assets whose filename contains '{hint}', "
                    "same Python tag e.g. cp312, same OS win/linux)"
                )
        else:
            lines.append("")
            lines.append(
                "pip_wheel_hint: no CUDA in this session — use **cpu** / **avx2** / "
                "**avx512** style wheels from the release page (match OS + Python version)."
            )
    except Exception as e:
        lines.append(f"(torch introspection failed: {e})")
    lines.append("")
    lines.append("-- nvidia-smi (driver / runtime; optional) --")
    try:
        r = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=6,
        )
        if r.returncode != 0:
            lines.append(f"nvidia-smi exit {r.returncode}: {r.stderr.strip() or r.stdout.strip()}")
        else:
            out = (r.stdout or "").strip().splitlines()
            lines.append("first lines:")
            for ln in out[:12]:
                lines.append(f"  {ln}")
            rq = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=6,
            )
            if rq.returncode == 0 and (rq.stdout or "").strip():
                lines.append("per-GPU (csv):")
                for ln in rq.stdout.strip().splitlines():
                    lines.append(f"  {ln}")
    except FileNotFoundError:
        lines.append("nvidia-smi: not found (AMD/Intel only, or NVIDIA driver not on PATH).")
    except Exception as e:
        lines.append(f"nvidia-smi: {e}")
    lines.append("")
    lines.append("-- llama-cpp-python (optional GGUF) --")
    try:
        import llama_cpp

        ver = getattr(llama_cpp, "__version__", "unknown")
        lines.append(f"import: OK  __version__: {ver}")
        try:
            from llama_cpp.llama_chat_format import Qwen3VLChatHandler  # noqa: F401

            lines.append("Qwen3VLChatHandler: available")
        except ImportError:
            lines.append("Qwen3VLChatHandler: NOT in this build (need Qwen3-capable wheel or fork)")
    except ImportError as e:
        lines.append(f"import: FAILED ({e})")
    lines.append("")
    lines.append("-- Install reminder --")
    lines.append(
        "Automated picker (scores GitHub wheels vs this Python/GPU):\n"
        f'  "{sys.executable}" -m llm_comfy_multimodal.install_llama_wheel\n'
        '  '
        "(add --install to run pip uninstall + pip install automatically)\n"
        "Or use node: **Multimodal LLM — llama-cpp wheel pick (GitHub)** (needs network)."
    )
    lines.append("")
    lines.append(
        f'  "{sys.executable}" -m pip uninstall -y llama-cpp-python\n'
        f'  "{sys.executable}" -m pip install <matching_wheel.whl>'
    )
    lines.append(
        "Manual browse: https://github.com/JamePeng/llama-cpp-python/releases"
    )
    return "\n".join(lines)


def _llm_diag_trace(
    lines: list[str], log_load_details: bool, message: str
) -> None:
    lines.append(message)
    if log_load_details:
        print(f"[llm_comfy_multimodal] {message}", flush=True)


def _llm_diag_on_failure(exc: BaseException, lines: list[str]) -> None:
    if not lines:
        return
    block = "\n".join(lines)
    if len(block) > 8000:
        block = (
            block[:3500]
            + "\n...[llm_comfy_multimodal trace truncated]...\n"
            + block[-3500:]
        )
    add = getattr(exc, "add_note", None)
    if callable(add):
        try:
            add("[llm_comfy_multimodal] execution trace:\n" + block)
        except Exception:
            pass
    print(
        f"[llm_comfy_multimodal] FAILED {type(exc).__name__}: {exc}\n--- trace ---\n{block}\n--- end trace ---",
        flush=True,
    )


# In-process caches (this ComfyUI server only).
_PIPE_CACHE: dict[str, Any] = {}
_CAUSAL_CACHE: dict[str, tuple[Any, Any]] = {}


def _image_text_pipe_cache_key(
    model_id: str,
    load_source: str,
    device_mode: str,
    dtype_choice: str,
    trust_remote_code: bool,
) -> str:
    mid = _resolve_model_id(load_source, model_id)
    dtype = _dtype_from_choice(dtype_choice)
    return _cache_key(
        ("image-text-to-text", mid, device_mode, str(dtype), trust_remote_code)
    )


def _causal_lm_cache_key(
    model_id: str,
    load_source: str,
    device_mode: str,
    dtype_choice: str,
    trust_remote_code: bool,
) -> str:
    mid = _resolve_model_id(load_source, model_id)
    dtype = _dtype_from_choice(dtype_choice)
    return _cache_key(("causal", mid, device_mode, str(dtype), trust_remote_code))


def _evict_pipeline_cache_key(key: str) -> None:
    pipe = _PIPE_CACHE.pop(key, None)
    if pipe is None:
        return
    try:
        if getattr(pipe, "model", None) is not None:
            del pipe.model
        if getattr(pipe, "image_processor", None) is not None:
            del pipe.image_processor
        if getattr(pipe, "tokenizer", None) is not None:
            del pipe.tokenizer
        if getattr(pipe, "processor", None) is not None:
            del pipe.processor
    except Exception:
        pass
    del pipe


def _evict_causal_cache_key(key: str) -> None:
    pair = _CAUSAL_CACHE.pop(key, None)
    if pair is None:
        return
    model, tok = pair
    try:
        del tok
        del model
    except Exception:
        pass


def clear_all_hosted_llm_caches() -> None:
    """Drop only this node's cached transformers pipelines / causal LMs / GGUF Llama (not Comfy checkpoints)."""
    for k in list(_PIPE_CACHE.keys()):
        _evict_pipeline_cache_key(k)
    for k in list(_CAUSAL_CACHE.keys()):
        _evict_causal_cache_key(k)
    gguf_multimodal.clear_llama_gguf_cache()


def _get_image_text_pipe(
    model_id: str,
    load_source: str,
    device_mode: str,
    dtype_choice: str,
    trust_remote_code: bool,
) -> Any:
    if _IMPORT_ERR is not None:
        raise RuntimeError(
            "transformers is not installed. pip install -r requirements.txt in this node folder."
        ) from _IMPORT_ERR
    mid = _resolve_model_id(load_source, model_id)
    dtype = _dtype_from_choice(dtype_choice)
    dev = _device_index(device_mode)
    key = _image_text_pipe_cache_key(
        model_id, load_source, device_mode, dtype_choice, trust_remote_code
    )
    if key not in _PIPE_CACHE:
        try:
            if dev >= 0:
                _PIPE_CACHE[key] = pipeline(
                    "image-text-to-text",
                    model=mid,
                    trust_remote_code=trust_remote_code,
                    device=dev,
                    dtype=dtype,
                )
            else:
                _PIPE_CACHE[key] = pipeline(
                    "image-text-to-text",
                    model=mid,
                    trust_remote_code=trust_remote_code,
                    device=-1,
                    dtype=torch.float32,
                )
        except TypeError:
            if dev >= 0:
                _PIPE_CACHE[key] = pipeline(
                    "image-text-to-text",
                    model=mid,
                    trust_remote_code=trust_remote_code,
                    device=dev,
                    torch_dtype=dtype,
                )
            else:
                _PIPE_CACHE[key] = pipeline(
                    "image-text-to-text",
                    model=mid,
                    trust_remote_code=trust_remote_code,
                    device=-1,
                    torch_dtype=torch.float32,
                )
    return _PIPE_CACHE[key]


def _get_causal_lm(
    model_id: str,
    load_source: str,
    device_mode: str,
    dtype_choice: str,
    trust_remote_code: bool,
) -> tuple[Any, Any]:
    if _IMPORT_ERR is not None:
        raise RuntimeError(
            "transformers is not installed. pip install -r requirements.txt in this node folder."
        ) from _IMPORT_ERR
    mid = _resolve_model_id(load_source, model_id)
    dtype = _dtype_from_choice(dtype_choice)
    extra = _hf_pretrained_extra_kw(mid)
    key = _causal_lm_cache_key(
        model_id, load_source, device_mode, dtype_choice, trust_remote_code
    )
    if key not in _CAUSAL_CACHE:
        tok = AutoTokenizer.from_pretrained(
            mid, trust_remote_code=trust_remote_code, **extra
        )
        load_dtype = dtype if torch.cuda.is_available() and device_mode == "cuda" else torch.float32
        try:
            m = AutoModelForCausalLM.from_pretrained(
                mid,
                trust_remote_code=trust_remote_code,
                dtype=load_dtype,
                **extra,
            )
        except TypeError:
            m = AutoModelForCausalLM.from_pretrained(
                mid,
                trust_remote_code=trust_remote_code,
                torch_dtype=load_dtype,
                **extra,
            )
        if torch.cuda.is_available() and device_mode == "cuda":
            m = m.to("cuda")
        else:
            m = m.to("cpu")
        _CAUSAL_CACHE[key] = (m, tok)
    return _CAUSAL_CACHE[key]


def _build_hf_generate_kwargs(
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    use_repetition_penalty: bool = True,
) -> dict[str, Any]:
    g: dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": bool(do_sample),
    }
    if do_sample:
        g["temperature"] = float(temperature)
        g["top_p"] = float(top_p)
        if top_k > 0:
            g["top_k"] = int(top_k)
    if (
        use_repetition_penalty
        and abs(float(repetition_penalty) - 1.0) > 1e-6
    ):
        g["repetition_penalty"] = float(repetition_penalty)
    return g


def _compose_user_message(
    prompt: str,
    extra_context: str,
    image_caption: Optional[str] = None,
) -> str:
    parts: list[str] = []
    ec = (extra_context or "").strip()
    up = (prompt or "").strip()
    if ec:
        parts.append(ec)
    if image_caption:
        parts.append(f"[Image description]\n{image_caption.strip()}")
    if up:
        parts.append(up)
    if not parts:
        return ""
    return "\n\n".join(parts)


def _send_stream_to_client(cumulative: str) -> None:
    try:
        from server import PromptServer

        PromptServer.instance.send_sync(
            "llm_comfy_multimodal.stream",
            {"text": cumulative},
        )
    except Exception:
        pass


def _postprocess_thinking_blocks(
    text: str,
    include_thinking_in_output: bool,
    thinking_open_tag: str,
    thinking_close_tag: str,
) -> str:
    """Strip one or more thinking blocks when include_thinking_in_output is False."""
    if include_thinking_in_output or not (text or "").strip():
        return text
    o = (thinking_open_tag or "").strip() or "<think>"
    c = (thinking_close_tag or "").strip() or "</think>"
    pat = re.escape(o) + r"[\s\S]*?" + re.escape(c)
    out = re.sub(pat, "", text).strip()
    # Qwen3 fence `` often still appears when llama-cpp drops enable_thinking=False.
    bt = "\x60"
    out = re.sub(
        bt + r"think" + bt + r"[\s\S]*?" + bt + r"/" + bt,
        "",
        out,
        flags=re.DOTALL,
    ).strip()
    out = re.sub(
        r"<think>[\s\S]*?</think>",
        "",
        out,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()
    return out


def _call_gguf_vlm(
    *,
    gm: str,
    gmm: str,
    pil: Optional[Any],
    system_prompt: str,
    extra_context: str,
    user_prompt: str,
    gguf_vlm_handler: str,
    gguf_n_ctx: int,
    n_gl: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    use_repetition_penalty: bool,
    stream_gguf: bool,
    gguf_chat_template_enable_thinking: bool,
    gguf_allow_qwen25_if_qwen3_handler_missing: bool,
    sampling_seed: int,
) -> tuple[str, str]:
    on_chunk: Optional[Callable[[str], None]] = (
        _send_stream_to_client if stream_gguf else None
    )
    return gguf_multimodal.run_gguf_vlm_chat(
        main_gguf=gm,
        mmproj_gguf=gmm,
        pil_image=pil,
        system_prompt=system_prompt,
        extra_context=extra_context,
        user_prompt=user_prompt,
        handler_name=gguf_vlm_handler,
        n_ctx=int(gguf_n_ctx),
        n_gpu_layers=n_gl,
        max_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repeat_penalty=repetition_penalty,
        use_repeat_penalty=use_repetition_penalty,
        gguf_chat_template_enable_thinking=gguf_chat_template_enable_thinking,
        allow_qwen25_when_qwen3_missing=gguf_allow_qwen25_if_qwen3_handler_missing,
        stream=stream_gguf,
        on_chunk=on_chunk,
        seed=int(sampling_seed) & 0xFFFFFFFF,
    )


def _has_any_input(
    system_prompt: str, extra_context: str, prompt: str
) -> bool:
    return bool(
        (system_prompt or "").strip()
        or (extra_context or "").strip()
        or (prompt or "").strip()
    )


def _run_causal_chat(
    model: Any,
    tokenizer: Any,
    *,
    prompt: str,
    extra_context: str,
    system_prompt: str,
    image_caption: Optional[str],
    max_new_tokens: int,
    device_mode: str,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    use_repetition_penalty: bool = True,
    sampling_seed: int = 0,
) -> str:
    if not _has_any_input(system_prompt, extra_context, prompt) and not image_caption:
        raise ValueError(
            "Provide at least one of: prompt, extra context, system prompt, or an image description."
        )
    user_msg = _compose_user_message(prompt, extra_context, image_caption)
    if not user_msg.strip() and (system_prompt or "").strip():
        # Some chat templates require a non-empty user turn; keep this free of instructions.
        user_msg = _MIN_CHAT_USER_PLACEHOLDER

    device = "cuda" if device_mode == "cuda" and torch.cuda.is_available() else "cpu"
    messages: list[dict[str, str]] = []
    sys_s = (system_prompt or "").strip()
    if sys_s:
        messages.append({"role": "system", "content": sys_s})
    messages.append({"role": "user", "content": user_msg})

    tmpl = getattr(tokenizer, "chat_template", None)
    if hasattr(tokenizer, "apply_chat_template") and tmpl is not None:
        prompt_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
    else:
        flat = "\n\n".join(m["content"] for m in messages)
        prompt_ids = tokenizer(flat, return_tensors="pt").input_ids
    prompt_ids = prompt_ids.to(device)
    if next(model.parameters()).device.type != device:
        model = model.to(device)

    gen = _build_hf_generate_kwargs(
        max_new_tokens,
        do_sample,
        temperature,
        top_p,
        top_k,
        repetition_penalty,
        use_repetition_penalty,
    )
    gen = _merge_hf_generator_seed(
        gen,
        do_sample=do_sample,
        device_mode=device_mode,
        sampling_seed=sampling_seed,
    )
    pad = getattr(tokenizer, "pad_token_id", None) or getattr(
        tokenizer, "eos_token_id", None
    )
    with torch.inference_mode():
        out = model.generate(
            prompt_ids,
            pad_token_id=pad,
            **gen,
        )
    new_tokens = out[0, prompt_ids.shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# HF image-text pipeline often rejects truly empty `text=`; keep this neutral (not an instruction).
_MIN_CHAT_USER_PLACEHOLDER = "."


def _compose_vlm_user_text(
    system_prompt: str, extra_context: str, prompt: str
) -> str:
    blocks: list[str] = []
    sp = (system_prompt or "").strip()
    if sp:
        blocks.append(f"[System]\n{sp}")
    ec = (extra_context or "").strip()
    if ec:
        blocks.append(f"[Context]\n{ec}")
    up = (prompt or "").strip()
    if up:
        blocks.append(up)
    if not blocks:
        return ""
    return "\n\n".join(blocks)


class MultimodalLLMNode:
    """Multimodal LLM: transformers and/or GGUF+mmproj. See DESCRIPTION."""

    DESCRIPTION = (
        "**GGUF:** set **GGUF main + mmproj** (.gguf files under models/llm). "
        "**Use vision** ON + IMAGE → vision run; **OFF** → text-only (linked image is ignored — Comfy cannot hide the socket, but the model never sees pixels). "
        "**Image-only:** vision ON + IMAGE + empty text fields — GGUF sends image only; HF VLM may use a single `.` token only so the HF API accepts the call (not a written instruction). "
        "**combined_model** = transformers only (HF id / snapshot folder). Leave **empty** for GGUF-only (including text-only on GGUF). "
        "**load_source** applies only to **combined_model**."
    )

    CATEGORY = "llm/multimodal"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        _gguf_combo = gguf_picker.combo_choices_llm_gguf()
        _gguf_def_main, _gguf_def_mm = gguf_picker.suggest_default_main_and_mmproj()
        return {
            "required": {
                "use_vision": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label": "Use vision (see image) — OFF ignores image even if linked",
                    },
                ),
                "load_source": (
                    ["auto", "local", "huggingface"],
                    {"default": "auto"},
                ),
                "gguf_main_models_llm": (
                    _gguf_combo,
                    {
                        "default": _gguf_def_main,
                        "label": "GGUF main (.gguf) — primary weights",
                    },
                ),
                "gguf_mmproj_models_llm": (
                    _gguf_combo,
                    {
                        "default": _gguf_def_mm,
                        "label": "GGUF mmproj (.gguf) — vision projector",
                    },
                ),
                "gguf_model_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "Main .gguf if not using picker above (full path or empty)",
                    },
                ),
                "gguf_mmproj_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "mmproj .gguf if not using picker (full path or empty)",
                    },
                ),
                "gguf_vlm_handler": (
                    ["qwen3-vl", "qwen2.5-vl", "gemma3", "llava-1.5", "llava-1.6"],
                    {"default": "qwen3-vl"},
                ),
                "gguf_allow_qwen25_if_qwen3_handler_missing": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label": "GGUF: if Qwen3VLChatHandler missing, use Qwen2.5 handler (degraded)",
                    },
                ),
                "gguf_n_ctx": (
                    "INT",
                    {"default": 12288, "min": 512, "max": 131072, "step": 256},
                ),
                "gguf_streaming": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label": "stream GGUF decode (progress via llm_comfy_multimodal.stream)",
                    },
                ),
                "gguf_chat_template_enable_thinking": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label": "GGUF: enable_thinking (ON = reasoning/thinking; Jinja + Qwen3 force_reasoning)",
                    },
                ),
                "include_thinking_in_output": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label": "keep thinking blocks in reply (off = strip for faster-looking text)",
                    },
                ),
                "thinking_open_tag": (
                    "STRING",
                    {
                        "default": "<think>",
                        "multiline": False,
                        "placeholder": "start tag to strip when thinking hidden",
                    },
                ),
                "thinking_close_tag": (
                    "STRING",
                    {
                        "default": "</think>",
                        "multiline": False,
                        "placeholder": "end tag to strip",
                    },
                ),
                "combined_model": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "Optional if GGUF pair set — HF/local transformers only (not GGUF filenames)",
                    },
                ),
                "system_prompt": ("STRING", {"default": "", "multiline": True}),
                "extra_context": ("STRING", {"default": "", "multiline": True}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "max_new_tokens": (
                    "INT",
                    {"default": 5000, "min": 0, "max": 65536, "step": 1},
                ),
                "do_sample": ("BOOLEAN", {"default": False}),
                "generation_seed_mode": (
                    ["fixed", "increment_each_run"],
                    {
                        "default": "fixed",
                        "label": "sampling seed: fixed vs +1 each execution (global counter)",
                    },
                ),
                "generation_seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 4294967295, "step": 1},
                ),
                "temperature": (
                    "FLOAT",
                    {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05},
                ),
                "top_p": (
                    "FLOAT",
                    {"default": 0.95, "min": 0.0, "max": 128.0, "step": 0.01},
                ),
                "top_k": (
                    "INT",
                    {"default": 20, "min": -1, "max": 100, "step": 1},
                ),
                "repetition_penalty": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05},
                ),
                "use_repetition_penalty": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label": "apply repetition penalty (HF + GGUF)",
                    },
                ),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "dtype": (
                    [
                        "auto",
                        "bfloat16",
                        "float16",
                        "float32",
                        "False",
                        "True",
                    ],
                    {"default": "auto"},
                ),
                "trust_remote_code": ("BOOLEAN", {"default": False}),
                "keep_models_loaded": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label": "keep this node's HF weights cached between runs",
                    },
                ),
                "log_load_details": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label": "log which model/branch is used (Comfy console)",
                    },
                ),
            },
            "optional": {
                "image": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
            },
        }

    def run(
        self,
        use_vision: bool,
        load_source: str,
        combined_model: str,
        system_prompt: str,
        extra_context: str,
        prompt: str,
        max_new_tokens: int,
        do_sample: bool,
        generation_seed_mode: str,
        generation_seed: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        use_repetition_penalty: bool,
        device: str,
        dtype: str,
        trust_remote_code: bool,
        keep_models_loaded: bool,
        log_load_details: bool = True,
        gguf_model_path: str = "",
        gguf_mmproj_path: str = "",
        gguf_main_models_llm: str = gguf_picker.SENTINEL,
        gguf_mmproj_models_llm: str = gguf_picker.SENTINEL,
        gguf_vlm_handler: str = "qwen3-vl",
        gguf_n_ctx: int = 12288,
        gguf_streaming: bool = False,
        gguf_chat_template_enable_thinking: bool = False,
        gguf_allow_qwen25_if_qwen3_handler_missing: bool = True,
        include_thinking_in_output: bool = False,
        thinking_open_tag: str = "<think>",
        thinking_close_tag: str = "</think>",
        image: Optional[torch.Tensor] = None,
        image2: Optional[torch.Tensor] = None,
        image3: Optional[torch.Tensor] = None,
    ) -> tuple[str]:
        pipe_keys: set[str] = set()
        causal_keys: set[str] = set()
        llama_keys: set[str] = set()
        out_text: str | None = None
        device, dtype, top_p, top_k, max_new_tokens = _sanitize_sidebar_inputs(
            device, dtype, top_p, top_k, max_new_tokens
        )
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        n_gl = -1 if device == "cuda" and torch.cuda.is_available() else 0

        eff_seed = _resolve_sampling_seed(generation_seed_mode, generation_seed)
        gen_kw = _build_hf_generate_kwargs(
            max_new_tokens,
            do_sample,
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            use_repetition_penalty,
        )
        gen_kw = _merge_hf_generator_seed(
            gen_kw,
            do_sample=do_sample,
            device_mode=device,
            sampling_seed=eff_seed,
        )

        trace_lines: list[str] = []
        try:
            use_vision = _coerce_use_vision(use_vision)
            # Only decode pixels when vision is enabled — OFF + linked image(s) stays text-only.
            image_linked, pil = _maybe_tensor_to_pil_list(use_vision, image, image2, image3)

            user_prompt = (prompt or "").strip()
            g_raw_m = _resolve_gguf_path_from_picker_or_string(
                gguf_main_models_llm, gguf_model_path
            )
            g_raw_p = _resolve_gguf_path_from_picker_or_string(
                gguf_mmproj_models_llm, gguf_mmproj_path
            )
            gv = gguf_multimodal.gguf_paths_valid(g_raw_m, g_raw_p)
            _llm_diag_trace(
                trace_lines,
                log_load_details,
                f"start use_vision={use_vision} load_source={load_source!r} "
                f"device={device!r} image_linked={'yes' if image_linked else 'no'} "
                f"pil_for_vision={'yes' if pil is not None else 'no'} "
                f"gguf_paths_valid={gv} | eff_sampling_seed={eff_seed} "
                f"generation_seed_mode={generation_seed_mode!r} | {_torch_cuda_one_liner()}",
            )
            if log_load_details and image_linked and not use_vision:
                _llm_diag_trace(
                    trace_lines,
                    log_load_details,
                    "Image socket has data but use_vision OFF — not decoded; text-only branch.",
                )

            allow_image_only = bool(use_vision and image_linked)
            if not _has_any_input(system_prompt, extra_context, prompt) and not allow_image_only:
                raise ValueError(
                    "Set at least one of: prompt, extra context, or system prompt "
                    "(or link an image with use_vision on)."
                )

            want_vlm = pil is not None  # PIL only built when use_vision and image_linked
            if (
                log_load_details
                and use_vision
                and pil is None
                and gguf_multimodal.gguf_paths_valid(g_raw_m, g_raw_p)
            ):
                _llm_diag_trace(
                    trace_lines,
                    log_load_details,
                    "use_vision ON but no image — skipping GGUF/HF VLM; using text branch "
                    "(GGUF paths may stay configured for the next image run).",
                )
            if (
                log_load_details
                and gguf_multimodal.gguf_paths_valid(g_raw_m, g_raw_p)
                and not want_vlm
            ):
                    _llm_diag_trace(
                        trace_lines,
                        log_load_details,
                        "Vision off / no image — GGUF **text-only** branch (plain main.gguf "
                        "when possible; not multimodal decode).",
                    )
            if want_vlm:
                if (g_raw_m.strip() or g_raw_p.strip()) and not gguf_multimodal.gguf_paths_valid(
                    g_raw_m, g_raw_p
                ):
                    raise ValueError(
                        "For GGUF vision, set both GGUF pickers or both STRING paths to "
                        "existing files (main .gguf + mmproj .gguf), "
                        "or leave both unset and configure **combined_model** for transformers VLM."
                    )
                if gguf_multimodal.gguf_paths_valid(g_raw_m, g_raw_p):
                    gm = os.path.abspath(
                        os.path.expanduser(os.path.expandvars(g_raw_m.strip()))
                    )
                    gmm = os.path.abspath(
                        os.path.expanduser(os.path.expandvars(g_raw_p.strip()))
                    )
                    try:
                        eff = gguf_multimodal.gguf_resolve_cache_tag(
                            gguf_vlm_handler,
                            gguf_allow_qwen25_if_qwen3_handler_missing,
                        )
                    except Exception as tag_e:
                        eff = f"(could not resolve handler tag: {tag_e})"
                    _llm_diag_trace(
                        trace_lines,
                        log_load_details,
                        "branch=combined+GGUF+VLM "
                        f"handler={gguf_vlm_handler!r} gguf_cache_tag={eff!r} "
                        f"n_ctx={gguf_n_ctx} n_gpu_layers={n_gl}",
                    )
                    _llm_diag_trace(
                        trace_lines,
                        log_load_details,
                        f"GGUF main_gguf={gm}",
                    )
                    _llm_diag_trace(
                        trace_lines,
                        log_load_details,
                        f"GGUF mmproj_gguf={gmm}",
                    )
                    out_text, lk = _call_gguf_vlm(
                        gm=gm,
                        gmm=gmm,
                        pil=pil,
                        system_prompt=system_prompt,
                        extra_context=extra_context,
                        user_prompt=user_prompt,
                        gguf_vlm_handler=gguf_vlm_handler,
                        gguf_n_ctx=int(gguf_n_ctx),
                        n_gl=n_gl,
                        max_new_tokens=max_new_tokens,
                        do_sample=do_sample,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        repetition_penalty=repetition_penalty,
                        use_repetition_penalty=use_repetition_penalty,
                        stream_gguf=gguf_streaming,
                        gguf_chat_template_enable_thinking=gguf_chat_template_enable_thinking,
                        gguf_allow_qwen25_if_qwen3_handler_missing=gguf_allow_qwen25_if_qwen3_handler_missing,
                        sampling_seed=eff_seed,
                    )
                    llama_keys.add(lk)
                else:
                    mid = _resolve_combined_required_for_hf_vlm(
                        load_source, combined_model
                    )
                    _llm_diag_trace(
                        trace_lines,
                        log_load_details,
                        "branch=combined+HF+VLM_pipeline "
                        f"combined_model_field={combined_model!r} -> resolved_id={mid!r}",
                    )
                    vlm_prompt = _compose_vlm_user_text(
                        system_prompt, extra_context, user_prompt
                    )
                    if not (vlm_prompt or "").strip():
                        vlm_prompt = _MIN_CHAT_USER_PLACEHOLDER
                        _llm_diag_trace(
                            trace_lines,
                            log_load_details,
                            "HF+VLM: empty text fields — using minimal placeholder for pipeline API only.",
                        )
                    pipe_keys.add(
                        _image_text_pipe_cache_key(
                            mid, load_source, device, dtype, trust_remote_code
                        )
                    )
                    pipe = _get_image_text_pipe(
                        mid, load_source, device, dtype, trust_remote_code
                    )
                    gen = pipe(
                        images=pil,
                        text=vlm_prompt,
                        generate_kwargs=dict(gen_kw),
                        return_full_text=False,
                    )
                    text = (
                        gen[0]["generated_text"]
                        if isinstance(gen, list)
                        else str(gen)
                    )
                    out_text = text.strip()
            else:
                # Text-only: GGUF pair → llama chat without image (ignore combined_model).
                if gv:
                    gm = os.path.abspath(
                        os.path.expanduser(os.path.expandvars(g_raw_m.strip()))
                    )
                    gmm = os.path.abspath(
                        os.path.expanduser(os.path.expandvars(g_raw_p.strip()))
                    )
                    try:
                        eff = gguf_multimodal.gguf_resolve_cache_tag(
                            gguf_vlm_handler,
                            gguf_allow_qwen25_if_qwen3_handler_missing,
                        )
                    except Exception as tag_e:
                        eff = f"(could not resolve handler tag: {tag_e})"
                    stem = os.path.splitext(os.path.basename(gm))[0]
                    hn = (gguf_vlm_handler or "").lower()
                    if "qwen" in hn:
                        # Qwen text-only: use the same VL stack (handler + mmproj) but do NOT attach
                        # an image. Attaching a dummy image makes short prompts trigger image talk.
                        _llm_diag_trace(
                            trace_lines,
                            log_load_details,
                            "branch=combined+GGUF+text_qwen_vl_text_only — VL handler, no image. "
                            f"handler={gguf_vlm_handler!r} gguf_cache_tag={eff!r} "
                            f"display_stem={stem!r}",
                        )
                        out_text, lk = _call_gguf_vlm(
                            gm=gm,
                            gmm=gmm,
                            pil=None,
                            system_prompt=system_prompt,
                            extra_context=extra_context,
                            user_prompt=user_prompt,
                            gguf_vlm_handler=gguf_vlm_handler,
                            gguf_n_ctx=int(gguf_n_ctx),
                            n_gl=n_gl,
                            max_new_tokens=max_new_tokens,
                            do_sample=do_sample,
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                            repetition_penalty=repetition_penalty,
                            use_repetition_penalty=use_repetition_penalty,
                            stream_gguf=gguf_streaming,
                            gguf_chat_template_enable_thinking=gguf_chat_template_enable_thinking,
                            gguf_allow_qwen25_if_qwen3_handler_missing=gguf_allow_qwen25_if_qwen3_handler_missing,
                            sampling_seed=eff_seed,
                        )
                    else:
                        try:
                            out_text, lk = gguf_multimodal.run_gguf_plain_text_chat(
                                main_gguf=gm,
                                system_prompt=system_prompt,
                                extra_context=extra_context,
                                user_prompt=user_prompt,
                                handler_name=gguf_vlm_handler,
                                n_ctx=int(gguf_n_ctx),
                                n_gpu_layers=n_gl,
                                max_tokens=max_new_tokens,
                                do_sample=do_sample,
                                temperature=temperature,
                                top_p=top_p,
                                top_k=top_k,
                                repeat_penalty=repetition_penalty,
                                use_repeat_penalty=use_repetition_penalty,
                                gguf_chat_template_enable_thinking=gguf_chat_template_enable_thinking,
                                stream=gguf_streaming,
                                on_chunk=(
                                    _send_stream_to_client
                                    if gguf_streaming
                                    else None
                                ),
                                seed=int(eff_seed) & 0xFFFFFFFF,
                            )
                            _llm_diag_trace(
                                trace_lines,
                                log_load_details,
                                "branch=combined+GGUF+text_plain "
                                f"handler={gguf_vlm_handler!r} gguf_cache_tag={eff!r} "
                                f"display_stem={stem!r}",
                            )
                        except Exception as plain_e:
                            _llm_diag_trace(
                                trace_lines,
                                log_load_details,
                                f"GGUF plain text path failed ({type(plain_e).__name__}: {plain_e}); "
                                "falling back to main+mmproj VL stack (text-only, no image).",
                            )
                            _llm_diag_trace(
                                trace_lines,
                                log_load_details,
                                "branch=combined+GGUF+text_only_vlm_fallback "
                                f"handler={gguf_vlm_handler!r} gguf_cache_tag={eff!r} "
                                f"display_stem={stem!r}",
                            )
                            out_text, lk = _call_gguf_vlm(
                                gm=gm,
                                gmm=gmm,
                                pil=None,
                                system_prompt=system_prompt,
                                extra_context=extra_context,
                                user_prompt=user_prompt,
                                gguf_vlm_handler=gguf_vlm_handler,
                                gguf_n_ctx=int(gguf_n_ctx),
                                n_gl=n_gl,
                                max_new_tokens=max_new_tokens,
                                do_sample=do_sample,
                                temperature=temperature,
                                top_p=top_p,
                                top_k=top_k,
                                repetition_penalty=repetition_penalty,
                                use_repetition_penalty=use_repetition_penalty,
                                stream_gguf=gguf_streaming,
                                gguf_chat_template_enable_thinking=gguf_chat_template_enable_thinking,
                                gguf_allow_qwen25_if_qwen3_handler_missing=gguf_allow_qwen25_if_qwen3_handler_missing,
                                sampling_seed=eff_seed,
                            )
                    _llm_diag_trace(trace_lines, log_load_details, f"GGUF main_gguf={gm}")
                    _llm_diag_trace(trace_lines, log_load_details, f"GGUF mmproj_gguf={gmm}")
                    llama_keys.add(lk)
                else:
                    mid_try = _resolve_combined_if_filled(load_source, combined_model)
                    if mid_try is None:
                        raise ValueError(
                            "Text-only needs **combined_model** (HF causal id or transformers folder under "
                            "`models/llm`), or set **both GGUF main + mmproj** paths so llama.cpp can chat "
                            "without loading Hugging Face weights."
                        )
                    if _vision_lm_snapshot_unusable_as_causal_lm(mid_try):
                        raise ValueError(
                            f"combined_model path looks like a vision-only snapshot, not a causal LM: {mid_try!r}. "
                            "Use an instruct/causal checkpoint for text-only, or GGUF main+mmproj."
                        ) from None
                    try:
                        model, tokenizer = _get_causal_lm(
                            mid_try, load_source, device, dtype, trust_remote_code
                        )
                    except Exception as e:
                        raise ValueError(
                            f"Could not load combined_model as AutoModelForCausalLM: {mid_try!r}. "
                            "Use a transformers instruct checkpoint or clear this field and choose "
                            "**GGUF main + mmproj** for GGUF-only (folder names under models/llm are not "
                            "Safetensors/PyTorch checkpoints). "
                            "Check load_source / trust_remote_code."
                        ) from e
                    picked_mid = mid_try
                    causal_keys.add(
                        _causal_lm_cache_key(
                            picked_mid, load_source, device, dtype, trust_remote_code
                        )
                    )
                    _llm_diag_trace(
                        trace_lines,
                        log_load_details,
                        "branch=combined+causal_LM "
                        f"resolved_id={picked_mid!r}",
                    )
                    out_text = _run_causal_chat(
                        model,
                        tokenizer,
                        prompt=user_prompt,
                        extra_context=extra_context,
                        system_prompt=system_prompt,
                        image_caption=None,
                        max_new_tokens=max_new_tokens,
                        device_mode=device,
                        do_sample=do_sample,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        repetition_penalty=repetition_penalty,
                        use_repetition_penalty=use_repetition_penalty,
                        sampling_seed=eff_seed,
                    )
        except Exception as e:
            _llm_diag_on_failure(e, trace_lines)
            raise
        finally:
            if not keep_models_loaded:
                for k in pipe_keys:
                    _evict_pipeline_cache_key(k)
                for k in causal_keys:
                    _evict_causal_cache_key(k)
                for k in llama_keys:
                    gguf_multimodal.evict_llama_cache_key(k)

        assert out_text is not None
        out_text = _postprocess_thinking_blocks(
            out_text,
            include_thinking_in_output,
            thinking_open_tag,
            thinking_close_tag,
        )
        return (out_text,)


class MultimodalLLMWheelRecommendation:
    """Fetches JamePeng-style release wheels via GitHub API and picks the best match for this Python/GPU."""

    CATEGORY = "llm/multimodal"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "run": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label": "resolve wheel (needs network)",
                    },
                ),
                "github_repo": (
                    "STRING",
                    {
                        "default": wheel_resolver.DEFAULT_REPO,
                        "multiline": False,
                        "placeholder": "owner/repo",
                    },
                ),
            }
        }

    DESCRIPTION = (
        "Calls GitHub Releases for llama-cpp-python builds, scores **.whl** files against "
        "this machine (Python cp tag, OS platform, PyTorch CUDA tag like cu124 vs filenames). "
        "Outputs ready-to-copy **pip install URL** commands. Requires internet; optional "
        "**GITHUB_TOKEN** env avoids API rate limits. CLI: "
        "**python -m llm_comfy_multimodal.install_llama_wheel**."
    )

    def run(self, run: bool, github_repo: str) -> tuple[str]:
        if not run:
            return ("(skipped)",)
        repo = (github_repo or "").strip() or wheel_resolver.DEFAULT_REPO
        profile = env_probe.build_install_profile()
        try:
            best, alts = wheel_resolver.resolve_best_wheel(profile, repo=repo)
        except Exception as e:
            return (
                "=== llama-cpp-python wheel resolver — error ===\n\n"
                f"{type(e).__name__}: {e}\n\n"
                "Try again later, set GITHUB_TOKEN, or install manually from the repo releases page.",
            )
        text = wheel_resolver.format_resolution_report(
            sys.executable, profile, repo, best, alts
        )
        return (text,)


class MultimodalLLMEnvironmentReport:
    """Prints Python path, PyTorch/CUDA, nvidia-smi snippet, and llama-cpp-python hints for wheel picking."""

    CATEGORY = "llm/multimodal"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "run": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label": "generate report (wire to Show Text or read console)",
                    },
                ),
            }
        }

    DESCRIPTION = (
        "Copy the **report** output when choosing a llama-cpp-python wheel: it shows "
        "**python.exe**, **torch.version.cuda** → **cu124**-style hint, **GPU name**, "
        "and whether **Qwen3VLChatHandler** exists."
    )

    def run(self, run: bool) -> tuple[str]:
        if not run:
            return ("(skipped)",)
        return (build_llm_environment_report(),)


class UnloadHostedMultimodalLLMCache:
    """Drops only this extension's HF/transformers caches; does not touch ComfyUI model loading."""

    CATEGORY = "llm/multimodal"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "run": ("BOOLEAN", {"default": True}),
            }
        }

    def run(self, run: bool) -> tuple[str]:
        if run:
            clear_all_hosted_llm_caches()
            return ("multimodal extension HF cache cleared (Comfy models unchanged)",)
        return ("skipped",)


class ContextSchemaBuilder:
    """
    Builds a "programmable" output contract for the LLM.

    Intended wiring:
      base_system_prompt + contract → MultimodalLLMNode.system_prompt
      base_extra_context → MultimodalLLMNode.extra_context
      contract_prompt(user prompt) → MultimodalLLMNode.prompt

    Then parse the model output with ContextSchemaParser to get updated context.
    """

    CATEGORY = "llm/multimodal"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("system_prompt_addition", "prompt_wrapped")

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "schema_json_template": (
                    "STRING",
                    {
                        "default": '{\n  "answer": "",\n  "should_update_context": false,\n  "context": ""\n}\n',
                        "multiline": True,
                        "label": "JSON output template (edit this)",
                    },
                ),
                "context_open_tag": (
                    "STRING",
                    {"default": "<context>", "multiline": False},
                ),
                "context_close_tag": (
                    "STRING",
                    {"default": "</context>", "multiline": False},
                ),
                "base_instructions": (
                    "STRING",
                    {
                        "default": "Return ONLY JSON matching the template. If context needs updating, write the FULL updated context inside <context>...</context> (and also fill the context field).",
                        "multiline": True,
                    },
                ),
                "user_prompt": ("STRING", {"default": "", "multiline": True}),
            }
        }

    def run(
        self,
        schema_json_template: str,
        context_open_tag: str,
        context_close_tag: str,
        base_instructions: str,
        user_prompt: str,
    ) -> tuple[str, str]:
        ot = (context_open_tag or "<context>").strip() or "<context>"
        ct = (context_close_tag or "</context>").strip() or "</context>"
        tmpl = (schema_json_template or "").strip()
        instr = (base_instructions or "").strip()
        sys_add = (
            "=== OUTPUT CONTRACT (strict) ===\n"
            f"{instr}\n\n"
            f"Tags for updated context: {ot} ... {ct}\n\n"
            "JSON TEMPLATE (fill values; keep keys/structure):\n"
            f"{tmpl}\n"
            "=== END CONTRACT ==="
        ).strip()
        wrapped = (
            (user_prompt or "").strip()
            + "\n\n"
            + "Follow the OUTPUT CONTRACT exactly."
        ).strip()
        return (sys_add, wrapped)


class ContextSchemaParser:
    """Extracts updated context between tags (and returns the raw model text too)."""

    CATEGORY = "llm/multimodal"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("updated_context", "raw_text")

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "model_text": ("STRING", {"default": "", "multiline": True}),
                "context_open_tag": ("STRING", {"default": "<context>", "multiline": False}),
                "context_close_tag": ("STRING", {"default": "</context>", "multiline": False}),
                "fallback_to_empty": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label": "if no <context> block, output empty (else echo model_text)",
                    },
                ),
            }
        }

    def run(
        self,
        model_text: str,
        context_open_tag: str,
        context_close_tag: str,
        fallback_to_empty: bool,
    ) -> tuple[str, str]:
        text = str(model_text or "")
        ot = re.escape((context_open_tag or "<context>").strip() or "<context>")
        ct = re.escape((context_close_tag or "</context>").strip() or "</context>")
        m = re.search(ot + r"([\s\S]*?)" + ct, text, flags=re.DOTALL)
        if m:
            return (m.group(1).strip(), text)
        return ("", text) if fallback_to_empty else (text.strip(), text)


class GGUFSettingsSorter:
    """Heuristics-only "auto" settings based on selected GGUF filenames."""

    CATEGORY = "llm/multimodal"
    FUNCTION = "run"
    RETURN_TYPES = (
        "STRING",
        "INT",
        "BOOLEAN",
        "FLOAT",
        "FLOAT",
        "INT",
        "INT",
        "FLOAT",
        "BOOLEAN",
        "BOOLEAN",
    )
    RETURN_NAMES = (
        "gguf_vlm_handler",
        "gguf_n_ctx",
        "do_sample",
        "temperature",
        "top_p",
        "top_k",
        "max_new_tokens",
        "repetition_penalty",
        "use_repetition_penalty",
        "gguf_chat_template_enable_thinking",
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        _gguf_combo = gguf_picker.combo_choices_llm_gguf()
        _gguf_def_main, _gguf_def_mm = gguf_picker.suggest_default_main_and_mmproj()
        return {
            "required": {
                "gguf_main_models_llm": (_gguf_combo, {"default": _gguf_def_main}),
                "gguf_mmproj_models_llm": (_gguf_combo, {"default": _gguf_def_mm}),
                "prefer_fast": (
                    "BOOLEAN",
                    {"default": True, "label": "prefer speed (lower n_ctx, greedy)"},
                ),
            }
        }

    def run(
        self, gguf_main_models_llm: str, gguf_mmproj_models_llm: str, prefer_fast: bool
    ) -> tuple[Any, ...]:
        a = str(gguf_main_models_llm or "").lower()
        b = str(gguf_mmproj_models_llm or "").lower()
        base = f"{a} {b}"

        handler = "qwen3-vl"
        if "gemma-3" in base or "gemma3" in base:
            handler = "gemma3"
        elif "qwen3" in base:
            handler = "qwen3-vl"
        elif "qwen2.5" in base or "qwen25" in base or "qwen2_5" in base:
            handler = "qwen2.5-vl"
        elif "llava" in base:
            handler = "llava-1.6" if ("1.6" in base or "llava16" in base) else "llava-1.5"

        if "llava" in handler:
            n_ctx = 8192 if prefer_fast else 12288
        else:
            # Qwen VLM tends to like larger context; keep default.
            n_ctx = 12288 if prefer_fast else 16384

        do_sample = False if prefer_fast else True
        temperature = 0.7
        top_p = 0.95
        top_k = 20 if prefer_fast else 40
        max_new_tokens = 1024 if prefer_fast else 5000
        repetition_penalty = 1.0
        use_repetition_penalty = True
        gguf_chat_template_enable_thinking = False

        return (
            handler,
            int(n_ctx),
            bool(do_sample),
            float(temperature),
            float(top_p),
            int(top_k),
            int(max_new_tokens),
            float(repetition_penalty),
            bool(use_repetition_penalty),
            bool(gguf_chat_template_enable_thinking),
        )


class MultimodalContextHandler:
    """
    Build a single `extra_context` string from toggle-able blocks.
    Each block can come from an inline text field or a file path (e.g. .md).
    """

    CATEGORY = "llm/multimodal"
    FUNCTION = "run"
    RETURN_TYPES = (
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
    )
    RETURN_NAMES = (
        "extra_context",
        "block1",
        "block2",
        "block3",
        "block4",
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "join_with": (
                    ["blank_line", "double_blank_line"],
                    {"default": "double_blank_line"},
                ),
                "include_titles": ("BOOLEAN", {"default": True}),
                "block1_enable": ("BOOLEAN", {"default": True}),
                "block1_title": ("STRING", {"default": "Context 1", "multiline": False}),
                "block1_text": ("STRING", {"default": "", "multiline": True}),
                "block1_file": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "Optional path to .md/.txt (overrides text when set)",
                    },
                ),
                "block2_enable": ("BOOLEAN", {"default": False}),
                "block2_title": ("STRING", {"default": "Context 2", "multiline": False}),
                "block2_text": ("STRING", {"default": "", "multiline": True}),
                "block2_file": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "Optional path to .md/.txt (overrides text when set)",
                    },
                ),
                "block3_enable": ("BOOLEAN", {"default": False}),
                "block3_title": ("STRING", {"default": "Context 3", "multiline": False}),
                "block3_text": ("STRING", {"default": "", "multiline": True}),
                "block3_file": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "Optional path to .md/.txt (overrides text when set)",
                    },
                ),
                "block4_enable": ("BOOLEAN", {"default": False}),
                "block4_title": ("STRING", {"default": "Context 4", "multiline": False}),
                "block4_text": ("STRING", {"default": "", "multiline": True}),
                "block4_file": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "Optional path to .md/.txt (overrides text when set)",
                    },
                ),
            }
        }

    def run(
        self,
        join_with: str,
        include_titles: bool,
        block1_enable: bool,
        block1_title: str,
        block1_text: str,
        block1_file: str,
        block2_enable: bool,
        block2_title: str,
        block2_text: str,
        block2_file: str,
        block3_enable: bool,
        block3_title: str,
        block3_text: str,
        block3_file: str,
        block4_enable: bool,
        block4_title: str,
        block4_text: str,
        block4_file: str,
    ) -> tuple[str, str, str, str, str]:
        sep = "\n\n" if (join_with or "") == "blank_line" else "\n\n\n"

        def _one(enabled: bool, title: str, txt: str, fp: str) -> str:
            if not enabled:
                return ""
            file_txt = _read_text_file_maybe(fp)
            use_txt = file_txt if file_txt.strip() else (txt or "")
            if include_titles:
                return _format_block(title, use_txt)
            return (use_txt or "").strip()

        b1 = _one(block1_enable, block1_title, block1_text, block1_file)
        b2 = _one(block2_enable, block2_title, block2_text, block2_file)
        b3 = _one(block3_enable, block3_title, block3_text, block3_file)
        b4 = _one(block4_enable, block4_title, block4_text, block4_file)
        blocks = [b for b in (b1, b2, b3, b4) if (b or "").strip()]
        combined = sep.join(blocks).strip()
        return (combined, b1, b2, b3, b4)


class MultimodalSystemPrompter:
    """Preset + file/custom system prompts, with separate outputs for easy wiring."""

    CATEGORY = "llm/multimodal"
    FUNCTION = "run"
    RETURN_TYPES = (
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
    )
    RETURN_NAMES = (
        "selected",
        "preset_image_describer",
        "preset_image_prompt_writer",
        "preset_quality_control",
        "preset_context_updater",
        "from_file",
        "custom",
    )

    _FALLBACKS: dict[str, str] = {
        "image_describer": (
            "You are a precise image describer.\n"
            "Describe what you see with concrete, objective details.\n"
            "If asked for a prompt, produce a clean prompt-friendly description."
        ),
        "image_prompt_writer": (
            "You write high-quality image generation prompts.\n"
            "Given a description and constraints, output a single prompt with: subject, scene, style, lighting, camera, composition.\n"
            "Avoid extra commentary."
        ),
        "quality_control": (
            "You are a strict quality-control reviewer for generated images.\n"
            "Compare the intended goal vs what is shown.\n"
            "Output: (1) issues found (bullet list), (2) precise edit instructions to fix them, (3) an improved prompt."
        ),
        "context_updater": (
            "You maintain a running context/memory for a workflow.\n"
            "Given the current context and the latest interaction, decide if the context should be updated.\n"
            "If updating, output the FULL updated context between <context>...</context>."
        ),
        "custom": "",
    }

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "selection": (
                    [
                        "selected_only",
                        "image_describer",
                        "image_prompt_writer",
                        "quality_control",
                        "context_updater",
                        "from_file",
                        "custom",
                    ],
                    {"default": "image_describer"},
                ),
                "file_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "Optional path to .md/.txt system prompt",
                    },
                ),
                "custom": ("STRING", {"default": "", "multiline": True}),
            }
        }

    def run(self, selection: str, file_path: str, custom: str) -> tuple[str, ...]:
        p_img_desc = _read_preset(
            os.path.join("presets", "system", "image_describer.md"),
            self._FALLBACKS["image_describer"],
        )
        p_prompt = _read_preset(
            os.path.join("presets", "system", "image_prompt_writer.md"),
            self._FALLBACKS["image_prompt_writer"],
        )
        p_qc = _read_preset(
            os.path.join("presets", "system", "quality_control.md"),
            self._FALLBACKS["quality_control"],
        )
        p_ctx = _read_preset(
            os.path.join("presets", "system", "context_updater.md"),
            self._FALLBACKS["context_updater"],
        )
        _custom_default = _read_preset(
            os.path.join("presets", "system", "custom.md"),
            self._FALLBACKS["custom"],
        )
        from_file = _read_text_file_maybe(file_path).strip()
        custom_s = (custom or "").strip() or _custom_default.strip()

        sel = (selection or "").strip()
        selected = ""
        if sel in ("image_describer", "selected_only"):
            selected = p_img_desc
        elif sel == "image_prompt_writer":
            selected = p_prompt
        elif sel == "quality_control":
            selected = p_qc
        elif sel == "context_updater":
            selected = p_ctx
        elif sel == "from_file":
            selected = from_file
        elif sel == "custom":
            selected = custom_s
        else:
            selected = p_img_desc

        return (
            selected.strip(),
            p_img_desc.strip(),
            p_prompt.strip(),
            p_qc.strip(),
            p_ctx.strip(),
            from_file,
            custom_s,
        )


class MultimodalSystemPrompter:
    """Pick a system prompt preset/file/custom with a short preview."""

    CATEGORY = "llm/multimodal"
    FUNCTION = "run"
    RETURN_TYPES = (
        "STRING",
        "STRING",
    )
    RETURN_NAMES = (
        "system_prompt",
        "system_prompt_preview",
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        preset_files = _list_preset_files("system")
        # show filenames in dropdown, but keep a stable internal mapping to rel paths
        preset_labels = [_basename_no_ext(p) for p in preset_files]
        preset_map = {lbl: p for (lbl, p) in zip(preset_labels, preset_files)}
        return {
            "required": {
                "selection": (
                    [
                        "preset",
                        "from_file",
                        "custom",
                    ],
                    {"default": "preset"},
                ),
                "preset_file": (
                    preset_labels or ["(no presets found)"],
                    {"default": (preset_labels[0] if preset_labels else "(no presets found)")},
                ),
                "file_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "Optional path to .md/.txt system prompt",
                    },
                ),
                "custom": ("STRING", {"default": "", "multiline": True}),
                "preview_chars": (
                    "INT",
                    {"default": 1200, "min": 0, "max": 20000, "step": 50},
                ),
                "custom_empty_when_not_selected": (
                    "BOOLEAN",
                    {"default": True, "label": "clear custom unless selection=custom"},
                ),
            }
        }

    def run(
        self,
        selection: str,
        preset_file: str,
        file_path: str,
        custom: str,
        preview_chars: int,
        custom_empty_when_not_selected: bool,
    ) -> tuple[str, str]:
        preset_files = _list_preset_files("system")
        preset_labels = [_basename_no_ext(p) for p in preset_files]
        preset_map = {lbl: p for (lbl, p) in zip(preset_labels, preset_files)}
        rel_custom = os.path.join("presets", "system", "custom.md")
        custom_default = _read_preset(rel_custom, "")

        chosen_rel = preset_map.get(str(preset_file or "").strip(), "")
        chosen_path = _preset_abs_path(chosen_rel) if chosen_rel else ""
        chosen_text = _read_preset(chosen_rel, "") if chosen_rel else ""

        sel = (selection or "").strip()
        selected_text = ""
        selected_path = ""
        if sel == "preset":
            selected_text, selected_path = chosen_text, chosen_path
        elif sel == "from_file":
            selected_text, selected_path = _read_text_file_maybe(file_path).strip(), (file_path or "").strip()
        elif sel == "custom":
            selected_text, selected_path = (custom or "").strip() or custom_default.strip(), "custom"
        else:
            selected_text, selected_path = chosen_text, chosen_path

        if custom_empty_when_not_selected and sel != "custom":
            # UI helper only: keep field visually empty unless actively used.
            pass
        n = max(0, int(preview_chars))
        preview = selected_text if n <= 0 else (selected_text[:n] + ("…" if len(selected_text) > n else ""))

        return (selected_text, preview)

class MultimodalContextHandler:
    """Build a single `extra_context` from up to 4 toggle-able blocks (text/file/preset) + preview."""

    CATEGORY = "llm/multimodal"
    FUNCTION = "run"
    RETURN_TYPES = (
        "STRING",
        "STRING",
        "STRING",
    )
    RETURN_NAMES = (
        "extra_context",
        "extra_context_preview",
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        preset_files = _list_preset_files("context")
        preset_labels = [_basename_no_ext(p) for p in preset_files]
        return {
            "required": {
                "join_with": (
                    ["blank_line", "double_blank_line"],
                    {"default": "double_blank_line"},
                ),
                "include_titles": ("BOOLEAN", {"default": True}),
                "preview_chars": (
                    "INT",
                    {"default": 1600, "min": 0, "max": 20000, "step": 50},
                ),
                "block1_enable": ("BOOLEAN", {"default": True}),
                "block1_title": ("STRING", {"default": "Context 1", "multiline": False}),
                "block1_source": (["text", "file", "preset"], {"default": "preset"}),
                "block1_preset_file": (
                    preset_labels or ["(no presets found)"],
                    {"default": (preset_labels[0] if preset_labels else "(no presets found)")},
                ),
                "block1_text": ("STRING", {"default": "", "multiline": True}),
                "block1_file": ("STRING", {"default": "", "multiline": False}),
                "block2_enable": ("BOOLEAN", {"default": False}),
                "block2_title": ("STRING", {"default": "Context 2", "multiline": False}),
                "block2_source": (["text", "file", "preset"], {"default": "preset"}),
                "block2_preset_file": (
                    preset_labels or ["(no presets found)"],
                    {"default": (preset_labels[0] if preset_labels else "(no presets found)")},
                ),
                "block2_text": ("STRING", {"default": "", "multiline": True}),
                "block2_file": ("STRING", {"default": "", "multiline": False}),
                "block3_enable": ("BOOLEAN", {"default": False}),
                "block3_title": ("STRING", {"default": "Context 3", "multiline": False}),
                "block3_source": (["text", "file", "preset"], {"default": "preset"}),
                "block3_preset_file": (
                    preset_labels or ["(no presets found)"],
                    {"default": (preset_labels[0] if preset_labels else "(no presets found)")},
                ),
                "block3_text": ("STRING", {"default": "", "multiline": True}),
                "block3_file": ("STRING", {"default": "", "multiline": False}),
                "block4_enable": ("BOOLEAN", {"default": False}),
                "block4_title": ("STRING", {"default": "Context 4", "multiline": False}),
                "block4_source": (["text", "file", "preset"], {"default": "preset"}),
                "block4_preset_file": (
                    preset_labels or ["(no presets found)"],
                    {"default": (preset_labels[0] if preset_labels else "(no presets found)")},
                ),
                "block4_text": ("STRING", {"default": "", "multiline": True}),
                "block4_file": ("STRING", {"default": "", "multiline": False}),
            }
        }

    def run(self, **kw: Any) -> tuple[str, ...]:
        sep = "\n\n" if (kw.get("join_with") or "") == "blank_line" else "\n\n\n"

        preset_files = _list_preset_files("context")
        preset_labels = [_basename_no_ext(p) for p in preset_files]
        preset_map = {lbl: p for (lbl, p) in zip(preset_labels, preset_files)}

        def _resolve_block(i: int) -> tuple[str, str]:
            if not bool(kw.get(f"block{i}_enable")):
                return ("", "")
            title = str(kw.get(f"block{i}_title") or "").strip()
            src = str(kw.get(f"block{i}_source") or "text").strip()
            txt = str(kw.get(f"block{i}_text") or "")
            fp = str(kw.get(f"block{i}_file") or "").strip()
            preset_label = str(kw.get(f"block{i}_preset_file") or "").strip()
            path = ""
            content = ""
            if src == "preset":
                rel = preset_map.get(preset_label, "")
                path = _preset_abs_path(rel) if rel else ""
                content = _read_preset(rel, "") if rel else ""
            elif src == "file":
                path = fp
                content = _read_text_file_maybe(fp)
            else:
                content = txt
            content = (content or "").strip()
            if not content:
                return ("", path)
            if bool(kw.get("include_titles")):
                return (_format_block(title, content), path)
            return (content, path)

        b1, p1 = _resolve_block(1)
        b2, p2 = _resolve_block(2)
        b3, p3 = _resolve_block(3)
        b4, p4 = _resolve_block(4)
        blocks = [b for b in (b1, b2, b3, b4) if (b or "").strip()]
        combined = sep.join(blocks).strip()
        n = max(0, int(kw.get("preview_chars") or 0))
        preview = combined if n <= 0 else (combined[:n] + ("…" if len(combined) > n else ""))
        return (
            combined,
            preview,
        )


class MultimodalContextHandlerDebug:
    """Same as Context Handler, but also outputs each resolved block + its source path."""

    CATEGORY = "llm/multimodal"
    FUNCTION = "run"
    RETURN_TYPES = (
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
    )
    RETURN_NAMES = (
        "extra_context",
        "block1_text",
        "block1_path",
        "block2_text",
        "block2_path",
        "block3_text",
        "block3_path",
        "block4_text",
        "block4_path",
        "extra_context_preview",
    )

    INPUT_TYPES = MultimodalContextHandler.INPUT_TYPES  # type: ignore

    def run(self, **kw: Any) -> tuple[str, ...]:
        sep = "\n\n" if (kw.get("join_with") or "") == "blank_line" else "\n\n\n"
        preset_files = _list_preset_files("context")
        preset_labels = [_basename_no_ext(p) for p in preset_files]
        preset_map = {lbl: p for (lbl, p) in zip(preset_labels, preset_files)}

        def _resolve_block(i: int) -> tuple[str, str]:
            if not bool(kw.get(f"block{i}_enable")):
                return ("", "")
            title = str(kw.get(f"block{i}_title") or "").strip()
            src = str(kw.get(f"block{i}_source") or "text").strip()
            txt = str(kw.get(f"block{i}_text") or "")
            fp = str(kw.get(f"block{i}_file") or "").strip()
            preset_label = str(kw.get(f"block{i}_preset_file") or "").strip()
            path = ""
            content = ""
            if src == "preset":
                rel = preset_map.get(preset_label, "")
                path = _preset_abs_path(rel) if rel else ""
                content = _read_preset(rel, "") if rel else ""
            elif src == "file":
                path = fp
                content = _read_text_file_maybe(fp)
            else:
                content = txt
            content = (content or "").strip()
            if not content:
                return ("", path)
            if bool(kw.get("include_titles")):
                return (_format_block(title, content), path)
            return (content, path)

        b1, p1 = _resolve_block(1)
        b2, p2 = _resolve_block(2)
        b3, p3 = _resolve_block(3)
        b4, p4 = _resolve_block(4)
        blocks = [b for b in (b1, b2, b3, b4) if (b or "").strip()]
        combined = sep.join(blocks).strip()
        n = max(0, int(kw.get("preview_chars") or 0))
        preview = combined if n <= 0 else (combined[:n] + ("…" if len(combined) > n else ""))
        return (combined, b1, p1, b2, p2, b3, p3, b4, p4, preview)

NODE_CLASS_MAPPINGS = {
    "MultimodalLLMNode": MultimodalLLMNode,
    "MultimodalLLMEnvironmentReport": MultimodalLLMEnvironmentReport,
    "MultimodalLLMWheelRecommendation": MultimodalLLMWheelRecommendation,
    "UnloadHostedMultimodalLLMCache": UnloadHostedMultimodalLLMCache,
    "ContextSchemaBuilder": ContextSchemaBuilder,
    "ContextSchemaParser": ContextSchemaParser,
    "GGUFSettingsSorter": GGUFSettingsSorter,
    "MultimodalContextHandler": MultimodalContextHandler,
    "MultimodalContextHandlerDebug": MultimodalContextHandlerDebug,
    "MultimodalSystemPrompter": MultimodalSystemPrompter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MultimodalLLMNode": "Multimodal — LLM (local first: GGUF + optional transformers)",
    "MultimodalLLMEnvironmentReport": "Multimodal — CUDA / Python report",
    "MultimodalLLMWheelRecommendation": "Multimodal — llama-cpp wheel pick (GitHub)",
    "UnloadHostedMultimodalLLMCache": "Multimodal — Unload in-ComfyUI LLM / VLM cache",
    "ContextSchemaBuilder": "Multimodal — Context schema builder (template → prompt)",
    "ContextSchemaParser": "Multimodal — Context schema parser (<context>…</context>)",
    "GGUFSettingsSorter": "Multimodal — GGUF settings sorter (auto defaults)",
    "MultimodalContextHandler": "Multimodal — Context handler",
    "MultimodalContextHandlerDebug": "Multimodal — Context handler (debug outputs)",
    "MultimodalSystemPrompter": "Multimodal — System prompter",
}
