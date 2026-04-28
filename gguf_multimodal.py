"""
Optional GGUF + mmproj vision path (llama-cpp-python), e.g. LM Studio–style pairs:
  main weights .gguf + mmproj-*.gguf
"""

from __future__ import annotations

import base64
import hashlib
import io
import os
import warnings
from typing import Any, Callable, Optional, Tuple

_LLAMA_CACHE: dict[str, Any] = {}

_QWEN3_HANDLER_MISSING_MSG = (
    "Qwen3-VL GGUF needs Qwen3VLChatHandler in the same Python environment "
    "ComfyUI uses. Your llama-cpp-python wheel does not export it.\n\n"
    "Fix (pick one):\n"
    "  1) Install a build that includes Qwen3-VL (often from "
    "https://github.com/JamePeng/llama-cpp-python/releases — match CUDA vs CPU).\n"
    "     Example: run ComfyUI's python.exe -m pip install <downloaded_wheel.whl>\n"
    "  2) Use Qwen2.5-VL GGUF weights from Hugging Face and set gguf_vlm_handler to qwen2.5-vl.\n"
    "  3) Clear both GGUF path fields and set combined_model for a transformers VLM instead.\n\n"
    "To try a degraded run anyway, enable the node option "
    "'GGUF: allow Qwen2.5 handler if Qwen3 missing' (vision usually will not work)."
)


def _gguf_vlm_cache_eff_tag(
    handler_name: str, allow_qwen25_when_qwen3_missing: bool
) -> str:
    """Stable cache key segment; raises if Qwen3 is required but unavailable (no fallback)."""
    h = (handler_name or "qwen2.5-vl").strip().lower()
    if h in ("qwen3-vl", "qwen3vl", "qwen3_vl"):
        try:
            from llama_cpp.llama_chat_format import Qwen3VLChatHandler  # noqa: F401
        except ImportError as e:
            if allow_qwen25_when_qwen3_missing:
                return "qwen3-vl|fallback_qwen25"
            raise RuntimeError(_QWEN3_HANDLER_MISSING_MSG) from e
        return "qwen3-vl"
    return h


def gguf_resolve_cache_tag(
    handler_name: str, allow_qwen25_when_qwen3_missing: bool
) -> str:
    """Which GGUF chat-handler flavor will load (for logs). Matches the internal Llama cache key segment."""
    return _gguf_vlm_cache_eff_tag(handler_name, allow_qwen25_when_qwen3_missing)


def normalize_gguf_vlm_handler_for_filenames(
    main_gguf: str, mmproj_gguf: str, handler_name: str
) -> str:
    """
    If GGUF basenames look like Qwen3-VL but the widget still says Qwen2.5-style,
    coerce to qwen3-vl (common after workflow / widget order drift).
    """
    h = (handler_name or "").strip().lower()
    base = f"{os.path.basename(main_gguf)} {os.path.basename(mmproj_gguf)}".lower()
    if "qwen3" in base and h in ("qwen2.5-vl", "qwen25-vl", "qwen_vl", "auto"):
        warnings.warn(
            "GGUF filenames look like Qwen3-VL but handler was Qwen2.5-style; "
            "using qwen3-vl. Set gguf_vlm_handler explicitly after fixing the graph.",
            UserWarning,
            stacklevel=2,
        )
        return "qwen3-vl"
    return (handler_name or "qwen2.5-vl").strip()


def _llama_import_error() -> Exception | None:
    try:
        import llama_cpp  # noqa: F401
    except ImportError as e:
        return e
    return None


def _cache_key(parts: tuple[Any, ...]) -> str:
    return hashlib.sha256(repr(parts).encode("utf-8")).hexdigest()[:24]


def _try_free_comfy_vram() -> bool:
    """
    Best-effort attempt to free GPU memory inside a running ComfyUI process.
    Safe to call outside ComfyUI (no-ops).
    """
    freed = False
    try:
        import torch

        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                freed = True
            except Exception:
                pass
    except Exception:
        pass

    # ComfyUI's own model manager (only exists when running inside ComfyUI).
    try:
        import importlib

        mm = importlib.import_module("comfy.model_management")
        for fn_name in (
            "soft_empty_cache",
            "unload_all_models",
            "unload_all_models_except",
            "cleanup_models",
        ):
            fn = getattr(mm, fn_name, None)
            if callable(fn):
                try:
                    fn()
                    freed = True
                except Exception:
                    pass
    except Exception:
        pass

    return freed


def _pil_to_data_url(pil_image: Any) -> str:
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _make_chat_handler(
    handler_name: str,
    mmproj_path: str,
    verbose: bool = False,
    *,
    gguf_chat_template_enable_thinking: bool = False,
    allow_qwen25_when_qwen3_missing: bool = False,
) -> Tuple[Any, str]:
    from llama_cpp.llama_chat_format import (
        Gemma3ChatHandler,
        Llava15ChatHandler,
        Llava16ChatHandler,
        Qwen25VLChatHandler,
    )

    h = (handler_name or "qwen2.5-vl").strip().lower()
    if h in ("gemma3", "gemma-3", "gemma_3"):
        return Gemma3ChatHandler(clip_model_path=mmproj_path, verbose=verbose), "gemma3"
    if h in ("qwen3-vl", "qwen3vl", "qwen3_vl"):
        try:
            from llama_cpp.llama_chat_format import Qwen3VLChatHandler
        except ImportError as e:
            if allow_qwen25_when_qwen3_missing:
                warnings.warn(
                    "This llama-cpp-python has no Qwen3VLChatHandler; using "
                    "Qwen25VLChatHandler instead. Qwen3-VL GGUF will usually misbehave "
                    "(no vision or wrong replies). Install a Qwen3-capable build "
                    "(https://github.com/JamePeng/llama-cpp-python/releases) or use "
                    "Qwen2.5-VL GGUF with handler qwen2.5-vl.",
                    UserWarning,
                    stacklevel=2,
                )
                return (
                    Qwen25VLChatHandler(clip_model_path=mmproj_path, verbose=verbose),
                    "qwen3-vl|fallback_qwen25",
                )
            raise RuntimeError(_QWEN3_HANDLER_MISSING_MSG) from e
        variants: list[dict[str, Any]] = []
        if gguf_chat_template_enable_thinking:
            variants.extend(
                [
                    {
                        "clip_model_path": mmproj_path,
                        "verbose": verbose,
                        "image_min_tokens": 1024,
                        "force_reasoning": True,
                    },
                    {
                        "clip_model_path": mmproj_path,
                        "verbose": verbose,
                        "force_reasoning": True,
                    },
                    {
                        "clip_model_path": mmproj_path,
                        "verbose": verbose,
                        "image_min_tokens": 1024,
                    },
                    {"clip_model_path": mmproj_path, "verbose": verbose},
                ]
            )
        else:
            variants.extend(
                [
                    {
                        "clip_model_path": mmproj_path,
                        "verbose": verbose,
                        "image_min_tokens": 1024,
                        "force_reasoning": False,
                    },
                    {
                        "clip_model_path": mmproj_path,
                        "verbose": verbose,
                        "force_reasoning": False,
                    },
                    {
                        "clip_model_path": mmproj_path,
                        "verbose": verbose,
                        "image_min_tokens": 1024,
                    },
                    {"clip_model_path": mmproj_path, "verbose": verbose},
                ]
            )
        last_te: TypeError | None = None
        for kw in variants:
            try:
                return Qwen3VLChatHandler(**kw), "qwen3-vl"
            except TypeError as te:
                last_te = te
                continue
        raise TypeError(
            "Qwen3VLChatHandler did not accept any known argument combination"
        ) from last_te
    if h in ("qwen2.5-vl", "qwen25-vl", "qwen_vl", "auto"):
        cls = Qwen25VLChatHandler
    elif h in ("llava-1.6", "llava1.6", "llava_16"):
        cls = Llava16ChatHandler
    elif h in ("llava-1.5", "llava1.5", "llava_15", "llava"):
        cls = Llava15ChatHandler
    else:
        raise ValueError(
            f"Unknown gguf_vlm_handler {handler_name!r}. "
            "Try gemma3 for Gemma 3 GGUF+mmproj, qwen3-vl for Qwen3-VL GGUF, qwen2.5-vl for Qwen2.5-VL, or llava-1.5 / llava-1.6."
        )
    return cls(clip_model_path=mmproj_path, verbose=verbose), h


def _get_llama_vlm(
    main_gguf: str,
    mmproj_gguf: str,
    handler_name: str,
    n_ctx: int,
    n_gpu_layers: int,
    *,
    gguf_chat_template_enable_thinking: bool = False,
    allow_qwen25_when_qwen3_missing: bool = False,
) -> Any:
    err = _llama_import_error()
    if err is not None:
        raise RuntimeError(
            "llama-cpp-python is required for GGUF+mmproj vision. "
            "Install: pip install llama-cpp-python  (use a CUDA wheel if you use GPU)"
        ) from err
    from llama_cpp import Llama

    mf = os.path.abspath(os.path.expanduser(os.path.expandvars((main_gguf or "").strip())))
    mp = os.path.abspath(os.path.expanduser(os.path.expandvars((mmproj_gguf or "").strip())))
    if not os.path.isfile(mf):
        raise ValueError(f"GGUF main file not found: {mf!r}")
    if not os.path.isfile(mp):
        raise ValueError(f"GGUF mmproj file not found: {mp!r}")

    eff_tag = _gguf_vlm_cache_eff_tag(handler_name, allow_qwen25_when_qwen3_missing)
    key = _cache_key(
        (
            "vlm",
            mf,
            mp,
            eff_tag,
            int(n_ctx),
            int(n_gpu_layers),
            bool(gguf_chat_template_enable_thinking),
            bool(allow_qwen25_when_qwen3_missing),
        )
    )
    if key not in _LLAMA_CACHE:
        ch, _eff2 = _make_chat_handler(
            handler_name,
            mp,
            verbose=False,
            gguf_chat_template_enable_thinking=gguf_chat_template_enable_thinking,
            allow_qwen25_when_qwen3_missing=allow_qwen25_when_qwen3_missing,
        )
        llama_kw: dict[str, Any] = {
            "model_path": mf,
            "chat_handler": ch,
            "n_ctx": int(n_ctx),
            "n_gpu_layers": int(n_gpu_layers),
            "verbose": False,
            "chat_template_kwargs": {
                "enable_thinking": bool(gguf_chat_template_enable_thinking),
            },
        }
        try:
            _LLAMA_CACHE[key] = Llama(**llama_kw)
        except TypeError:
            del llama_kw["chat_template_kwargs"]
            _LLAMA_CACHE[key] = Llama(**llama_kw)
        except ValueError as e:
            # llama-cpp-python raises a generic ValueError for several root causes:
            # - VRAM allocation failure (common when another process uses the GPU)
            # - incompatible / corrupted GGUF
            # - wheel/build mismatch for the GGUF features needed
            msg = str(e)
            is_generic_load_fail = "Failed to load model from file" in msg
            if not is_generic_load_fail:
                raise

            # Retry 1: ask ComfyUI to free VRAM and try again with the same settings.
            # This helps when another SD model is still resident on the GPU.
            if int(n_gpu_layers) != 0 and _try_free_comfy_vram():
                warnings.warn(
                    "GGUF model load failed; attempting to free ComfyUI VRAM and retry model load.",
                    UserWarning,
                    stacklevel=2,
                )
                try:
                    _LLAMA_CACHE[key] = Llama(**llama_kw)
                    return _LLAMA_CACHE[key], key
                except TypeError:
                    # If chat_template_kwargs wasn't supported, retry minimal.
                    llama_kw2 = dict(llama_kw)
                    llama_kw2.pop("chat_template_kwargs", None)
                    _LLAMA_CACHE[key] = Llama(**llama_kw2)
                    return _LLAMA_CACHE[key], key
                except ValueError:
                    pass

            # Retry 2 (safe fallback): fall back to CPU layers (often succeeds when VRAM is fragmented/occupied).
            # If the wheel doesn't accept the extra args, we just retry with the minimal set.
            if int(n_gpu_layers) != 0:
                warnings.warn(
                    "GGUF model load failed (often VRAM busy/fragmented). Retrying with n_gpu_layers=0 (CPU). "
                    "If this succeeds, reduce VRAM pressure (close extra ComfyUI instances, lower n_ctx) or install a better-matching llama-cpp-python wheel.",
                    UserWarning,
                    stacklevel=2,
                )
                llama_kw_retry = dict(llama_kw)
                llama_kw_retry["n_gpu_layers"] = 0
                llama_kw_retry.setdefault("use_mmap", False)
                llama_kw_retry.setdefault("use_mlock", False)
                try:
                    _LLAMA_CACHE[key] = Llama(**llama_kw_retry)
                    return _LLAMA_CACHE[key], key
                except TypeError:
                    # Older builds may not support use_mmap/use_mlock
                    llama_kw_retry.pop("use_mmap", None)
                    llama_kw_retry.pop("use_mlock", None)
                    _LLAMA_CACHE[key] = Llama(**llama_kw_retry)
                    return _LLAMA_CACHE[key], key

            # Re-raise with a higher-signal hint block.
            raise ValueError(
                f"{msg}\n\n"
                "Common causes:\n"
                "  - VRAM allocation failed (another ComfyUI instance or big SD model already on GPU)\n"
                "  - GGUF is incompatible with your llama-cpp-python build (wheel mismatch)\n"
                "  - GGUF file is corrupted / incomplete download\n\n"
                f"Checked paths:\n  main={mf}\n  mmproj={mp}\n"
            ) from e
    return _LLAMA_CACHE[key], key


def _get_llama_plain(
    main_gguf: str,
    n_ctx: int,
    n_gpu_layers: int,
    *,
    gguf_chat_template_enable_thinking: bool = False,
) -> tuple[Any, str]:
    """Text-only: load **main** .gguf only (no mmproj / VL chat_handler). Much faster than VLM stack without an image."""
    err = _llama_import_error()
    if err is not None:
        raise RuntimeError(
            "llama-cpp-python is required for GGUF text chat. "
            "Install: pip install llama-cpp-python  (use a CUDA wheel if you use GPU)"
        ) from err
    from llama_cpp import Llama

    mf = os.path.abspath(os.path.expanduser(os.path.expandvars((main_gguf or "").strip())))
    if not os.path.isfile(mf):
        raise ValueError(f"GGUF main file not found: {mf!r}")
    key = _cache_key(
        (
            "plain",
            mf,
            int(n_ctx),
            int(n_gpu_layers),
            bool(gguf_chat_template_enable_thinking),
        )
    )
    if key not in _LLAMA_CACHE:
        llama_kw: dict[str, Any] = {
            "model_path": mf,
            "n_ctx": int(n_ctx),
            "n_gpu_layers": int(n_gpu_layers),
            "verbose": False,
            "chat_template_kwargs": {
                "enable_thinking": bool(gguf_chat_template_enable_thinking),
            },
        }
        try:
            _LLAMA_CACHE[key] = Llama(**llama_kw)
        except TypeError:
            del llama_kw["chat_template_kwargs"]
            _LLAMA_CACHE[key] = Llama(**llama_kw)
    return _LLAMA_CACHE[key], key


def evict_llama_cache_key(key: str) -> None:
    llm = _LLAMA_CACHE.pop(key, None)
    if llm is None:
        return
    try:
        close = getattr(llm, "close", None)
        if callable(close):
            close()
    except Exception:
        pass
    try:
        del llm
    except Exception:
        pass


def clear_llama_gguf_cache() -> None:
    for k in list(_LLAMA_CACHE.keys()):
        evict_llama_cache_key(k)


def _execute_llama_chat_completion(
    llm: Any,
    kwargs: dict[str, Any],
    *,
    gguf_chat_template_enable_thinking: bool,
    vision: bool,
    on_chunk: Optional[Callable[[str], None]] = None,
) -> str:
    """Shared create_chat_completion + stream decode for VLM and plain GGUF paths."""
    stream = bool(kwargs.get("stream", False))

    def _create_chat() -> Any:
        merged = dict(kwargs)
        merged["chat_template_kwargs"] = {
            "enable_thinking": bool(gguf_chat_template_enable_thinking),
        }
        try:
            return llm.create_chat_completion(**merged)
        except TypeError:
            pass
        if not gguf_chat_template_enable_thinking:
            try:
                merged2 = dict(kwargs)
                merged2["chat_template_kwargs"] = {"enable_thinking": False}
                return llm.create_chat_completion(**merged2)
            except TypeError:
                pass
        if not gguf_chat_template_enable_thinking:
            warnings.warn(
                "create_chat_completion fell back without chat_template_kwargs — some llama-cpp "
                "builds ignore enable_thinking=False at the API; output may still contain Qwen "
                '`think` fences (stripped when "keep thinking blocks" is off).',
                UserWarning,
                stacklevel=3,
            )
        try:
            return llm.create_chat_completion(**kwargs)
        except TypeError:
            k2 = {k: v for k, v in kwargs.items() if k != "seed"}
            if len(k2) != len(kwargs):
                warnings.warn(
                    "create_chat_completion ignored unsupported `seed=` (retrying without seed). "
                    "Upgrade llama-cpp-python for deterministic sampling.",
                    UserWarning,
                    stacklevel=3,
                )
                return llm.create_chat_completion(**k2)
            raise

    if stream:
        stream_iter = _create_chat()
        parts: list[str] = []
        for chunk in stream_iter:
            try:
                ch0 = (chunk or {}).get("choices", [{}])[0]
                delta = ch0.get("delta") or {}
                piece = delta.get("content") or ""
            except (IndexError, TypeError, AttributeError):
                piece = ""
            if piece:
                parts.append(piece)
                if on_chunk:
                    on_chunk("".join(parts))
        content = "".join(parts)
        if not str(content).strip():
            raise RuntimeError(
                "GGUF streaming returned empty text. Try gguf_streaming OFF, or check handler / n_ctx / paths as for non-stream errors."
            )
    else:
        out = _create_chat()
        choice = out["choices"][0]
        content = ""
        msg = choice.get("message")
        if isinstance(msg, dict):
            content = msg.get("content") or ""
        if not content:
            content = choice.get("text") or ""
    out_s = str(content).strip()
    if not out_s:
        base = (
            "GGUF vision returned empty text"
            if vision
            else "GGUF text-only chat returned empty text"
        )
        raise RuntimeError(
            f"{base}. Check: (1) "
            + (
                "use_vision ON + IMAGE connected, "
                if vision
                else ""
            )
            + "(2) gguf_vlm_handler matches weights (qwen3-vl vs qwen2.5-vl, or llava-1.5 / llava-1.6), "
            "(3) raise gguf_n_ctx (vision uses extra tokens), "
            "(4) main .gguf vs mmproj .gguf not swapped (we auto-swap if filename contains 'mmproj')."
        )
    return out_s


def run_gguf_plain_text_chat(
    *,
    main_gguf: str,
    system_prompt: str,
    extra_context: str,
    user_prompt: str,
    handler_name: str,
    n_ctx: int,
    n_gpu_layers: int,
    max_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    repeat_penalty: float,
    use_repeat_penalty: bool = True,
    gguf_chat_template_enable_thinking: bool = False,
    stream: bool = False,
    on_chunk: Optional[Callable[[str], None]] = None,
    seed: Optional[int] = None,
) -> tuple[str, str]:
    """
    Fast text-only path: **main .gguf only**, no mmproj / VL ChatHandler.
    Falls back to run_gguf_vlm_chat(..., pil=None) from the caller if this raises.
    """
    llm, key = _get_llama_plain(
        main_gguf,
        n_ctx,
        n_gpu_layers,
        gguf_chat_template_enable_thinking=gguf_chat_template_enable_thinking,
    )
    uparts: list[str] = []
    if (extra_context or "").strip():
        uparts.append((extra_context or "").strip())
    if (user_prompt or "").strip():
        uparts.append((user_prompt or "").strip())
    user_text = "\n\n".join(uparts).strip()
    hn_low = (handler_name or "").lower()
    if (
        not gguf_chat_template_enable_thinking
        and "qwen" in hn_low
        and "/no_think" not in (user_text or "")
        and "/nothink" not in (user_text or "").replace(" ", "").lower()
    ):
        user_text = f"{user_text} /no_think".strip() if user_text else "/no_think"

    sp = (system_prompt or "").strip()
    if not user_text.strip() and sp:
        user_text = "."
    if not user_text.strip():
        raise ValueError(
            "GGUF text-only chat needs prompt, extra context, or system prompt "
            "(same as transformers text-only)."
        )

    messages: list[dict[str, Any]] = []
    if sp:
        messages.append({"role": "system", "content": sp})
    messages.append({"role": "user", "content": user_text.strip()})

    kwargs: dict[str, Any] = {
        "messages": messages,
        "max_tokens": int(max_tokens),
    }
    if do_sample:
        kwargs["temperature"] = float(temperature)
        kwargs["top_p"] = float(top_p)
        if top_k > 0:
            kwargs["top_k"] = int(top_k)
    else:
        kwargs["temperature"] = 0.0
    if (
        use_repeat_penalty
        and abs(float(repeat_penalty) - 1.0) > 1e-6
    ):
        kwargs["repeat_penalty"] = float(repeat_penalty)

    kwargs["stream"] = bool(stream)
    if seed is not None:
        kwargs["seed"] = int(seed) & 0xFFFFFFFF

    out_s = _execute_llama_chat_completion(
        llm,
        kwargs,
        gguf_chat_template_enable_thinking=gguf_chat_template_enable_thinking,
        vision=False,
        on_chunk=on_chunk,
    )
    return (out_s, key)


def run_gguf_vlm_chat(
    *,
    main_gguf: str,
    mmproj_gguf: str,
    pil_image: Optional[Any] = None,
    system_prompt: str,
    extra_context: str,
    user_prompt: str,
    handler_name: str,
    n_ctx: int,
    n_gpu_layers: int,
    max_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    repeat_penalty: float,
    use_repeat_penalty: bool = True,
    gguf_chat_template_enable_thinking: bool = False,
    allow_qwen25_when_qwen3_missing: bool = False,
    stream: bool = False,
    on_chunk: Optional[Callable[[str], None]] = None,
    seed: Optional[int] = None,
) -> tuple[str, str]:
    """
    Returns (assistant_text, cache_key) so the caller can evict the right entry.
    If stream=True, tokens are accumulated; on_chunk receives cumulative text when set.
    """
    main_gguf, mmproj_gguf = normalize_gguf_pair(main_gguf, mmproj_gguf)
    a = (main_gguf or "").strip()
    b = (mmproj_gguf or "").strip()
    if a and b:
        ap = os.path.normcase(
            os.path.abspath(os.path.expanduser(os.path.expandvars(a)))
        )
        bp = os.path.normcase(
            os.path.abspath(os.path.expanduser(os.path.expandvars(b)))
        )
        if ap == bp:
            raise ValueError(
                "GGUF main path and mmproj path are the same file. "
                "You need two files: the main LLM .gguf and a separate mmproj-*.gguf (vision)."
            )
    handler_name = normalize_gguf_vlm_handler_for_filenames(
        main_gguf, mmproj_gguf, handler_name
    )
    llm, key = _get_llama_vlm(
        main_gguf,
        mmproj_gguf,
        handler_name,
        n_ctx,
        n_gpu_layers,
        gguf_chat_template_enable_thinking=gguf_chat_template_enable_thinking,
        allow_qwen25_when_qwen3_missing=allow_qwen25_when_qwen3_missing,
    )
    uparts: list[str] = []
    if (extra_context or "").strip():
        uparts.append((extra_context or "").strip())
    if (user_prompt or "").strip():
        uparts.append((user_prompt or "").strip())
    user_text = "\n\n".join(uparts).strip()
    # Qwen3: chat_template_kwargs(enable_thinking=False) is sometimes ignored on text-only turns in
    # llama-cpp; the official soft switch disables reasoning for that request (fixes slow /think runs).
    hn_low = (handler_name or "").lower()
    if (
        not gguf_chat_template_enable_thinking
        and "qwen" in hn_low
        and "/no_think" not in (user_text or "")
        and "/nothink" not in (user_text or "").replace(" ", "").lower()
    ):
        user_text = f"{user_text} /no_think".strip() if user_text else "/no_think"

    messages: list[dict[str, Any]] = []
    sp = (system_prompt or "").strip()
    if sp:
        messages.append({"role": "system", "content": sp})
    user_content: list[dict[str, Any]] = []
    imgs: list[Any] = []
    if pil_image is None:
        imgs = []
    elif isinstance(pil_image, (list, tuple)):
        imgs = [im for im in pil_image if im is not None]
    else:
        imgs = [pil_image]
    vision = len(imgs) > 0
    if vision:
        if not user_text and sp:
            # Non-empty user turn only where the stack requires it; not an instruction.
            user_text = "."
        if user_text:
            user_content.append({"type": "text", "text": user_text})
        for im in imgs:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _pil_to_data_url(im)},
                }
            )
    else:
        # Text-only via same GGUF+VLM stack (no pixels decoded upstream).
        if not user_text and sp:
            user_text = "."
        if not user_text.strip():
            raise ValueError(
                "GGUF text-only chat needs prompt, extra context, or system prompt "
                "(same as transformers text-only)."
            )
        user_content.append({"type": "text", "text": user_text.strip()})
    messages.append({"role": "user", "content": user_content})
    kwargs: dict[str, Any] = {
        "messages": messages,
        "max_tokens": int(max_tokens),
    }
    if do_sample:
        kwargs["temperature"] = float(temperature)
        kwargs["top_p"] = float(top_p)
        if top_k > 0:
            kwargs["top_k"] = int(top_k)
    else:
        kwargs["temperature"] = 0.0
    if (
        use_repeat_penalty
        and abs(float(repeat_penalty) - 1.0) > 1e-6
    ):
        kwargs["repeat_penalty"] = float(repeat_penalty)

    kwargs["stream"] = bool(stream)
    if seed is not None:
        kwargs["seed"] = int(seed) & 0xFFFFFFFF

    out_s = _execute_llama_chat_completion(
        llm,
        kwargs,
        gguf_chat_template_enable_thinking=gguf_chat_template_enable_thinking,
        vision=vision,
        on_chunk=on_chunk,
    )
    return (out_s, key)


def normalize_gguf_pair(main_gguf: str, mmproj_gguf: str) -> tuple[str, str]:
    """
    If filenames suggest main and mmproj were swapped (mmproj in first path), swap back.
    """
    a = (main_gguf or "").strip()
    b = (mmproj_gguf or "").strip()
    if not a or not b:
        return a, b
    ba = os.path.basename(a).lower()
    bb = os.path.basename(b).lower()
    if "mmproj" in ba and "mmproj" not in bb:
        return b, a
    if "mmproj" in bb and "mmproj" not in ba:
        return a, b
    return a, b


def gguf_paths_valid(main: str, mmproj: str) -> bool:
    m = (main or "").strip()
    p = (mmproj or "").strip()
    if not m and not p:
        return False
    if bool(m) ^ bool(p):
        return False
    m = os.path.abspath(os.path.expanduser(os.path.expandvars(m)))
    p = os.path.abspath(os.path.expanduser(os.path.expandvars(p)))
    return os.path.isfile(m) and os.path.isfile(p)
