"""
Bonus nodes: detect image-generation intent in LLM (or any) text and expose prompts for CLIP + KSampler.

Does not run sampling inside this pack — wire outputs to CLIP Text Encode → KSampler (standard Comfy).
"""

from __future__ import annotations

import re
from typing import Tuple

# Intent: user asked for a new image from scratch (vs. analyzing an existing image).
_TRIGGER_HEAD = re.compile(
    r"(?is)"
    r"\b(?:please\s+)?(?:generate|create|make|draw|render|paint|illustrate)\s+"
    r"(?:me\s+)?(?:an?\s+)?(?:new\s+)?(?:image|picture|photo|illustration|drawing|artwork|visual)\b"
)
_TRIGGER_OF = re.compile(
    r"(?is)\b(?:an?\s+)?(?:image|picture|photo|illustration|drawing)\s+of\b"
)
_TRIGGER_SHOW = re.compile(
    r"(?is)\b(?:show|give)\s+(?:me\s+)?(?:an?\s+)?(?:image|picture|photo)\b"
)


def _strip_common_thinking_blocks(text: str) -> str:
    """Remove common Qwen-style reasoning blocks before parsing (same defaults as Multimodal LLM)."""
    if not (text or "").strip():
        return ""
    t = text
    open_tag = "<" + "redacted_thinking" + ">"
    close_tag = "</" + "redacted_thinking" + ">"
    t = re.sub(
        re.escape(open_tag) + r"[\s\S]*?" + re.escape(close_tag),
        "",
        t,
        flags=re.IGNORECASE,
    )
    return t.strip()


def _collapse_ws(s: str) -> str:
    return " ".join((s or "").split())


def _extract_subject(clean: str) -> Tuple[bool, str]:
    """
    Returns (matched_pattern, positive_prompt_fragment).
    """
    # "... image of a red cat ..."
    m = _TRIGGER_OF.search(clean)
    if m:
        start = m.end()
        rest = clean[start:].strip()
        # Take first sentence-like chunk
        frag = _first_sentence(rest)
        if frag:
            return True, frag

    # "generate an image ..." take text after keyword block
    m = _TRIGGER_HEAD.search(clean)
    if m:
        start = m.end()
        rest = clean[start:].strip()
        # Skip leading "of" / ":" / "that" / "showing"
        rest = re.sub(r"^(?:of|that|showing|with|depicting)\s+", "", rest, flags=re.I)
        frag = _first_sentence(rest)
        if frag:
            return True, frag

    m = _TRIGGER_SHOW.search(clean)
    if m:
        start = m.end()
        rest = clean[start:].strip()
        rest = re.sub(r"^(?:of|that)\s+", "", rest, flags=re.I)
        frag = _first_sentence(rest)
        if frag:
            return True, frag

    # Fallback: any strong trigger words together
    if _intent_keywords_only(clean):
        frag = _first_sentence(clean)
        return True, frag if frag else clean[:2000]

    return False, ""


def _intent_keywords_only(s: str) -> bool:
    low = s.lower()
    has_action = bool(
        re.search(
            r"\b(generate|create|make|draw|render|paint|illustrate)\b",
            low,
        )
    )
    has_target = bool(
        re.search(r"\b(image|picture|photo|illustration|drawing|artwork)\b", low)
    )
    has_of = "image of" in low or "picture of" in low or "photo of" in low
    return has_of or (has_action and has_target)


def _first_sentence(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    # Stop at newline or sentence end if very long
    line = s.split("\n")[0].strip()
    parts = re.split(r"(?<=[.!?])\s+", line)
    frag = parts[0].strip() if parts else line
    if len(frag) > 1800:
        frag = frag[:1800].rsplit(" ", 1)[0]
    return frag.strip(' "\'')


def parse_bonus_image_intent(
    text: str,
    *,
    strip_thinking: bool,
    positive_suffix: str,
    negative_default: str,
) -> Tuple[bool, str, str, str]:
    """
    Returns:
      run_branch, positive_prompt, negative_prompt, note
    """
    raw = (text or "").strip()
    if strip_thinking:
        raw = _strip_common_thinking_blocks(raw)
    raw = raw.strip()
    if not raw:
        return False, "", (negative_default or "").strip(), "empty input"

    ok_subj, subject = _extract_subject(raw)
    if not ok_subj:
        return (
            False,
            "",
            (negative_default or "").strip(),
            "no image-generation request detected",
        )

    pos = _collapse_ws(subject)
    suf = (positive_suffix or "").strip()
    if suf:
        pos = _collapse_ws(f"{pos}, {suf}") if pos else suf

    neg = (negative_default or "").strip()
    note = "parsed image prompt from LLM-style request"
    return True, pos, neg, note


class BonusImageIntentRouter:
    """
    Reads assistant/LLM text; if it asks to generate/draw an image, outputs prompts for SD.
    Connect **positive_prompt** / **negative_prompt** to CLIP Text Encode → KSampler.
    Use **run_image_gen** with a Switch / bypass to choose KSampler vs. other logic.
    """

    CATEGORY = "llm/multimodal"
    FUNCTION = "run"
    RETURN_TYPES = ("BOOLEAN", "STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = (
        "run_image_gen",
        "positive_prompt",
        "negative_prompt",
        "parse_note",
        "run_image_gen_int",
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "negative_prompt": (
                    "STRING",
                    {
                        "default": "worst quality, low quality, blurry, watermark, text, logo",
                        "multiline": True,
                    },
                ),
                "positive_suffix": (
                    "STRING",
                    {
                        "default": "high quality, detailed, coherent",
                        "multiline": False,
                        "placeholder": "Appended after extracted subject (comma-separated style)",
                    },
                ),
                "strip_thinking_tags": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label": "strip <think> / redacted_thinking blocks before parse",
                    },
                ),
            }
        }

    DESCRIPTION = (
        "**Bonus:** Detects phrases like “generate an image of …”, “draw a picture …”. "
        "Outputs **positive_prompt** / **negative_prompt** for your CLIP Encode → **KSampler** chain. "
        "**run_image_gen** is True when intent matches; use with any Switch/reroute node. "
        "English keyword heuristics only; tune **positive_suffix** for style."
    )

    def run(
        self,
        text: str,
        negative_prompt: str,
        positive_suffix: str,
        strip_thinking_tags: bool,
    ) -> tuple:
        run_b, pos, neg, note = parse_bonus_image_intent(
            text,
            strip_thinking=strip_thinking_tags,
            positive_suffix=positive_suffix,
            negative_default=negative_prompt,
        )
        return (run_b, pos, neg, note, 1 if run_b else 0)


NODE_CLASS_MAPPINGS = {
    "BonusImageIntentRouter": BonusImageIntentRouter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BonusImageIntentRouter": "Bonus: LLM text → image intent (KSampler prompts)",
}
