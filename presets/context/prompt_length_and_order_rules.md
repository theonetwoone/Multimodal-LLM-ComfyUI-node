# Prompt length + word order rules (practical)

## Universal
- Put the main subject first.
- Put the key constraint/delta early (what must be true).
- Add lighting + composition/camera before style fluff.
- Iterate by changing 1–3 deltas per run.

## SDXL
- Structure: Subject → attributes → environment → lighting → camera → style → quality.
- Prompts that are too long can cause late details to be ignored; front-load.
- Negatives: short + targeted.

## FLUX
- Structured natural language is best.
- Too-short prompts under-specify; too-long prompts may conflict.
- Target ~30–80 words for most scenes.

## Video (WAN/LTX)
- Emphasize verbs and camera moves.
- Avoid contradictory motion instructions.
- Keep the motion plan simple: one main motion + one camera move.

