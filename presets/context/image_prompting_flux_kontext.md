# FLUX (Kontext/Klein) – context-focused prompt schema

Use when you want consistent character/style across iterations.

Include:
- Identity anchors: distinctive features, clothing, palette
- Style anchors: medium + reference style words
- Consistency constraints: "same person", "same outfit", "same style"
- Change request: what to change this iteration (only 1–3 deltas)

Output:
- Main prompt (single block)
- Optional negative prompt (short)
