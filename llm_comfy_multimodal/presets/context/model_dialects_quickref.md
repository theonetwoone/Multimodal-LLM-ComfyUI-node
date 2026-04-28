# Model dialects quick reference

Use this as a reminder for how to format prompts per generator.

## SDXL
- Likes structured spec + optional tag stack.
- Can use weighting `(token:1.2)` and sometimes `BREAK`.
- Negatives are supported and useful; keep them short and targeted.
- Front-load subject and identity anchors.

## FLUX
- Prefer structured natural language (subject + action + style + context).
- Do NOT rely on SDXL weighting syntax; it’s usually ignored.
- Usually avoid negative prompts unless your specific pipeline supports them.
- 30–80 words is a good sweet spot for complex scenes.

## Z-Image
- Plain natural language, explicit constraints.
- Subject first; then scene/lighting/composition/style/materials.

## WAN (video)
- Focus on motion + camera movement + environmental motion.
- Time markers can help multi-beat motion: `[0-3s] ... [3-7s] ...`
- Negatives: flicker, morphing, jitter, distorted face, watermark/text.

## LTX (video)
- One flowing paragraph, present tense, 4–8 sentences.
- Include shot/camera, scene/lighting/atmosphere, characters (physical details), action sequence, camera movement end-state.

