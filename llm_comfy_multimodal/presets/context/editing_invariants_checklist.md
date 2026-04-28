# Editing invariants checklist (Preserve vs Change)

Use this to keep edits consistent and prevent drift.

## Preserve (unless explicitly changed)
- Identity: same person/character, same face structure, same age, same skin texture
- Hair: same style + color
- Anatomy: correct hands, correct limb counts, no distortions
- Pose/body proportions (unless pose change requested)
- Camera: same angle, framing, perspective, focal length feel
- Lighting: same direction, softness/hardness, shadow logic, color temperature
- Background: same scene + objects (unless background change requested)
- Style/medium: photo vs illustration vs 3D, same rendering style

## Change (only what you request)
- List the exact deltas (1–3 deltas per iteration works best)

## Avoid (common failures)
- text, watermark, logo
- blurry, low quality, jpeg artifacts
- extra fingers/limbs, deformed hands, distorted face
- identity drift, hair color drift, outfit drift
- flicker/morphing (video)

## Iteration rule
- Keep the preserve list stable every round.
- Modify one variable at a time.

