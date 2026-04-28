# JSON schema contracts (local / prompt-only)

When you need machine-readable output without native schema enforcement, use strict contracts.

## Contract rules
- Output JSON only. No markdown fences. No extra commentary.
- Preserve keys + structure exactly as given.
- Use double quotes for all keys and strings.
- Do not invent fields.
- If unknown: use empty string / null / false depending on the field type.

## Context update convention
- If asked to update a context block, write the FULL updated context into a single field AND (optionally) also between tags:

<context>
...full updated context...
</context>

## Failure handling (if you can retry)
- If your output would be invalid JSON, stop and output the smallest valid JSON object that matches the template with empty defaults.

