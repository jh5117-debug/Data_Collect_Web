# Stage Error Analysis

- Variant: `stage2_bce`
- Evaluated clips: `1026`
- Error records: `169`
- False rejects: `137`
- Stage 1 miss false rejects: `55`
- Stage 2 reject false rejects: `82`
- False-reject percentage caused by Stage 1: `0.40145985401459855`
- False-reject percentage caused by Stage 2: `0.5985401459854015`

## Error Counts

| Category | Count |
|---|---:|
| FINAL_FALSE_ACCEPT | 2 |
| STAGE1_FALSE_CANDIDATE | 30 |
| STAGE1_MISS | 55 |
| STAGE2_REJECT | 82 |

## Errors By Prompt

| Category | Prompt | Count |
|---|---|---:|
| FINAL_FALSE_ACCEPT | P4_negative | 2 |
| STAGE1_FALSE_CANDIDATE | P4_negative | 30 |
| STAGE1_MISS | P1_vigil_only | 23 |
| STAGE1_MISS | P2_phrase_plus_vigil | 20 |
| STAGE1_MISS | P3_vigil_plus_phrase | 12 |
| STAGE2_REJECT | P1_vigil_only | 44 |
| STAGE2_REJECT | P2_phrase_plus_vigil | 16 |
| STAGE2_REJECT | P3_vigil_plus_phrase | 22 |

## Interpretation

Outer-test errors are reporting-only. Few-shot recipe selection must use development pseudo-targets, not these errors.
