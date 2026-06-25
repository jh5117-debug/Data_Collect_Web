# Strict Nested Zero-Shot Five-Fold Report

- Label: `STRICT NESTED PARTICIPANT-DISJOINT FIVE-FOLD V2`
- Outer-test folds are final-evaluation only.
- Development thresholds and model choice use inner-fold OOF predictions.

| Method | Recall mean | FPR mean | Precision mean | F1 mean |
|---|---:|---:|---:|---:|
| qwen_exact | 0.546406 | 0.000000 | 1.000000 | 0.678914 |
| stage1_only | 0.950890 | 0.093139 | 0.936792 | 0.942661 |
| stage2_bce | 0.945590 | 0.037708 | 0.973827 | 0.958948 |
| stage2_bce_supcon | 0.947442 | 0.016334 | 0.988127 | 0.967011 |
| validation_selected | 0.945590 | 0.016334 | 0.988127 | 0.966045 |

## V1 Comparison

V1 used one development fold as validation for each outer fold. V2 uses four inner held-out development folds for OOF selection.
