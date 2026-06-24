# Balanced Versus Full-Data Ablation

Status: pending.

Balanced max-100 primary protocol completed with the validation-selected `stage2_bce` zero-shot result:

- Recall mean: `0.7652259275562515`
- FPR mean: `0.004651162790697674`
- Precision mean: `0.9951219512195122`
- F1 mean: `0.8629426150555287`

The full unbalanced selected-method ablation was not run in this pass. No full-data metric is fabricated here.

Exact next command direction: build an alias-only full unbalanced manifest using the same participant fold assignment, then run only the validation-selected two-stage method per outer fold and aggregate against this balanced result.
