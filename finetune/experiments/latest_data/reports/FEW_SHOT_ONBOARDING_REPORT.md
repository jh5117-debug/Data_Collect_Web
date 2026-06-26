# Strict Positive-Only Participant Onboarding

- Adaptation recipe: `no_adaptation_zero_shot_fallback`
- 3-shot eligible participants: `33`
- 5-shot eligible participants: `31`
- Target negatives never enter adaptation.
- Query sets remove support clips and are paired with the zero-shot comparison.
- Because the selected recipe is a safety fallback, 3-shot and 5-shot do not claim adaptation improvement.

| Condition | Recall | FPR | Precision | F1 |
|---|---:|---:|---:|---:|
| 0-shot | 0.8653316231711301 | 0.0046875 | 0.9913626309792444 | 0.9225699943498468 |
| 3-shot | 0.8505312779196801 | 0.004545454545454545 | 0.9923076923076923 | 0.9175142316828513 |
| 5-shot | 0.8810868294065446 | 0.004838709677419355 | 0.9903743315508021 | 0.9278570664199073 |
