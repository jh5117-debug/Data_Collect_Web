# Strict Positive-Only Participant Onboarding

- Adaptation recipe: `no_adaptation_zero_shot_fallback`
- 3-shot eligible participants: `26`
- 5-shot eligible participants: `24`
- Target negatives never enter adaptation.
- Query sets remove support clips and are paired with the zero-shot comparison.
- Because the selected recipe is a safety fallback, 3-shot and 5-shot do not claim adaptation improvement.

| Condition | Recall | FPR | Precision | F1 |
|---|---:|---:|---:|---:|
| 0-shot | 0.7896503108553723 | 0.004 | 0.992517006802721 | 0.8804619950886032 |
| 3-shot | 0.772969562389626 | 0.0038461538461538464 | 0.9933333333333334 | 0.8799333600260071 |
| 5-shot | 0.807721121693264 | 0.004166666666666667 | 0.9916666666666667 | 0.8810126566121408 |
