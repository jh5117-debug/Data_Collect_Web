# Real Few-Shot Onboarding Report

- Device: `cuda:0`
- Learned personalization claimed: `True`
- Support clips are positive-only and removed from paired query sets.
- Development pseudo-targets select the recipe; outer-test participants are reporting-only.

| Condition | Recall | FPR | Precision | F1 |
|---|---:|---:|---:|---:|
| 0-shot on 3-shot query | 0.9564154786150713 | 0.016129032258064516 | 0.985312631137222 | 0.9706490285241836 |
| 3-shot | 0.9462321792260693 | 0.016129032258064516 | 0.9851569126378287 | 0.9653023062538957 |
| 0-shot on 5-shot query | 0.9599088838268792 | 0.016129032258064516 | 0.9836601307189542 | 0.9716393820613326 |
| 5-shot | 0.9535307517084283 | 0.016129032258064516 | 0.9835526315789473 | 0.9683090446449225 |
