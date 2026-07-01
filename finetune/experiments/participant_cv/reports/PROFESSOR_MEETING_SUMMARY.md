# Professor Meeting Summary

## Dataset

| Item | Value |
|---|---:|
| Original clips | 1298 |
| Formal balanced clips | 1026 |
| Balanced windows | 1040 |
| Participants | 30 |
| Positive clips | 585 |
| Negative clips | 441 |

## Folds

| Fold | Participants | Clips | Pos | Neg |
|---:|---:|---:|---:|---:|
| 0 | 6 | 205 | 119 | 86 |
| 1 | 6 | 208 | 124 | 84 |
| 2 | 6 | 212 | 125 | 87 |
| 3 | 6 | 206 | 108 | 98 |
| 4 | 6 | 195 | 109 | 86 |

## Zero-Shot Results

| Method | Recall | FPR | Precision | F1 |
|---|---:|---:|---:|---:|
| qwen_exact | 0.54640630845742 | 0.0 | 1.0 | 0.6789141371086982 |
| stage1_only | 0.9055543469408414 | 0.07431277242252153 | 0.946636506398043 | 0.9246570588270571 |
| stage2_bce | 0.7652259275562515 | 0.004651162790697674 | 0.9951219512195122 | 0.8629426150555287 |
| stage2_bce_supcon | 0.8314670285805289 | 0.011627906976744186 | 0.9900990099009901 | 0.9033316349354116 |
| validation_selected | 0.7652259275562515 | 0.004651162790697674 | 0.9951219512195122 | 0.8629426150555287 |

## 0/3/5-Shot Onboarding

| Condition | Recall | FPR | Precision | F1 |
|---|---:|---:|---:|---:|
| 0-shot | 0.7896503108553723 | 0.004 | 0.992517006802721 | 0.8804619950886032 |
| 3-shot | 0.772969562389626 | 0.0038461538461538464 | 0.9933333333333334 | 0.8799333600260071 |
| 5-shot | 0.807721121693264 | 0.004166666666666667 | 0.9916666666666667 | 0.8810126566121408 |

Adaptation recipe: `no_adaptation_zero_shot_fallback`. This is a safety fallback, so we do not claim few-shot improvement yet.

## Accuracy-Cost Table

| System | Recall | FPR | F1 | Extra encoder forward |
|---|---:|---:|---:|---:|
| Qwen transcript keyword | 0.54640630845742 | 0.0 | 0.6789141371086982 | 0 |
| Stage 1 only + continuous ASR | 0.9055543469408414 | 0.07431277242252153 | 0.9246570588270571 | 0 |
| Two-stage BCE + continuous ASR | 0.7652259275562515 | 0.004651162790697674 | 0.8629426150555287 | 1 |
| Shared hidden-state feasibility | None | None | None | None |

## Simple Speaking Script

We capped each participant at 100 clips. All models use the same participant folds. No participant appears in both training and testing. Zero-shot means the test doctors are unseen. The 3-shot and 5-shot setting simulates a new doctor giving only a few positive VIGIL examples. In this run, we used a safety fallback and did not adapt the heads yet. openWakeWord and Qwen remain frozen. Stage 2 reduces false positives compared with Stage 1 only, but it adds an extra Qwen audio-encoder forward in the current prototype. In the intended medical system, Qwen ASR runs continuously for the transcript, and the VIGIL branch runs in parallel.

## Limitations

- This zero-shot worker uses one development validation fold per outer fold; full nested OOF refit remains the next protocol hardening step.
- Few-shot head adaptation was not claimed because no development-safe recipe has been verified yet.
- Stage 1 and Stage 2 latency were not separately benchmarked in this run.

## Exact Next Research Decision

Decide whether to prioritize a development-safe few-shot head-adaptation recipe or a shared-hidden-state ASR runtime prototype.
