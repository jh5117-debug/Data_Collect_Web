# Final Professor Meeting Report

## Summary

We first corrected the evaluation protocol. The old few-shot table used a no-adaptation fallback, so it was not learned personalization.
The strict nested participant-CV result strengthens the Stage 2 conclusion: Stage 2 keeps high recall while reducing the Stage 1 false-positive rate.
The real prototype onboarding path is implemented and selected with development pseudo-targets, but the outer-test paired result is negative: 3-shot and 5-shot did not improve F1.

## Strict Nested CV

Status: `ok`

| Method | Recall | FPR | Precision | F1 |
|---|---:|---:|---:|---:|
| qwen_exact | 0.546406 | 0.000000 | 1.000000 | 0.678914 |
| stage1_only | 0.950890 | 0.093139 | 0.936792 | 0.942661 |
| stage2_bce | 0.945590 | 0.037708 | 0.973827 | 0.958948 |
| stage2_bce_supcon | 0.947442 | 0.016334 | 0.988127 | 0.967011 |
| validation_selected | 0.945590 | 0.016334 | 0.988127 | 0.966045 |

## Stage 2 Operating Point

See `STAGE2_OPERATING_POINT_REPORT.md`. Threshold targets were selected from development OOF predictions only.

## Real 0/3/5-Shot Onboarding

Status: `ok`
Learned personalization claimed: `True`

| Condition | Recall | FPR | Precision | F1 |
|---|---:|---:|---:|---:|
| 0-shot on 3-shot query | 0.9564154786150713 | 0.016129032258064516 | 0.985312631137222 | 0.9706490285241836 |
| 3-shot prototype | 0.9462321792260693 | 0.016129032258064516 | 0.9851569126378287 | 0.9653023062538957 |
| 0-shot on 5-shot query | 0.9599088838268792 | 0.016129032258064516 | 0.9836601307189542 | 0.9716393820613326 |
| 5-shot prototype | 0.9535307517084283 | 0.016129032258064516 | 0.9835526315789473 | 0.9683090446449225 |

3-shot and 5-shot support recordings changed model outputs, but the paired F1 deltas were negative. This is a safe negative result rather than an onboarding win.

## Accuracy-Cost

Status: `partial_head_benchmark`

| Component | Median latency | P95 latency | Peak allocated VRAM |
|---|---:|---:|---:|
| Stage 1 head | 0.0009344830177724361 | 0.0009559532627463341 | 0.012787818908691406 GB |
| Stage 2 head | 0.0004961066879332066 | 0.000510866753757 | 0.013950824737548828 GB |

Current System C accounting: one loaded Qwen instance, one continuous ASR forward, plus one additional audio-encoder forward for each Stage 2 candidate.

## Shared-Qwen

Status: `blocked_by_runtime_interface`
Blocker: public wrapper returns transcript objects but does not expose reusable hidden-state handoff

## Balanced Versus Full Data

Status: `not_run`. Full unbalanced nested ablation was not run, so no full-data result is claimed.

## Long-Speech False Activations

Status: `blocked_until_final_model_locked`. No false-activations-per-hour result is claimed.

## Future Blind-Test Protocol

Status: lock fields present = `True`. The lock is a protocol scaffold until final thresholds/checkpoints are frozen.

## Limitations

- Full unbalanced-data ablation is pending.
- Complete Qwen ASR/audio-encoder/cascade latency benchmark is pending.
- Gradient adaptation search is guarded in code but not fully run.
- Long-speech false-activation stress test waits for a locked final model.
- The final deployment candidate is not trained because the research choices are not fully frozen.

## Simple Speaking Script

We first corrected the evaluation protocol. The old few-shot table used a no-adaptation fallback. We now separate development selection from outer-test reporting, and we do not claim personalization unless support recordings change the model output.
