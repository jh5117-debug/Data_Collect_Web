# Latest Professor Meeting Report

## Latest Data Counts

- Production raw clips: `1673` (946 positive, 727 negative).
- Accounts shown by Admin: `39`; submitted sessions: `37` / `38`.
- ZIP source: read-only Admin fallback export because production export job failed at 433/1673 with read timeout.
- ZIP SHA-256: `e2e38518d6725449653138e0ee484c4b5903467e418e8968d4b98ada5fd41701`.
- Valid unique clips after audio QC: `1618`; manifest windows: `1636`.
- Rejected silent clips: `55` (`C001506` to `C001560`).
- Prompt groups: P1 `173`, P2 `472`, P3 `301`, P4 `727`.

## Balanced Max-100 Dataset

- Balanced clips: `1346`; windows: `1364`; participants: `37`.
- Balanced manifest SHA-256: `549134e307f21470cb942acd44c2c27d2b29fcaa8527b9e7f8e2722e3232b58e`.

## Five-Fold Table

| Fold | Participants | Clips | Pos | Neg | P1 | P2 | P3 | P4 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 7 | 262 | 150 | 112 | 44 | 68 | 38 | 112 |
| 1 | 8 | 277 | 154 | 123 | 21 | 78 | 55 | 123 |
| 2 | 7 | 264 | 151 | 113 | 27 | 74 | 50 | 113 |
| 3 | 7 | 270 | 146 | 124 | 27 | 77 | 42 | 124 |
| 4 | 8 | 273 | 143 | 130 | 29 | 68 | 46 | 130 |

## Nested Zero-Shot Table

| Method | Recall | FPR | Precision | F1 |
|---|---:|---:|---:|---:|
| qwen_exact | 0.618909 | 0.000000 | 1.000000 | 0.751427 |
| stage1_only | 0.955789 | 0.052846 | 0.957336 | 0.956367 |
| stage2_bce | 0.880851 | 0.005357 | 0.995489 | 0.934038 |
| stage2_bce_supcon | 0.884737 | 0.000000 | 1.000000 | 0.936468 |
| validation_selected | 0.850184 | 0.000000 | 1.000000 | 0.916830 |

## Stage 2 Operating Point Table

- Status: `requires_inner_oof_predictions`; no development-only operating point is claimed yet.

## Real 0/3/5-Shot Onboarding Table

- Selected recipe: `no_adaptation_zero_shot_fallback`.
- Safe support-using improvement found: `False`.
- The table uses the no-adaptation fallback, so 3-shot/5-shot differences are not adaptation-improvement claims.

| Condition | Recall | FPR | Precision | F1 |
|---|---:|---:|---:|---:|
| 0-shot | 0.865332 | 0.004687 | 0.991363 | 0.922570 |
| 3-shot | 0.850531 | 0.004545 | 0.992308 | 0.917514 |
| 5-shot | 0.881087 | 0.004839 | 0.990374 | 0.927857 |

## Stage Error Analysis

- Validation-selected errors: 33 Stage 1 misses, 79 Stage 2 rejects, 32 Stage 1 false candidates.

## Accuracy-Cost Table

- Status: `partial_head_benchmark`; full Qwen forward latency is still a limitation.
| Component | Median ms | P95 ms | Peak GB |
|---|---:|---:|---:|
| stage1_head | 0.845493 | 1.074642 | 0.012577 |
| stage2_head | 0.456376 | 0.483669 | 0.013739 |

## Shared-Qwen Status

- Status: `blocked_by_runtime_interface`.
- Blocker: public wrapper returns transcript objects but does not expose reusable hidden-state handoff

## Long-Speech False Accepts/Hour

- Status: `blocked_until_final_model_locked`. No false-accepts/hour result is claimed.

## Final Model Status

- Status: `not_trained_choices_not_frozen`. Choices are not frozen, so no deployment bundle was trained.

## Blind-Test Protocol

- Status: `protocol_ready_lock_pending_final_model`. Future exports must reject known development participants and cannot tune thresholds.

## Known Limitations

- Production background export failed; latest ZIP was reconstructed through read-only Admin APIs.
- 55 raw clips were silent and excluded from training/evaluation manifests.
- Stage 2 operating-point search still needs inner OOF predictions.
- Few-shot support-using adaptation was not safely proven on this latest run.
- Full Qwen/cascade latency and long-speech false accepts/hour were not completed.

## Speaking Script

English:

- The latest collection has 1673 raw clips. After audio QC, 55 silent clips were excluded.
- We capped each participant at 100 clips, leaving 1346 balanced clips.
- No participant appears in both train and test folds.
- The participant-level five-fold zero-shot result is the strict unseen-participant estimate.
- The validation-selected two-stage method reached recall 0.850 with FPR 0.000.
- Stage1 alone had higher recall 0.956, but higher FPR 0.053.
- We do not yet have a safe support-using few-shot improvement.
- We still need inner operating-point search, full latency, long-speech stress testing, and future blind-test data.

Chinese:

- 最新收集有 1673 条原始 clips，其中 55 条是静音，已经从训练/评估 manifest 里剔除。
- 每个参与者最多保留 100 条，平衡后是 1346 条 clips。
- 五折是按参与者切分的，同一个人不会同时出现在训练和测试里。
- validation-selected 两阶段方法的 recall 是 0.850，FPR 是 0.000。
- 目前还不能声称 few-shot adaptation 有安全收益。
- 下一步需要补 inner operating point、完整延迟、长语音误触发和未来盲测。

## Next Decision

Decide whether deployment prioritizes zero FPR via Stage2 or higher recall via Stage1, then run inner OOF operating-point search before locking a final model.
