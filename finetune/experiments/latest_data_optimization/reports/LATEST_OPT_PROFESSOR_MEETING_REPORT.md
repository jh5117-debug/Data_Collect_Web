# Latest Optimized VIGIL Professor Meeting Report

## English Summary

- Latest export: 1673 raw clips; 1618 valid clips after audio QC; 55 silent clips rejected.
- Formal balanced max-100 set: 1346 clips, 1364 windows, 37 participants.
- Split policy: participant-level five-fold CV; the same participant never appears in both train/development and outer test.
- Qwen transcript keyword baseline: F1 0.7514, recall 0.6189, FPR 0.0000.
- Optimized two-stage detector: F1 0.9675, recall 0.9409, FPR 0.0050, precision 0.9957.
- Stage1-only: F1 0.9563, recall 0.9556, FPR 0.0532.
- Selected Stage2 config: stage2_bce_supcon, top_k=1, threshold-only, selected on development predictions only.
- Stage errors after optimization: {'STAGE1_FALSE_CANDIDATE_REJECTED': 29, 'STAGE1_MISS': 33, 'STAGE2_FALSE_ACCEPT': 3, 'STAGE2_REJECT': 11}.
- Real few-shot onboarding: Real support-based onboarding was implemented and evaluated, but no safe improvement was found on the latest dataset.
- 3-shot paired F1: 0.9730; 5-shot paired F1: 0.9758.
- Compute: Stage2 head median 1.4751 ms; extra Qwen encoder forward median 13.6634 ms on 20 measured samples.
- Long normal speech: subset LibriSpeech run, 0.0814 h, 0 false accepts, FAPH 0.0000. Full run is tracked separately if tmux is active.
- Balanced-vs-full ablation: partial_full_unbalanced_heads_only; full-unbalanced selected Stage2 F1 0.9699, FPR 0.0072.
- Shared Qwen status: blocked_by_runtime_interface because the public wrapper does not expose reusable ASR hidden states.
- Final deployment candidate: trained_deployment_candidate_not_scientific_test; Qwen weights are not included in the bundle.

## Simple Chinese Summary

- 最新数据有 1673 条原始 clips；音频质检后 1618 条有效 clips，剔除了 55 条静音。
- 正式 balanced max-100 数据集有 1346 clips、37 位参与者。
- 我们使用 participant-level split，所以同一个人不会同时出现在训练和测试里。
- Qwen 文字关键词 baseline 的 F1 是 0.7514；优化后的 two-stage detector F1 是 0.9675，recall 是 0.9409，FPR 是 0.0050。
- Stage2 把 Stage1-only 的 FPR 从 0.0532 降到 0.0050，同时 F1 从 0.9563 提高到 0.9675。
- 3-shot/5-shot 个性化已经真实实现并评估，但结果是：Real support-based onboarding was implemented and evaluated, but no safe improvement was found on the latest dataset.
- 计算成本方面，Stage2 head 中位数 1.4751 ms，额外 Qwen encoder forward 中位数 13.6634 ms。
- 普通长语音 false wake-up subset 结果是 0.0000 次/小时；完整 LibriSpeech run 若仍在跑，需要等 tmux 完成后更新。
- shared-Qwen 目前被 runtime interface 阻塞；需要 qwen_asr 暴露可复用 hidden states。

## Speaking Script

On the latest dataset, we have 1618 valid clips from 37 balanced participants. We use participant-level folds, so the same person never appears in both training and testing. The Qwen transcript baseline gets F1 0.7514. Our selected two-stage detector gets F1 0.9675, recall 0.9409, and FPR 0.0050. Stage2 reduces false triggers from 0.0532 to 0.0050, but adds about 13.6634 ms for the extra Qwen encoder forward plus 1.4751 ms for the verifier head. For 3-shot/5-shot onboarding, we tried support-based personalization; it did not improve paired F1 safely. Long-speech false accepts per hour is 0.0000 on the current subset. The best next step is to let the full long-speech job finish and then run a future blind export against the trained deployment candidate.

## Limitations

- Long-speech report committed here is a subset, not the full LibriSpeech test-clean/test-other stress test.
- Full-unbalanced ablation is heads-only because full-unbalanced Qwen transcript cache was not generated.
- Final model bundle is for deployment/future blind test only, not a new held-out scientific score.

## Current Frozen-Qwen LibriSpeech Benchmark

LibriSpeech has been completed for the frozen base Qwen branch. It is not a new fine-tuned Qwen benchmark because Qwen is not fine-tuned in the current method.

- test-clean WER: 1.8411%
- test-other WER: 3.6662%
- combined normalized WER: 2.7516%
- successes/failures: `5559` / `0`
- text extraction path: `['$[0].text']`
