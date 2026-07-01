# VIGIL Experiment Results

## VIGIL Trigger Metrics

| Method | Recall | FPR | Precision | F1 |
|---|---:|---:|---:|---:|
| Base Qwen exact keyword | 0.6189 | 0.0000 | 1.0000 | 0.7514 |
| Optimized two-stage | 0.9409 | 0.0050 | 0.9957 | 0.9675 |

## LibriSpeech ASR Preservation

These values are from the corrected Qwen transcript extraction benchmark. The old roughly 40% WER result is invalid and must not be cited as valid.

| Split | Corrected WER |
|---|---:|
| test-clean | 1.8411% |
| test-other | 3.6662% |
| combined | 2.7516% |

## Stage 1 Parameter And Latency

| Component | Value |
|---|---:|
| Feature extractor | frozen openWakeWord |
| Trainable head | LayerNorm -> 2-layer GRU -> Linear |
| Trainable parameters | 56,321 |
| Full Stage 1 median latency | about 30.203 ms |

## Few-Shot Ablation

| Method | Shot | F1 | Recall | FPR | Delta F1 |
|---|---:|---:|---:|---:|---:|
| zero_shot | 5 | 0.92864 | 0.86679 | 0.00000 | 0.00000 |
| stage2_positive_bias | 5 | 0.96997 | 0.95288 | 0.01076 | +0.04133 |
| stage2_cosine_prototype | 5 | 0.97059 | 0.95510 | 0.01176 | +0.04195 |
| stage2_finetune_bias_only | 5 | 0.92864 | 0.86679 | 0.00000 | 0.00000 |
| stage2_finetune_head | 5 | 0.92864 | 0.86679 | 0.00000 | 0.00000 |
| stage1_finetune_bias_only | 5 | 0.92864 | 0.86679 | 0.00000 | 0.00000 |
| stage1_finetune_head | 5 | 0.92864 | 0.86679 | 0.00000 | 0.00000 |

The best measured method is 5-shot `stage2_cosine_prototype`.

## Shared-Qwen Cost And Blocker

The same frozen Qwen weights are used conceptually, but shared hidden-state reuse is blocked by the current public `qwen_asr` runtime interface. Current Stage 2 still needs an extra Qwen encoder forward.

## Demo Status

The browser assistant demo can run locally on HAL. It supports onboarding, prototype calibration, rolling transcript, and VIGIL trigger state. It does not implement downstream LLM/VQA responses.

## Primary Report Paths

- `finetune/reports/ASR_PRESERVATION_REPORT.md`
- `finetune/reports/QWEN_TRANSCRIPT_EXTRACTION_FIX_REPORT.md`
- `finetune/experiments/latest_data_optimization/reports/LATEST_OPT_PROFESSOR_MEETING_REPORT.md`
- `finetune/experiments/latest_data_optimization/reports/TARGET_DOCTOR_FEWSHOT_ONBOARDING_REPORT.md`
- `finetune/experiments/fewshot_ablation/reports/FEWSHOT_ABLATION_FINAL_REPORT.md`
- `finetune/experiments/shared_qwen_runtime/reports/FINAL_SHARED_QWEN_RUNTIME_REPORT.md`
- `finetune/demo_live_assistant/reports/WEB_ASSISTANT_DEMO_REPORT.md`
