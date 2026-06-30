# Few-Shot Ablation Final Report

This target-doctor-only ablation answers whether the few-shot effect is Stage 1 or Stage 2, whether Stage 2 cosine prototypes help, and whether simple few-shot fine-tuning helps. It uses the existing leave-one-target-doctor base predictions and frozen cached representations; no Qwen weights, openWakeWord feature extractor weights, audio, checkpoints, or private metadata are written.

## Protocol Check

- Support seeds: `[20260620, 20260621, 20260622, 20260623, 20260624]`
- Eligible 3-shot doctors: `33`
- Eligible 5-shot doctors: `31`
- Base training excludes the target doctor because the source rows are the existing leave-one-target-doctor base rows.
- Support uses target positive clips only.
- Support clips are removed from query.
- Query contains only the target doctor.
- Target negatives are query-only and are never used for adaptation.
- Method and hyperparameter selection uses development pseudo-target doctors only, never the held-out target doctor's query labels.

## Direct Answers

1. Our cosine method uses Stage 2 embeddings, not Stage 1.
2. Stage 1 remains the high-recall candidate detector.
3. Stage 2 is the main location for doctor-specific similarity and adaptation.
4. Best selected method: `stage2_cosine_prototype` at `5`-shot.
5. Simple fine-tuning is compared against cosine and positive-bias calibration below.
6. Stage 1 fine-tuning is treated as an ablation and must pass FPR safety before being useful.
7. Stage 2 fine-tuning updates only small cached-representation adapters in this experiment; Qwen remains frozen.
8. Train/test separation is strict at the target-doctor split and support/query pairing level.

## Metrics Table

| Shot | Method | F1 | Recall | FPR | Delta F1 | Delta Recall | Delta FPR | Improved | Degraded | Safety pass rate | Changed stage |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 3 | zero_shot | 0.925971951003018 | 0.8621487603305785 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 1.0 | none |
| 3 | stage2_positive_bias | 0.9679054054054054 | 0.947107438016529 | 0.01 | 0.04193345440238738 | 0.08495867768595045 | 0.01 | 12 | 2 | 0.9393939393939394 | stage2 |
| 3 | stage2_cosine_prototype | 0.9635901778154106 | 0.9404958677685951 | 0.011666666666666667 | 0.037618226812392574 | 0.07834710743801654 | 0.011666666666666667 | 12 | 2 | 0.9393939393939394 | stage2 |
| 3 | stage2_finetune_bias_only | 0.925971951003018 | 0.8621487603305785 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 1.0 | stage2 |
| 3 | stage2_finetune_head | 0.925971951003018 | 0.8621487603305785 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 1.0 | stage2 |
| 3 | stage1_finetune_bias_only | 0.925971951003018 | 0.8621487603305785 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 1.0 | stage1 |
| 3 | stage1_finetune_head | 0.925971951003018 | 0.8621487603305785 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 1.0 | stage1 |
| 3 | stage1_stage2_combined | 0.925971951003018 | 0.8621487603305785 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 1.0 | stage1+stage2 |
| 5 | zero_shot | 0.92864241701451 | 0.8667903525046382 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 1.0 | none |
| 5 | stage2_positive_bias | 0.9699716713881019 | 0.9528756957328386 | 0.010756302521008404 | 0.04132925437359192 | 0.08608534322820038 | 0.010756302521008404 | 11 | 2 | 0.9032258064516129 | stage2 |
| 5 | stage2_cosine_prototype | 0.9705882352941176 | 0.9551020408163265 | 0.011764705882352941 | 0.04194581827960764 | 0.08831168831168834 | 0.011764705882352941 | 11 | 2 | 0.9354838709677419 | stage2 |
| 5 | stage2_finetune_bias_only | 0.92864241701451 | 0.8667903525046382 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 1.0 | stage2 |
| 5 | stage2_finetune_head | 0.92864241701451 | 0.8667903525046382 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 1.0 | stage2 |
| 5 | stage1_finetune_bias_only | 0.92864241701451 | 0.8667903525046382 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 1.0 | stage1 |
| 5 | stage1_finetune_head | 0.92864241701451 | 0.8667903525046382 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 1.0 | stage1 |
| 5 | stage1_stage2_combined | 0.92864241701451 | 0.8667903525046382 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 1.0 | stage1+stage2 |

## Interpretation

- Stage 2 cosine result: 3-shot cosine delta F1 0.037618226812392574 vs positive-bias delta F1 0.04193345440238738; 5-shot cosine delta F1 0.04194581827960764 vs positive-bias delta F1 0.04132925437359192
- Stage 2 fine-tuning result: best Stage 2 fine-tune method `stage2_finetune_bias_only` at 3-shot delta F1 0.0
- Stage 1 fine-tuning result: best Stage 1 fine-tune method `stage1_finetune_bias_only` at 3-shot delta F1 0.0
- Best method: `stage2_cosine_prototype` at 5-shot with delta F1 0.04194581827960764, recall 0.9551020408163265, FPR 0.011764705882352941

## Limitations

- This run uses cached leave-one-target-out base predictions and cached frozen representations. Fine-tuning methods are lightweight bias/linear/head-style adapters on those cached representations, not a full checkpoint-producing training run.
- The source replay rows are non-target doctors only, but they come from sanitized cached leave-one-target-out rows so that no Qwen/openWakeWord feature extraction is repeated.
- No target negative clips are used for adaptation.

## Selected Recipes

Selection JSON is recorded in `fewshot_ablation_summary.json` under `selected_by_method`.
