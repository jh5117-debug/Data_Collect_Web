# Target-Doctor Few-Shot Ablation

This experiment answers the professor's few-shot questions under the strict target-doctor-only protocol.

## Protocol

- Base rows come from the existing leave-one-target-doctor run.
- For each target doctor, support clips are positive VIGIL clips from that same doctor only.
- Support clips are removed from the query set.
- Query contains only the target doctor.
- Target negatives are query-only and are never used for adaptation.
- Development pseudo-target doctors are used for method and hyperparameter selection.
- The held-out target doctor's query labels are not used for selection.

## Compared Methods

- `zero_shot`: no support.
- `stage2_cosine_prototype`: frozen Qwen audio features plus Stage 2 embedding head. This is the main cosine method and it uses Stage 2 embeddings, not Stage 1.
- `stage2_positive_bias`: bounded Stage 2 score-logit positive bias from target positive support.
- `stage2_finetune_bias_only`: simple bias-only adapter over frozen Stage 2 logits.
- `stage2_finetune_head`: simple linear adapter over frozen Stage 2 embeddings.
- `stage1_finetune_bias_only`: simple bias-only adapter over frozen Stage 1 scores.
- `stage1_finetune_head`: small score-head-style Stage 1 adapter.
- `stage1_stage2_combined`: combines the best safe Stage 1 and Stage 2 adaptations.

The fine-tuning ablations use cached frozen representations and do not save checkpoints. They are intentionally lightweight simple few-shot adaptation tests, not Qwen or openWakeWord retraining.

## Run

```bash
cd /home/hj/Data_Collect_Web
PYTHONPATH=finetune/src:finetune/experiments/fewshot_ablation/src:. \
python finetune/experiments/fewshot_ablation/scripts/run_target_doctor_fewshot_ablation.py
```

Aggregate check:

```bash
PYTHONPATH=finetune/src:finetune/experiments/fewshot_ablation/src:. \
python finetune/experiments/fewshot_ablation/scripts/aggregate_target_doctor_fewshot_ablation.py
```

## Outputs

- `reports/FEWSHOT_ABLATION_FINAL_REPORT.md`
- `reports/fewshot_ablation_summary.json`
- `reports/fewshot_ablation_per_doctor.csv`
- `reports/fewshot_ablation_per_method.csv`
- `reports/fewshot_ablation_support_seed_results.csv`

No audio, features, checkpoints, predictions, private metadata, or model weights should be committed.
