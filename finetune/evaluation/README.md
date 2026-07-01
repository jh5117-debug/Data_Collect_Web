# VIGIL Evaluation Workflow

This directory separates window-level and clip-level evaluation for the VIGIL two-stage detector.

Run the current development audit without retraining:

```bash
PATH=/home/hj/miniconda/envs/vigil-two-stage/bin:$PATH \
PYTHONPATH=finetune/src:finetune/evaluation:. \
python finetune/evaluation/audit_existing_run.py \
  --dataset-dir /home/hj/Data_Collect_Web/finetune/data/processed/0fad4c7828149099 \
  --run-dir /home/hj/Data_Collect_Web/finetune/runs/20260624_075127_0fad4c7828149099_full
```

Outputs are written under the run directory:

- `evaluation_audit.json`
- `EVALUATION_AUDIT.md`
- `evaluation/window_clip_metrics.json`
- `model_selection.json`
- `MODEL_SELECTION.md`
- `baseline_qwen_exact_clip/`

Definitions:

- Window-level metrics count each two-second training/evaluation window.
- Clip-level metrics count each original submitted recording once.
- Stage 1 clip score is the maximum Stage 1 score over windows from that clip.
- Cascade clip trigger is true when a top-K Stage 1 candidate window satisfies both Stage 1 and Stage 2 thresholds on that same window.
- Model selection uses validation clip-level cascade metrics only. Test metrics are reported only after selection.

The current 1298-clip run is a development evaluation because its test result has already been inspected.

