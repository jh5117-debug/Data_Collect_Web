# Codex Handoff: VIGIL Stage1, Target-Doctor Few-Shot, Shared Qwen

## Current Branch And Commit

- Branch: `research/vigil-stage1-fewshot-sharedqwen-20260626`
- Start commit: `f63f688 Optimize latest VIGIL operating point and reports`
- Remote: `origin git@github.com:jh5117-debug/Data_Collect_Web.git`

## Dataset Used

- Latest ZIP: `/home/hj/Data_Collect_Web/finetune/data/vigil_dataset_export_latest_readonly_20260626_042151.zip`
- Raw clips: `1673`
- Valid clips after QC: `1618`
- Silent clips rejected: `55`
- Balanced set: `1346` clips / `1364` windows / `37` participants

## Current Model And Checkpoint Paths

- Latest optimized package: `/home/hj/Data_Collect_Web/finetune/experiments/latest_data_optimization`
- Nested zero-shot run root: `/home/hj/Data_Collect_Web/finetune/experiments/latest_data/runs/nested_zero_shot`
- Latest feature cache: `/home/hj/Data_Collect_Web/finetune/experiments/latest_data/runs/latest_feature_cache_2b78e211183d47fb`
- Final deployment candidate bundle is ignored and not committed: `/home/hj/Data_Collect_Web/finetune/model_bundles/vigil_latest_optimized_20260626_085405`

## Stage 1 Audit Status

- Status: complete.
- Confirmed Stage 1 code path: `finetune/src/vigil_two_stage/stage1_model.py`.
- Confirmed official openWakeWord package is installed in `/home/hj/miniconda/envs/vigil-two-stage`.
- Report: `/home/hj/Data_Collect_Web/finetune/experiments/latest_data_optimization/reports/STAGE1_OPENWAKEWORD_STRUCTURE_REPORT.md`
- Stage 1 head: `LayerNorm -> 2-layer GRU -> Linear`, `56321` trainable parameters.
- Latency from latest compute report: openWakeWord feature extraction median `29.004013165831566` ms; Stage 1 head median `1.1985101737082005` ms.

## Few-Shot Target-Doctor Evaluation Status

- Status: complete.
- Required protocol implemented: target participant only in query; support positives removed from query; target negatives never used for adaptation.
- Script: `/home/hj/Data_Collect_Web/finetune/experiments/latest_data_optimization/scripts/run_target_doctor_fewshot.py`
- Log: `/home/hj/Data_Collect_Web/finetune/experiments/latest_data_optimization/logs/vigil_target_fewshot.log`
- Report: `/home/hj/Data_Collect_Web/finetune/experiments/latest_data_optimization/reports/TARGET_DOCTOR_FEWSHOT_ONBOARDING_REPORT.md`
- 3-shot eligible doctors: `33`; recall `0.9494214876033058`, FPR `0.011666666666666667`, F1 `0.9683074848280513`, delta F1 `0.0423355338250333`, improved/degraded/unchanged `13/2/18`.
- 5-shot eligible doctors: `31`; recall `0.9558441558441558`, FPR `0.011764705882352941`, F1 `0.970976253298153`, delta F1 `0.042333836283642956`, improved/degraded/unchanged `12/2/17`.
- Selected method: development-selected bounded `positive_bias` score-logit calibration. It does not update Qwen, openWakeWord, or target-negative examples.

## LibriSpeech Benchmark Status

- Corrected frozen-Qwen full run expected at: `/home/hj/Data_Collect_Web/finetune/benchmarks/asr/runs/20260624_185419_qwen3_asr_1_7b_fixed_text_extraction_baseline_full`
- Status: verified and integrated.
- Current frozen Qwen WER: test-clean `1.8411%`, test-other `3.6662%`, combined `2.7516%`.
- Report: `/home/hj/Data_Collect_Web/finetune/experiments/latest_data_optimization/reports/CURRENT_QWEN_LIBRISPEECH_BENCHMARK.md`

## Shared-Qwen Status

- Status: `blocked_by_runtime_interface`.
- Report: `/home/hj/Data_Collect_Web/finetune/experiments/latest_data_optimization/reports/SHARED_QWEN_ASR_HIDDEN_STATE_REPORT.md`
- Call-counter diagnostic used one Qwen model instance on one sample. Public `transcribe` returned text but no reusable hidden states; separate Stage 2 feature extraction increased `get_audio_features` count.
- Measured extra Qwen encoder cost from latest compute report: median `13.663365971297026` ms per Stage 1 candidate.

## GPU Assignment

- Avoid GPU `0`; it has unrelated user processes.
- GPU `1`: used for completed target-doctor few-shot run.
- GPU `2`: used for completed shared-Qwen one-sample diagnostic and sanitized rerun.
- GPUs `3`, `4`, `5`, `6` appeared idle after shared-Qwen diagnostic.

## Active Tmux Sessions

- None after target-doctor few-shot completion.
- Previous long-speech full run completed outside tmux with FAPH `30.711918757306318`; artifacts are in ignored run output.

## Test Status

- `PYTHONPATH=finetune/src:. pytest -q finetune/tests`: `56 passed`.
- `PYTHONPATH=finetune/benchmarks/asr/src:finetune/src:. pytest -q finetune/benchmarks/asr/tests`: `17 passed`.
- `PYTHONPATH=finetune/src:finetune/evaluation:. pytest -q finetune/evaluation/tests`: `13 passed`.
- `PYTHONPATH=finetune/src:finetune/experiments/latest_data/src:. pytest -q finetune/experiments/latest_data/tests`: `11 passed`.
- `PYTHONPATH=finetune/src:finetune/experiments/latest_data_optimization/src:. pytest -q finetune/experiments/latest_data_optimization/tests`: `20 passed`.
- `python -m compileall -q finetune/src finetune/scripts finetune/benchmarks/asr finetune/experiments/latest_data finetune/experiments/latest_data_optimization`: passed.
- `find finetune -path '*/scripts/*.sh' -o -path '*/demo/*.sh' | sort | xargs -r bash -n`: passed.

## Artifact Paths

- Handoff: `/home/hj/Data_Collect_Web/docs/CODEX_HANDOFF_VIGIL_STAGE1_FEWSHOT_SHAREDQWEN.md`
- Reports directory: `/home/hj/Data_Collect_Web/finetune/experiments/latest_data_optimization/reports`
- Stage 1 report: `/home/hj/Data_Collect_Web/finetune/experiments/latest_data_optimization/reports/STAGE1_OPENWAKEWORD_STRUCTURE_REPORT.md`
- LibriSpeech report: `/home/hj/Data_Collect_Web/finetune/experiments/latest_data_optimization/reports/CURRENT_QWEN_LIBRISPEECH_BENCHMARK.md`
- Shared-Qwen report: `/home/hj/Data_Collect_Web/finetune/experiments/latest_data_optimization/reports/SHARED_QWEN_ASR_HIDDEN_STATE_REPORT.md`
- Target-doctor few-shot report: `/home/hj/Data_Collect_Web/finetune/experiments/latest_data_optimization/reports/TARGET_DOCTOR_FEWSHOT_ONBOARDING_REPORT.md`
- Post-meeting report: `/home/hj/Data_Collect_Web/finetune/experiments/latest_data_optimization/reports/POST_MEETING_ACTION_REPORT.md`

## Git Push Status

- Ready to commit and push.
- Do not stage existing unrelated changes:
  - `finetune/reports/export_inspection.json`
  - `finetune/reports/export_inspection.md`
  - `docs/VIGIL_Recorder_Participant_Guide.docx`

## Exact Next Command

```bash
cd /home/hj/Data_Collect_Web && git status -sb
```

## Blockers

- None yet.
