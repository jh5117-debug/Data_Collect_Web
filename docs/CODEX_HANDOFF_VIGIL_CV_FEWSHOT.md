# Codex Handoff: VIGIL Participant CV And Few-Shot

## Current State

- Repository: `/home/hj/Data_Collect_Web`
- Branch: `research/vigil-participant-cv-fewshot-20260624`
- Base commit: `4b89904`
- Dataset fingerprint: `0fad4c7828149099`
- Processed dataset: `/home/hj/Data_Collect_Web/finetune/data/processed/0fad4c7828149099`
- Dataset ZIP: `/home/hj/Data_Collect_Web/finetune/data/vigil_dataset_export_20260624_072023_180681.zip`
- Completed two-stage run: `/home/hj/Data_Collect_Web/finetune/runs/20260624_075127_0fad4c7828149099_full`

## Completed Phases

- Read `docs/CODEX_HANDOFF_VIGIL_NEXT.md`.
- Verified source branch was `research/vigil-eval-live-demo-20260624` at `4b89904`.
- Fetched `origin`.
- Created branch `research/vigil-participant-cv-fewshot-20260624`.
- Verified `vigil_live_demo` belonged to user `hj` and ran `/home/hj/Data_Collect_Web/finetune/demo/app.py`.
- Stopped `vigil_live_demo` to release GPU 6.
- Created participant CV package scaffold under `finetune/experiments/participant_cv`.
- Completed participant audit: 1298 unique clips, 30 participants, 21 duplicate audio groups, 0 cross-participant duplicate groups.
- Built deterministic balanced max-100 manifest: 1026 clips, 1040 windows, SHA-256 `44815508a013b9022a8efc99a3972b6847884ebbb3578e356f18a50b822f5a03`.
- Built shared participant-level five-fold definition: SHA-256 `e6759ee22e4358c2d7f4a3578b8568d6eb829ab7cfad69c4441cdc11b57d01cb`.
- Protocol validation passed: 30 aliases, 5 folds, no participant leakage, no duplicate audio hash crosses folds, max 100 clips per participant.
- Feature coverage passed: 1040/1040 balanced windows covered by official openWakeWord and frozen Qwen encoder caches; no NaN/Inf in sampled features.
- Participant CV protocol tests passed: 34 tests.
- Completed corrected Qwen transcript cache: 1026/1026 clips, extraction path `$[0].text`, result type `qwen_asr.inference.qwen3_asr.ASRTranscription`, model revision `7278e1e70fe206f11671096ffdd38061171dd6e5`.
- Completed participant-disjoint zero-shot 5-fold worker run for Qwen exact, Stage1-only, Stage2 BCE, Stage2 BCE+SupCon, and validation-selected method.
- Aggregated zero-shot results in `ZERO_SHOT_5FOLD_REPORT.md`.
- Completed stage error analysis for `stage2_bce`.
- Completed strict support/query few-shot evaluation with conservative `no_adaptation_zero_shot_fallback`; no few-shot improvement is claimed.
- Generated compute/accuracy, shared-Qwen feasibility, continuous-ASR architecture, balanced-vs-full ablation status, and professor meeting reports.
- Final tests passed:
  - `finetune/tests`: 56 passed.
  - `finetune/evaluation/tests`: 13 passed.
  - `finetune/benchmarks/asr/tests`: 17 passed.
  - `finetune/experiments/participant_cv/tests`: 34 passed.
  - `compileall`: passed.
  - new script py_compile and shell syntax checks: passed.

## Current Phase

- Ready to commit/push sanitized source, configs, shared fold definitions, summaries, and reports.

## Generated Split Checksum

- Balanced manifest SHA-256: `44815508a013b9022a8efc99a3972b6847884ebbb3578e356f18a50b822f5a03`.
- Five-fold JSON SHA-256: `e6759ee22e4358c2d7f4a3578b8568d6eb829ab7cfad69c4441cdc11b57d01cb`.

## Selected GPUs

- Idle candidates after stopping demo: physical GPUs `1`, `2`, `3`, `4`, `6`.
- GPU `6` was used for `vigil_qwen_cache` and fold 4; released after completion.
- GPUs `1`, `2`, `3`, `4`, `6` were used for zero-shot fold jobs.
- Do not use GPU `0` or GPU `5` unless a later audit proves they are idle.

## Active tmux Sessions

- None.

## Fold Progress

- Fold 0: completed.
- Fold 1: completed.
- Fold 2: completed.
- Fold 3: completed.
- Fold 4: completed.

## Failed Jobs

- None.

## Exact Next Command

```bash
cd /home/hj/Data_Collect_Web && git status --short
```

## Artifact Paths

- Handoff: `/home/hj/Data_Collect_Web/docs/CODEX_HANDOFF_VIGIL_CV_FEWSHOT.md`
- Experiment package: `/home/hj/Data_Collect_Web/finetune/experiments/participant_cv`
- Reports: `/home/hj/Data_Collect_Web/finetune/experiments/participant_cv/reports`
- Shared sanitized artifacts: `/home/hj/Data_Collect_Web/finetune/experiments/participant_cv/shared`
- Private generated artifacts: ignored under `runs/`, `logs/`, and `shared/private`
- Participant audit: `/home/hj/Data_Collect_Web/finetune/experiments/participant_cv/reports/PARTICIPANT_DATA_AUDIT.md`
- Duplicate audit: `/home/hj/Data_Collect_Web/finetune/experiments/participant_cv/reports/DUPLICATE_AUDIO_AUDIT.md`
- Balanced report: `/home/hj/Data_Collect_Web/finetune/experiments/participant_cv/reports/BALANCED_MAX100_REPORT.md`
- Fold report: `/home/hj/Data_Collect_Web/finetune/experiments/participant_cv/reports/FOLD_BALANCE_REPORT.md`
- Feature coverage: `/home/hj/Data_Collect_Web/finetune/experiments/participant_cv/reports/FEATURE_COVERAGE_REPORT.md`
- Transcript cache: `/home/hj/Data_Collect_Web/finetune/experiments/participant_cv/shared/qwen_transcript_cache_balanced_max100.jsonl`
- Zero-shot summary: `/home/hj/Data_Collect_Web/finetune/experiments/participant_cv/reports/zero_shot_summary.json`
- Zero-shot report: `/home/hj/Data_Collect_Web/finetune/experiments/participant_cv/reports/ZERO_SHOT_5FOLD_REPORT.md`
- Stage error report: `/home/hj/Data_Collect_Web/finetune/experiments/participant_cv/reports/STAGE_ERROR_ANALYSIS.md`
- Few-shot report: `/home/hj/Data_Collect_Web/finetune/experiments/participant_cv/reports/FEW_SHOT_ONBOARDING_REPORT.md`
- Compute report: `/home/hj/Data_Collect_Web/finetune/experiments/participant_cv/reports/COMPUTE_ACCURACY_TRADEOFF.md`
- Professor summary: `/home/hj/Data_Collect_Web/finetune/experiments/participant_cv/reports/PROFESSOR_MEETING_SUMMARY.md`

## Git Status

- Existing unrelated files remain unstaged and must not be committed:
  - `finetune/reports/export_inspection.json`
  - `finetune/reports/export_inspection.md`
  - `docs/VIGIL_Recorder_Participant_Guide.docx`

## Push Status

- New branch not pushed yet.
