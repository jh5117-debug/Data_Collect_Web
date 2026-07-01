# Codex Handoff: VIGIL Latest Data Final

## Current State

- Repository: `/home/hj/Data_Collect_Web`
- Branch: `research/vigil-latest-data-final-20260626`
- Start commit: `4b36f79`
- Latest code changes are uncommitted.
- Production backend: `https://data-collect-web.onrender.com`
- Production frontend: `https://data-collect-web.vercel.app/admin`

## Latest Export

- Production Admin summary before export: 1673 clips, 946 positive, 727 negative, 39 accounts, 38 sessions, 37 submitted sessions.
- Background export job `27ced875-c9bf-4083-a33e-5e0652310ba3` failed at 433/1673 with `The read operation timed out`.
- Completed fallback export: read-only Admin API reconstruction.
- Latest ZIP: `/home/hj/Data_Collect_Web/finetune/data/vigil_dataset_export_latest_readonly_20260626_042151.zip`
- ZIP SHA-256: `e2e38518d6725449653138e0ee484c4b5903467e418e8968d4b98ada5fd41701`
- ZIP integrity: ok.
- Export inspection: 1673 canonical samples, 0 metadata rejections.

## Latest Processed Dataset

- Processed dataset: `/home/hj/Data_Collect_Web/finetune/data/processed/2b78e211183d47fb`
- Dataset fingerprint: `2b78e211183d47fb`
- Manifest windows: 1636.
- Unique valid clips before duplicate policy: 1618.
- Audio QC rejected: 55 silent clips (`C001506` to `C001560`), all `audio_conversion_or_validation_failed` with zero peak/RMS.
- Prompt counts after audio QC: P1 171, P2 460, P3 293, P4 712.
- Qwen ASR/KWS format report: `/home/hj/Data_Collect_Web/finetune/data/processed/2b78e211183d47fb/QWEN_ASR_FORMAT_REPORT.md`
- Qwen ASR manifests contain only `audio` and `text`; KWS manifests contain labels separately.

## Balanced And Folds

- Balanced max-100 manifest: `/home/hj/Data_Collect_Web/finetune/experiments/latest_data/shared/balanced_max100_latest_manifest.jsonl`
- Balanced clips/windows: 1346 clips, 1364 windows.
- Balanced participants: 37.
- Balanced manifest SHA-256: `549134e307f21470cb942acd44c2c27d2b29fcaa8527b9e7f8e2722e3232b58e`
- Fold definition: `/home/hj/Data_Collect_Web/finetune/experiments/latest_data/shared/latest_participant_folds_5fold.json`
- Fold SHA-256: `7c1c65da28f87922f111ee1549b61c053323fc876d2cd26346544de0b37b2a5e`
- Leakage validation: ok; no participant crosses folds and no duplicate audio hash crosses folds.

## Feature And Transcript Cache

- Feature run dir: `/home/hj/Data_Collect_Web/finetune/experiments/latest_data/runs/latest_feature_cache_2b78e211183d47fb`
- Feature coverage: 1364/1364 openWakeWord, 1364/1364 Qwen encoder.
- Stage 1 backend: official openWakeWord.
- Qwen feature backend: frozen Qwen audio encoder.
- Qwen transcript cache: `/home/hj/Data_Collect_Web/finetune/experiments/latest_data/shared/qwen_transcript_cache_balanced_max100_latest.jsonl`
- Qwen transcript cache: 1346/1346, extraction path `$[0].text`, result type `qwen_asr.inference.qwen3_asr.ASRTranscription`.

## Nested Zero-Shot

- Run root: `/home/hj/Data_Collect_Web/finetune/experiments/latest_data/runs/nested_zero_shot`
- Folds completed: 5/5.
- Qwen exact: recall 0.618908887779794, FPR 0.0, precision 1.0, F1 0.7514269798385881.
- Stage1 only: recall 0.9557893520190536, FPR 0.052845756718743635, precision 0.9573360226352112, F1 0.9563671267995225.
- Stage2 BCE: recall 0.8808507002991264, FPR 0.005357142857142857, precision 0.9954887218045113, F1 0.9340384955935721.
- Stage2 BCE + SupCon: recall 0.8847373511305291, FPR 0.0, precision 1.0, F1 0.9364677067795163.
- Validation-selected: recall 0.8501840336324596, FPR 0.0, precision 1.0, F1 0.9168298775352477.

## Few-Shot And Errors

- Few-shot selected recipe: `no_adaptation_zero_shot_fallback`.
- Safe support-using improvement found: false.
- Paired no-adaptation table:
  - 0-shot: recall 0.8653316231711301, FPR 0.0046875, precision 0.9913626309792444, F1 0.9225699943498468.
  - 3-shot: recall 0.8505312779196801, FPR 0.004545454545454545, precision 0.9923076923076923, F1 0.9175142316828513.
  - 5-shot: recall 0.8810868294065446, FPR 0.004838709677419355, precision 0.9903743315508021, F1 0.9278570664199073.
- Do not claim few-shot adaptation improvement from that table; support samples do not change the model under the selected fallback.
- Stage error analysis for validation-selected: 33 Stage1 misses, 79 Stage2 rejects, 32 Stage1 false candidates.

## Other Reports

- Stage2 operating points: `requires_inner_oof_predictions`; no development-only operating point is claimed.
- Balanced-vs-full ablation: not run.
- Compute cost: partial head benchmark only.
  - Stage1 head median 0.845493 ms, p95 1.074642 ms, peak 0.012577 GB.
  - Stage2 head median 0.456376 ms, p95 0.483669 ms, peak 0.013739 GB.
  - Full Qwen ASR/audio-encoder forward and full cascade latency remain incomplete.
- Shared Qwen: `blocked_by_runtime_interface`.
- Long speech: `blocked_until_final_model_locked`; no false accepts/hour result is claimed.
- Final model: `not_trained_choices_not_frozen`; no deployment bundle trained.
- Blind-test protocol: ready as protocol scaffold, lock pending final model/thresholds.
- Professor report: `/home/hj/Data_Collect_Web/finetune/experiments/latest_data/reports/LATEST_PROFESSOR_MEETING_REPORT.md`

## Selected GPUs

- Avoid GPU 0 because it has unrelated active processes.
- Qwen feature and transcript cache used GPU 1.
- Nested folds used GPUs 1, 2, 3, 4, 6.
- No active tmux sessions after nested completion.

## Tests

- Latest package tests passed: 11 passed.
- Final post-change test sweep passed:
  - `finetune/tests`: 56 passed.
  - `finetune/evaluation/tests`: 13 passed.
  - `finetune/benchmarks/asr/tests`: 17 passed.
  - `finetune/experiments/participant_cv/tests`: 34 passed.
  - `finetune/experiments/vigil_final/tests`: 28 passed.
  - `finetune/experiments/latest_data/tests`: 11 passed.
  - `compileall`: passed.
  - `bash -n`: passed.

## Active Tmux Sessions

- None.

## Exact Next Command

```bash
cd /home/hj/Data_Collect_Web && git status --short
```

## Push Status

- Not committed.
- Not pushed.
