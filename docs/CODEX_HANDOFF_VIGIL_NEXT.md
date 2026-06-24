# Codex Handoff: VIGIL Next Milestone

## Current State

- Repository: `/home/hj/Data_Collect_Web`
- Branch: `research/vigil-eval-live-demo-20260624`
- Commit: final handoff status is current `HEAD`; code fix commit `538f646`, benchmark/demo documentation commit `3bbc7bf`.
- Remote: `origin git@github.com:jh5117-debug/Data_Collect_Web.git`
- Dataset ZIP: `/home/hj/Data_Collect_Web/finetune/data/vigil_dataset_export_20260624_072023_180681.zip`
- Dataset fingerprint: `0fad4c7828149099`
- Processed dataset: `/home/hj/Data_Collect_Web/finetune/data/processed/0fad4c7828149099`
- Current completed full run: `/home/hj/Data_Collect_Web/finetune/runs/20260624_075127_0fad4c7828149099_full`
- Current run report: `/home/hj/Data_Collect_Web/finetune/runs/20260624_075127_0fad4c7828149099_full/FINAL_REPORT.md`
- Current evaluation audit: `/home/hj/Data_Collect_Web/finetune/runs/20260624_075127_0fad4c7828149099_full/EVALUATION_AUDIT.md`
- Current model selection: `/home/hj/Data_Collect_Web/finetune/runs/20260624_075127_0fad4c7828149099_full/MODEL_SELECTION.md`
- Invalid historical LibriSpeech full benchmark: `/home/hj/Data_Collect_Web/finetune/benchmarks/asr/runs/20260624_090118_qwen3_asr_1_7b_baseline_full`
- Corrected LibriSpeech full benchmark: `/home/hj/Data_Collect_Web/finetune/benchmarks/asr/runs/20260624_185419_qwen3_asr_1_7b_fixed_text_extraction_baseline_full`
- Preserved untracked file, do not touch/commit: `docs/VIGIL_Recorder_Participant_Guide.docx`

## Qwen Transcript Extraction Correction

- Root cause: `qwen_asr.inference.qwen3_asr.Qwen3ASRModel.transcribe()` returns `list[qwen_asr.inference.qwen3_asr.ASRTranscription]`; old extractors recursed into `[0]` and then used `str(result)`, storing `ASRTranscription(language=..., text=..., time_stamps=None)` instead of `.text`.
- Actual result type: outer `builtins.list`, length 1; item type `qwen_asr.inference.qwen3_asr.ASRTranscription`; transcript field `.text`; extraction path `$[0].text`.
- Files changed in fix commit `538f646`: `finetune/src/vigil_two_stage/qwen_text_result.py`, `finetune/src/vigil_two_stage/__init__.py`, `finetune/benchmarks/asr/src/qwen_runner.py`, `finetune/benchmarks/asr/scripts/run_qwen_librispeech.py`, `finetune/scripts/run_qwen_text_baseline.py`, `finetune/demo/inference.py`, `finetune/tests/test_qwen_text_result.py`, `finetune/benchmarks/asr/tests/test_qwen_transcript_call_sites.py`.
- Old invalid run: `/home/hj/Data_Collect_Web/finetune/benchmarks/asr/runs/20260624_090118_qwen3_asr_1_7b_baseline_full`; 5559/5559 stored hypotheses were structured object reprs. Its 0.40069005613854497 normalized WER is invalid for scientific reporting.
- Runtime diagnostic: `/home/hj/Data_Collect_Web/finetune/benchmarks/asr/reports/QWEN_RUNTIME_RESULT_DIAGNOSTIC.md`.
- Old-run audit: `/home/hj/Data_Collect_Web/finetune/benchmarks/asr/reports/QWEN_OLD_RUN_EXTRACTION_AUDIT.md`.
- One-utterance sanity check: `/home/hj/Data_Collect_Web/finetune/benchmarks/asr/reports/QWEN_FIXED_ONE_UTTERANCE_SANITY.md`; clean transcript, path `$[0].text`, latency 2.6987866358831525s, peak GPU 3.885563850402832 GB.
- Corrected smoke run: `/home/hj/Data_Collect_Web/finetune/benchmarks/asr/runs/20260624_185009_qwen3_asr_1_7b_fixed_text_extraction_smoke_smoke`; 64 successes, 0 failures, normalized WER 0.037165082108902334, test-clean 0.015455950540958269, test-other 0.06470588235294118, malformed hypotheses 0.
- Corrected full run: `/home/hj/Data_Collect_Web/finetune/benchmarks/asr/runs/20260624_185419_qwen3_asr_1_7b_fixed_text_extraction_baseline_full`; 5559 successes, 0 failures, normalized WER 0.02751646508258752, test-clean 0.018411442483262326, test-other 0.03666201784383776, normalized CER 0.009904616200632524, raw WER 0.9862560642019081, malformed hypotheses 0.
- Corrected VIGIL Qwen baseline: `/home/hj/Data_Collect_Web/finetune/runs/20260624_075127_0fad4c7828149099_full/baseline_qwen_exact_clip_fixed_text_extraction`; n 93, precision 1.0, recall 0.6862745098039216, FPR 0.0, F1 0.813953488372093. The old 0.6862745098039216 recall did not change because exact wake-word matching still found `vigil` inside many object reprs, but old transcripts are not valid transcript artifacts.
- Selected GPU: physical GPU 6 only.
- Benchmark tmux sessions: `librispeech_qwen_fixed_smoke` and `librispeech_qwen_fixed_full` completed and exited.
- Active demo tmux session: `vigil_live_demo`, Python PID `1099372`, log `/home/hj/Data_Collect_Web/finetune/demo/logs/vigil_demo_20260624_205136_gpu6.log`.
- Demo status: running on `127.0.0.1:7860`, `share=False`; file-upload validation passed for P1/P2/P3 and three P4 examples. Browser microphone capture has not been human-validated.
- Demo validation report: `/home/hj/Data_Collect_Web/finetune/demo/reports/VIGIL_LIVE_DEMO_FILE_UPLOAD_VALIDATION.md`.
- ASR preservation report: `/home/hj/Data_Collect_Web/finetune/reports/ASR_PRESERVATION_REPORT.md`.
- Progress: code fix, corrected reports, documentation, ASR preservation report, and demo status are committed and pushed. This file's final status update is the only later handoff-only change.
- Exact next command:

```bash
cd /home/hj/Data_Collect_Web && PATH=/home/hj/miniconda/envs/vigil-two-stage/bin:$PATH PYTHONPATH=finetune/src:. pytest -q finetune/tests
```

- Push status: branch pushed to `origin/research/vigil-eval-live-demo-20260624`; `3bbc7bf` is pushed, followed by a final handoff-only status commit if present.

## Completed Phases

- Initial Git/artifact audit completed.
- Confirmed current branch contains the `finetune/` pipeline, Admin background export fix, and LibriSpeech benchmark code.
- Created branch `research/vigil-eval-live-demo-20260624` from complete feature HEAD.
- Audited existing completed full run without retraining.
- Added `finetune/evaluation/` audit, split-report, clip-aggregation, window-vs-clip comparison, and validation-only model-selection tools.
- Fixed `finetune/scripts/run_qwen_text_baseline.py` so the formal default baseline is clip-level, deduplicated by `clip_id`; legacy window mode remains available with `--evaluation-unit window`.
- Recomputed current run metrics from existing predictions only.
- Materialized corrected Qwen clip-level baseline at `baseline_qwen_exact_clip/`.
- Created validation-only model selection artifacts.
- Downloaded and prepared LibriSpeech `test-clean` and `test-other` manifests: 2620 + 2939 = 5559 utterances, smoke subset 64.
- Fixed ASR benchmark CER scoring to avoid corpus-wide character DP; resume progress now reports `skipped_existing`.
- Historical invalid LibriSpeech 64-utterance smoke on physical GPU 6:
  - sanity check succeeded.
  - 64 predictions, 0 failures.
  - normalized WER 0.43042350907519444, invalid due the same structured-object transcript extraction bug.
  - raw WER 1.0924805531547104, invalid due the same structured-object transcript extraction bug.
  - normalized CER 0.5938211382113822, invalid due the same structured-object transcript extraction bug.
  - mean latency 0.809393223884399 seconds.
  - mean real-time factor 0.12325373119946118.
  - peak GPU memory 3.9380016326904297 GB.
  - resume verification: `completed_now=0`, `skipped_existing=64`, WER unchanged.
- Historical invalid full LibriSpeech baseline on physical GPU 6:
  - 5559 predictions, 0 failures.
  - normalized WER 0.40069005613854497, invalid for scientific reporting.
  - raw WER 1.0919852457610157, invalid for scientific reporting.
  - normalized CER 0.5702244505102952, invalid for scientific reporting.
  - test-clean normalized WER 0.3694841752891053, invalid for scientific reporting.
  - test-other normalized WER 0.43203484706646544, invalid for scientific reporting.
  - mean latency 0.8274251775317502 seconds.
  - mean real-time factor 0.12909503772819741.
  - peak GPU memory 4.068508625030518 GB.
- Added `finetune/demo/` Gradio browser microphone demo source, startup script, and CPU tests. Demo is now running in `vigil_live_demo` on GPU 6.
- Ran tests:
  - `PYTHONPATH=finetune/src:. pytest -q finetune/tests` -> 56 passed.
  - `PYTHONPATH=finetune/src:finetune/evaluation:. pytest -q finetune/evaluation/tests` -> 13 passed.
  - `PYTHONPATH=finetune/benchmarks/asr/src:finetune/src:. pytest -q finetune/benchmarks/asr/tests` -> 17 passed.
  - `PYTHONPATH=finetune/src:finetune/demo:. pytest -q finetune/demo/tests` -> 4 passed.
  - `python -m compileall -q finetune/src finetune/scripts finetune/benchmarks/asr finetune/evaluation finetune/demo` -> passed.
  - `bash -n` on ASR benchmark launchers, demo launcher, and `finetune/scripts/*.sh` -> passed.

## Exact Current Audit Results

- Train split: 21 speakers, 21 participant keys, 21 sessions, 1099 clips, 1113 windows, 657 positive windows, 456 negative windows.
- Val split: 4 speakers, 4 participant keys, 4 sessions, 106 clips, 106 windows, 57 positive windows, 49 negative windows.
- Test split: 5 speakers, 5 participant keys, 5 sessions, 93 clips, 93 windows, 51 positive windows, 42 negative windows.
- Prompt clip counts:
  - Train: P1 117, P2 337, P3 203, P4 442.
  - Val: P1 11, P2 26, P3 20, P4 49.
  - Test: P1 12, P2 22, P3 17, P4 42.
- Checks: no speaker leakage, no session leakage, no duplicate audio leakage, one split per clip, one label per clip, one speaker per clip, split manifests consistent.
- Threshold audit: Stage 1 and both Stage 2 thresholds recompute exactly from validation predictions; test metrics were present but not used for threshold selection.
- Training history:
  - Stage 1: 13 epochs, best val epoch 8, early-stop epoch 13, 35 train steps/epoch, approx 455 optimizer steps.
  - Stage 2 BCE: 8 epochs, best val epoch 3, early-stop epoch 8, 140 train steps/epoch, approx 1120 optimizer steps.
  - Stage 2 BCE + SupCon: 8 epochs, best val epoch 3, early-stop epoch 8, 140 train steps/epoch, approx 1120 optimizer steps.
- Current test split has no multi-window clips, so corrected clip-level cascade metrics equal legacy window-level cascade metrics for test.
- Corrected test clip-level cascade:
  - Stage2 BCE: precision 1.0, recall 0.8235294117647058, FPR 0.0, F1 0.9032258064516129.
  - Stage2 BCE + SupCon: precision 1.0, recall 0.9411764705882353, FPR 0.0, F1 0.9696969696969697.
- Corrected Qwen clip baseline: n 93, precision 1.0, recall 0.6862745098039216, FPR 0.0, F1 0.813953488372093.
- Validation-only model selection selected `stage2_bce`.
  - Both variants missed the validation recall target 0.90 at clip-level cascade.
  - BCE had higher validation recall/F1: recall 0.8947368421052632, P4 FPR 0.0, precision 1.0, F1 0.9444444444444444.
  - Test metrics were not used for selection.
  - Selection changed from previously reported test-best/default SupCon.

## Active tmux Sessions

- `vigil_live_demo`: running the Gradio browser/upload demo on physical GPU 6.
- Demo Python PID: `1099372`.
- Demo log: `/home/hj/Data_Collect_Web/finetune/demo/logs/vigil_demo_20260624_205136_gpu6.log`.
- Completed corrected full benchmark log: `/home/hj/Data_Collect_Web/finetune/benchmarks/asr/logs/librispeech_full_20260624_185419_gpu6.log`.
- Completed corrected full benchmark run dir: `/home/hj/Data_Collect_Web/finetune/benchmarks/asr/runs/20260624_185419_qwen3_asr_1_7b_fixed_text_extraction_baseline_full`.
- Invalid historical full benchmark run dir: `/home/hj/Data_Collect_Web/finetune/benchmarks/asr/runs/20260624_090118_qwen3_asr_1_7b_baseline_full`.

## Current GPU

- Physical GPU 6 is assigned to the active `vigil_live_demo` tmux session.
- Do not start another concurrent GPU job on GPU 6 while the demo is running.
- Required check:

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
```

## Commands Already Executed

```bash
cd /home/hj/Data_Collect_Web
git status --short
git branch --show-current
git log --oneline --decorate -15
git remote -v
git log --all --graph --decorate --oneline -25
test -d finetune && test -f finetune/scripts/run_full.sh
test -f backend/app/services/export_jobs.py
test -d finetune/benchmarks/asr
git switch -c research/vigil-eval-live-demo-20260624
tmux ls
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
PATH=/home/hj/miniconda/envs/vigil-two-stage/bin:$PATH PYTHONPATH=finetune/src:finetune/evaluation:. pytest -q finetune/evaluation/tests
PATH=/home/hj/miniconda/envs/vigil-two-stage/bin:$PATH PYTHONPATH=finetune/src:finetune/evaluation:. python finetune/evaluation/audit_existing_run.py --dataset-dir /home/hj/Data_Collect_Web/finetune/data/processed/0fad4c7828149099 --run-dir /home/hj/Data_Collect_Web/finetune/runs/20260624_075127_0fad4c7828149099_full
PATH=/home/hj/miniconda/envs/vigil-two-stage/bin:$PATH PYTHONPATH=finetune/src:. pytest -q finetune/tests
PATH=/home/hj/miniconda/envs/vigil-two-stage/bin:$PATH PYTHONPATH=finetune/benchmarks/asr/src:finetune/src:. pytest -q finetune/benchmarks/asr/tests
PATH=/home/hj/miniconda/envs/vigil-two-stage/bin:$PATH python -m compileall -q finetune/src finetune/scripts finetune/evaluation finetune/benchmarks/asr
PATH=/home/hj/miniconda/envs/vigil-two-stage/bin:$PATH bash finetune/benchmarks/asr/scripts/download_librispeech_eval.sh
tmux new -d -s librispeech_qwen_smoke 'cd /home/hj/Data_Collect_Web && PATH=/home/hj/miniconda/envs/vigil-two-stage/bin:$PATH bash finetune/benchmarks/asr/scripts/run_librispeech_smoke.sh 6 Qwen/Qwen3-ASR-1.7B qwen3_asr_1_7b_smoke'
PATH=/home/hj/miniconda/envs/vigil-two-stage/bin:$PATH CUDA_VISIBLE_DEVICES=6 PYTHONPATH=finetune/benchmarks/asr/src:finetune/src:. HF_HOME=/home/hj/Data_Collect_Web/finetune/cache/huggingface HF_HUB_CACHE=/home/hj/Data_Collect_Web/finetune/cache/huggingface/hub TRANSFORMERS_CACHE=/home/hj/Data_Collect_Web/finetune/cache/huggingface/transformers TORCH_HOME=/home/hj/Data_Collect_Web/finetune/cache/torch PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python finetune/benchmarks/asr/scripts/run_qwen_librispeech.py --manifest finetune/benchmarks/asr/manifests/smoke_all.jsonl --output-dir finetune/benchmarks/asr/runs/20260624_085312_qwen3_asr_1_7b_smoke_smoke --run-name qwen3_asr_1_7b_smoke --model Qwen/Qwen3-ASR-1.7B --resume --require-baseline-model
tmux new -d -s librispeech_qwen_full 'cd /home/hj/Data_Collect_Web && PATH=/home/hj/miniconda/envs/vigil-two-stage/bin:$PATH bash finetune/benchmarks/asr/scripts/run_librispeech_full.sh 6 Qwen/Qwen3-ASR-1.7B qwen3_asr_1_7b_baseline'
PATH=/home/hj/miniconda/envs/vigil-two-stage/bin:$PATH PYTHONPATH=finetune/src:finetune/demo:. pytest -q finetune/demo/tests
PATH=/home/hj/miniconda/envs/vigil-two-stage/bin:$PATH python -m compileall -q finetune/src finetune/scripts finetune/evaluation finetune/demo finetune/benchmarks/asr
bash -n finetune/demo/run_demo.sh
```

## Exact Next Command

```bash
cd /home/hj/Data_Collect_Web
PATH=/home/hj/miniconda/envs/vigil-two-stage/bin:$PATH PYTHONPATH=finetune/src:. pytest -q finetune/tests
```

The corrected full benchmark and file-upload demo validation are complete. Browser microphone validation still requires a human opening the page.

## Blockers

- None for Phase A.
- Phase B may be blocked by no idle RTX 3090, missing Qwen cache/runtime, OpenSLR download failure, or disk/network issues. Do not use CPU fallback.

## Generated Artifact Paths

- Handoff file: `/home/hj/Data_Collect_Web/docs/CODEX_HANDOFF_VIGIL_NEXT.md`
- Evaluation source: `/home/hj/Data_Collect_Web/finetune/evaluation/`
- Demo source: `/home/hj/Data_Collect_Web/finetune/demo/`
- Current run audit JSON: `/home/hj/Data_Collect_Web/finetune/runs/20260624_075127_0fad4c7828149099_full/evaluation_audit.json`
- Current run audit Markdown: `/home/hj/Data_Collect_Web/finetune/runs/20260624_075127_0fad4c7828149099_full/EVALUATION_AUDIT.md`
- Window/clip metrics: `/home/hj/Data_Collect_Web/finetune/runs/20260624_075127_0fad4c7828149099_full/evaluation/window_clip_metrics.json`
- Corrected Qwen clip baseline: `/home/hj/Data_Collect_Web/finetune/runs/20260624_075127_0fad4c7828149099_full/baseline_qwen_exact_clip/`
- Model selection JSON: `/home/hj/Data_Collect_Web/finetune/runs/20260624_075127_0fad4c7828149099_full/model_selection.json`
- Model selection Markdown: `/home/hj/Data_Collect_Web/finetune/runs/20260624_075127_0fad4c7828149099_full/MODEL_SELECTION.md`
- LibriSpeech dataset report: `/home/hj/Data_Collect_Web/finetune/benchmarks/asr/reports/dataset_download_report.md`
- LibriSpeech smoke run: `/home/hj/Data_Collect_Web/finetune/benchmarks/asr/runs/20260624_085312_qwen3_asr_1_7b_smoke_smoke`
- LibriSpeech smoke log: `/home/hj/Data_Collect_Web/finetune/benchmarks/asr/logs/librispeech_smoke_20260624_085312_gpu6.log`
- LibriSpeech full completed run: `/home/hj/Data_Collect_Web/finetune/benchmarks/asr/runs/20260624_090118_qwen3_asr_1_7b_baseline_full`
- LibriSpeech full completed log: `/home/hj/Data_Collect_Web/finetune/benchmarks/asr/logs/librispeech_full_20260624_090118_gpu6.log`

## Git Commit And Push Status

- Qwen extractor code/test commit created locally: `538f646 Fix structured Qwen ASR transcript extraction`.
- Documentation/status commit created and pushed: `3bbc7bf Record corrected Qwen benchmark and live demo status`.
- Final handoff status update is this file only.
- Source/docs commit created: `2f68674 Fix VIGIL clip evaluation workflow`.
- Handoff status commit created: `8c6e3d7 Update VIGIL handoff status`.
- Branch push status commit created: `df32ddb Record VIGIL branch push status`.
- Branch has been pushed to `origin/research/vigil-eval-live-demo-20260624`, at least through `df32ddb`.
- If this file has a newer final handoff-status commit, run `git rev-parse --short HEAD` for the exact latest commit.
- Already staged/committed in `2f68674`: `.gitignore`, `finetune/scripts/run_qwen_text_baseline.py`, `finetune/evaluation/`, `finetune/benchmarks/asr/scripts/run_qwen_librispeech.py`, `finetune/benchmarks/asr/src/scoring.py`, `finetune/benchmarks/asr/tests/test_scoring.py`, `finetune/demo/`, `docs/CODEX_HANDOFF_VIGIL_NEXT.md`.
- Do not stage: `docs/VIGIL_Recorder_Participant_Guide.docx`.
- Existing tracked generated changes from before this phase remain unstaged: `finetune/reports/export_inspection.json`, `finetune/reports/export_inspection.md`.
