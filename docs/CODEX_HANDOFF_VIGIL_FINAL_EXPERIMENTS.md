# Codex Handoff: VIGIL Final Experiments

## Current State

- Repository: `/home/hj/Data_Collect_Web`
- Branch: `research/vigil-fewshot-cost-nested-20260625`
- Start commit: `5912a09`
- Dataset fingerprint: `0fad4c7828149099`
- Balanced manifest checksum: `44815508a013b9022a8efc99a3972b6847884ebbb3578e356f18a50b822f5a03`
- Fold checksum: `e6759ee22e4358c2d7f4a3578b8568d6eb829ab7cfad69c4441cdc11b57d01cb`

## Completed Phases

- Read `docs/CODEX_HANDOFF_VIGIL_CV_FEWSHOT.md`.
- Verified current branch started from `research/vigil-participant-cv-fewshot-20260624` at `5912a09`.
- Fetched `origin`.
- Confirmed balanced manifest, shared folds, feature coverage, and Qwen transcript cache are present.
- Confirmed old few-shot implementation is only `no_adaptation_zero_shot_fallback`.
- Created branch `research/vigil-fewshot-cost-nested-20260625`.
- Started final experiment package under `finetune/experiments/vigil_final`.
- Implemented final-experiment package configs, source modules, scripts, and tests.
- Final package tests passed: `28 passed`.
- Current result audit passed: balanced manifest 1040 rows, transcript cache 1026 rows, feature coverage 1040/1040.
- Completed strict nested outer fold 0 smoke successfully.
- Completed strict nested participant-disjoint five-fold V2.
- Completed development-only Stage 2 operating-point report.
- Completed real prototype onboarding search/evaluation. Prototype recipes were selected on development pseudo-targets, but outer-test 3-shot/5-shot F1 decreased versus paired zero-shot.
- Completed partial component head latency/memory benchmark and Qwen call accounting.
- Completed shared-Qwen runtime diagnostic: blocked by public wrapper interface.
- Created future blind-test protocol scaffold and lock placeholder.

## Current Phase

- Updating reports, running tests, and preparing commit/push.

## Selected GPUs

- Avoid GPU 0 while it has active non-VIGIL processes.
- Fold 0 smoke used GPU `1` and completed.
- Nested folds used GPUs `1`, `2`, `3`, `4`, `6`.
- Real few-shot and component-cost scripts used GPU `1`.
- No active GPU jobs at this handoff update.

## Active Tmux Sessions

- None.

## Nested-CV Fold Progress

- Fold 0: complete. Selected variant `bce`; theta1 `0.7013291716575623`; validation-selected recall `0.9747899159663865`, FPR `0.05813953488372093`, F1 `0.9666666666666667`.
- Fold 1: complete. Selected variant `bce_supcon`; validation-selected recall `0.9596774193548387`, FPR `0.011904761904761904`, F1 `0.9754098360655739`.
- Fold 2: complete. Selected variant `bce`; validation-selected recall `0.904`, FPR `0.0`, F1 `0.9495798319327732`.
- Fold 3: complete. Selected variant `bce`; validation-selected recall `0.9537037037037037`, FPR `0.0`, F1 `0.976303317535545`.
- Fold 4: complete. Selected variant `bce_supcon`; validation-selected recall `0.9357798165137615`, FPR `0.011627906976744186`, F1 `0.9622641509433962`.
- Aggregate validation-selected: recall `0.9455901711077381`, FPR `0.016334440753045402`, precision `0.9881271229506005`, F1 `0.966044760628791`.

## Few-Shot Recipe Status

- Starting limitation confirmed: previous report used `no_adaptation_zero_shot_fallback`.
- Real prototype personalization was implemented and selected per outer fold by development pseudo-target safety gates.
- Paired outer-test results are negative:
  - 0-shot on 3-shot query F1 `0.9706490285241836`; 3-shot F1 `0.9653023062538957`; mean paired delta `-0.007877765522886233`.
  - 0-shot on 5-shot query F1 `0.9716393820613326`; 5-shot F1 `0.9683090446449225`; mean paired delta `-0.0033080380349756596`.
- Gradient adaptation guard/code exists, but full development gradient search was not run; no gradient recipe is selected.

## Cost-Benchmark Progress

- Partial head benchmark complete:
  - Stage 1 head params: total/trainable `56321`.
  - Stage 2 head params: total/trainable `561922`.
  - Stage 1 head median latency `0.0009344830177724361` s, p95 `0.0009559532627463341` s, peak allocated `0.012787818908691406` GB.
  - Stage 2 head median latency `0.0004961066879332066` s, p95 `0.000510866753757` s, peak allocated `0.013950824737548828` GB.
- Full Qwen ASR/audio-encoder and complete cascade latency remain incomplete in this branch.

## Shared-Qwen Prototype Status

- `blocked_by_runtime_interface`.
- Public `qwen_asr` wrapper returns transcript objects but does not expose reusable hidden-state handoff.
- Current System C accounting: one loaded Qwen instance, continuous ASR forward, additional audio-encoder forward per Stage 2 candidate.

## Long-Speech Progress

- Not run. Blocked until final model and thresholds are locked.

## Exact Next Command

```bash
cd /home/hj/Data_Collect_Web && git status --short
```

## Blockers

- Full unbalanced-data ablation not run.
- Full Qwen/cascade latency benchmark not complete.
- Gradient adaptation search not run.
- Long-speech false-activation stress test blocked until final model is locked.
- Final deployment candidate not trained because research choices are not fully frozen.

## Artifact Paths

- Package: `/home/hj/Data_Collect_Web/finetune/experiments/vigil_final`
- Handoff: `/home/hj/Data_Collect_Web/docs/CODEX_HANDOFF_VIGIL_FINAL_EXPERIMENTS.md`
- Previous package: `/home/hj/Data_Collect_Web/finetune/experiments/participant_cv`
- Nested report: `/home/hj/Data_Collect_Web/finetune/experiments/vigil_final/reports/NESTED_ZERO_SHOT_5FOLD_REPORT.md`
- Operating points: `/home/hj/Data_Collect_Web/finetune/experiments/vigil_final/reports/STAGE2_OPERATING_POINT_REPORT.md`
- Few-shot report: `/home/hj/Data_Collect_Web/finetune/experiments/vigil_final/reports/REAL_FEW_SHOT_ONBOARDING_REPORT.md`
- Component cost: `/home/hj/Data_Collect_Web/finetune/experiments/vigil_final/reports/COMPONENT_COST_REPORT.md`
- Shared-Qwen: `/home/hj/Data_Collect_Web/finetune/experiments/vigil_final/reports/SHARED_QWEN_PROTOTYPE_REPORT.md`
- Professor report: `/home/hj/Data_Collect_Web/finetune/experiments/vigil_final/reports/FINAL_PROFESSOR_MEETING_REPORT.md`

## Push Status

- Not pushed.
