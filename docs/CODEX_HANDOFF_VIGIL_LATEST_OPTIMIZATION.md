# Codex Handoff: VIGIL Latest Data Optimization

## Current State

- Repository: `/home/hj/Data_Collect_Web`
- Branch: `research/vigil-latest-data-optimization-20260626`
- Base commit: `132ead0 Add latest VIGIL data workflow and reports`
- Dataset ZIP: `/home/hj/Data_Collect_Web/finetune/data/vigil_dataset_export_latest_readonly_20260626_042151.zip`
- Dataset SHA-256: `e2e38518d6725449653138e0ee484c4b5903467e418e8968d4b98ada5fd41701`
- Balanced manifest SHA-256: `549134e307f21470cb942acd44c2c27d2b29fcaa8527b9e7f8e2722e3232b58e`
- Fold SHA-256: `7c1c65da28f87922f111ee1549b61c053323fc876d2cd26346544de0b37b2a5e`

## Current Phase

- Reporting, commit, and push. Full LibriSpeech long-speech stress run is active in tmux.

## Selected GPUs

- Avoid GPU 0 because it has unrelated active processes.
- Idle candidates at start: GPUs `1`, `2`, `3`, `4`, `6`.

## Tmux Sessions

- `vigil_opt_longspeech`: active full LibriSpeech long-speech run on GPU 2. Output is ignored under `finetune/experiments/latest_data_optimization/runs/long_speech_full/`.

## Completed Jobs

- Git state verified on `research/vigil-latest-data-final-20260626` at `132ead0`.
- Created branch `research/vigil-latest-data-optimization-20260626`.
- Phase A artifact audit completed: latest ZIP SHA, balanced manifest SHA, fold SHA, leakage checks, transcript cache, and feature coverage verified.
- Phase B/C Stage2 operating point and recall optimization completed using fold validation predictions only.
- Phase D real support-based few-shot methods completed. No safe support-based F1 improvement was selected.
- Phase E full compute-cost measurement completed on 100 deterministic clips, with 10 warmups and 20 measured Qwen encoder forwards.
- Phase F long-speech subset completed on 40 LibriSpeech utterances; full run started in tmux.
- Phase G partial full-unbalanced heads-only ablation completed.
- Phase H shared-Qwen feasibility inspection completed.
- Phase I final deployment candidate trained under ignored `finetune/model_bundles/`.
- Phase J professor meeting report generated.

## Failed Jobs

- First full long-speech tmux launch exited because the alternate reports directory did not contain `latest_opt_stage2_selected_config.json`. Script now accepts `--selected-config`; full tmux was restarted successfully.

## Results

- Artifact audit status: `ok`.
- Latest raw/valid/balanced counts: 1673 raw clips, 1618 valid clips after audio QC, 1346 balanced clips, 1364 balanced windows, 37 participants.
- Leakage checks: participant leakage free `true`; duplicate-audio leakage free `true`.
- Feature coverage: official openWakeWord 1364/1364, frozen Qwen encoder 1364/1364.
- Qwen transcript cache: 1346/1346, result type `qwen_asr.inference.qwen3_asr.ASRTranscription`, extraction path `$[0].text`.
- Stage2 selected config: `stage2_bce_supcon`, `threshold_only`, `top_k=1`.
- Stage2 selected development metrics: recall `0.9509283819628647`, FPR `0.0`, F1 `0.9748470428280082`.
- Stage2 selected outer-test metrics: precision `0.9957325746799431`, recall `0.9408602150537635`, FPR `0.0049833887043189366`, F1 `0.9675190048375951`.
- Recomputed Stage1-only outer-test metrics: recall `0.9556451612903226`, FPR `0.053156146179401995`, F1 `0.9562878278412912`.
- Qwen exact baseline F1 mean from latest nested report: `0.7514269798385881`.
- Few-shot: real support-based threshold calibration, positive-bias adaptation, prototype fusion, and Qwen exact OR search were evaluated. Selected recipe remains `no_adaptation_zero_shot_fallback`; 3-shot paired F1 `0.9729729729729729`, 5-shot paired F1 `0.9758125472411188`, delta `0.0`.
- Optimized stage errors: Stage1 miss `33`, Stage2 reject `11`, Stage2 false accept `3`, Stage1 false candidates rejected `29`.
- Compute: Stage2 head median `1.4751083217561245` ms; extra Qwen encoder forward median `13.663365971297026` ms; Qwen ASR transcript median from recorded cache `250.64418883994222` ms; Stage2 F1 gain over Qwen exact `0.21609202499900704`; gain over Stage1-only `0.011231176996303938`.
- Long-speech subset: 40 utterances, `0.08139862847222222` hours, 873 windows, 168 Stage1 candidates, 30 Stage2 invocations, 0 false accepts, FAPH `0.0`. Full run active in tmux.
- Full-unbalanced partial heads-only ablation: 1597 clips/1615 windows; selected Stage2 recall `0.946843853820598`, FPR `0.007204610951008645`, F1 `0.9699376063528077`; Qwen exact full cache not generated.
- Shared Qwen: `blocked_by_runtime_interface`; qwen-asr version `0.0.6`, public transcribe returns `List[ASRTranscription]` and does not expose reusable hidden states.
- Final deployment candidate: `trained_deployment_candidate_not_scientific_test`; bundle `finetune/model_bundles/vigil_latest_optimized_20260626_085405`; Qwen weights not included; checkpoints not committed.

## Artifact Paths

- Optimization package: `/home/hj/Data_Collect_Web/finetune/experiments/latest_data_optimization`
- Previous latest-data package: `/home/hj/Data_Collect_Web/finetune/experiments/latest_data`
- Start audit: `/home/hj/Data_Collect_Web/finetune/experiments/latest_data_optimization/reports/OPTIMIZATION_START_AUDIT.md`
- Stage2 operating points: `/home/hj/Data_Collect_Web/finetune/experiments/latest_data_optimization/reports/LATEST_OPT_STAGE2_OPERATING_POINT_REPORT.md`
- Stage2 optimization: `/home/hj/Data_Collect_Web/finetune/experiments/latest_data_optimization/reports/LATEST_OPT_STAGE2_RECALL_REPORT.md`
- Few-shot: `/home/hj/Data_Collect_Web/finetune/experiments/latest_data_optimization/reports/LATEST_OPT_REAL_FEW_SHOT_ONBOARDING_REPORT.md`
- Compute: `/home/hj/Data_Collect_Web/finetune/experiments/latest_data_optimization/reports/LATEST_OPT_COMPUTE_COST_REPORT.md`
- Long-speech subset: `/home/hj/Data_Collect_Web/finetune/experiments/latest_data_optimization/reports/LATEST_OPT_LONG_SPEECH_FALSE_ACCEPTS_REPORT.md`
- Full long-speech active output: `/home/hj/Data_Collect_Web/finetune/experiments/latest_data_optimization/runs/long_speech_full/reports`
- Full-data partial ablation: `/home/hj/Data_Collect_Web/finetune/experiments/latest_data_optimization/reports/LATEST_OPT_BALANCED_VS_FULL_ABLATION_REPORT.md`
- Shared-Qwen feasibility: `/home/hj/Data_Collect_Web/finetune/experiments/latest_data_optimization/reports/LATEST_OPT_SHARED_QWEN_FEASIBILITY_REPORT.md`
- Final model report: `/home/hj/Data_Collect_Web/finetune/experiments/latest_data_optimization/reports/LATEST_OPT_FINAL_MODEL_REPORT.md`
- Professor report: `/home/hj/Data_Collect_Web/finetune/experiments/latest_data_optimization/reports/LATEST_OPT_PROFESSOR_MEETING_REPORT.md`

## Tests

- `finetune/tests`: 56 passed.
- `finetune/evaluation/tests`: 13 passed.
- `finetune/benchmarks/asr/tests`: 17 passed in `vigil-two-stage` env.
- `finetune/experiments/participant_cv/tests`: 34 passed.
- `finetune/experiments/vigil_final/tests`: 28 passed.
- `finetune/experiments/latest_data/tests`: 11 passed.
- `finetune/experiments/latest_data_optimization/tests`: 13 passed.
- `compileall`: passed.
- `bash -n`: passed.

## Exact Next Command

```bash
cd /home/hj/Data_Collect_Web && tmux capture-pane -pt vigil_opt_longspeech -S -80
```

## Push Status

- Not committed.
- Not pushed.
