# VIGIL Main Merge Summary

## What Was Merged

The latest VIGIL research, evaluation, demo, and integration documentation were merged toward `main` through an integration branch.

## Relevant Branches

- `feature/vigil-two-stage-smoke-20260620`
- `research/vigil-eval-live-demo-20260624`
- `research/vigil-latest-data-optimization-20260626`
- `research/vigil-stage1-fewshot-sharedqwen-20260626`
- `research/vigil-shared-qwen-runtime-20260627`
- `research/vigil-browser-assistant-demo-20260627`
- `research/vigil-target-doctor-fewshot-ablation-20260630`

## Key Results

| Area | Result |
|---|---|
| Optimized trigger | recall 0.9409, FPR 0.0050, precision 0.9957, F1 0.9675 |
| Corrected LibriSpeech ASR | combined WER 2.7516% |
| Few-shot best | 5-shot Stage 2 cosine prototype, F1 0.97059 |
| Stage 1 | frozen openWakeWord feature extractor, 56,321 trainable head params |
| Shared Qwen | hidden-state reuse not verified; Stage 2 still needs extra encoder forward |

## How To Run

Frontend/backend development:

```bash
cd frontend && npm install && npm run dev
cd backend && PYTHONPATH=. uvicorn app.main:app --reload --port 8000
```

Finetune smoke:

```bash
cd /home/hj/Data_Collect_Web
PATH=/home/hj/miniconda/envs/vigil-two-stage/bin:$PATH bash finetune/scripts/run_official_smoke_local_3090.sh
```

Browser demo:

```bash
cd /home/hj/Data_Collect_Web
PATH=/home/hj/miniconda/envs/vigil-two-stage/bin:$PATH \
bash finetune/demo_live_assistant/scripts/run_demo.sh 6 \
  /home/hj/Data_Collect_Web/finetune/runs/20260624_075127_0fad4c7828149099_full
```

## ROS 2 Integration Suggestion

Package the trigger as a node that subscribes to audio chunks, runs Stage 1 streaming candidate detection, runs Stage 2 verification on candidates, and publishes trigger events plus optional transcript segments.

## Remaining Blockers

- Public Qwen runtime does not yet provide verified shared hidden-state reuse.
- Browser demo has no downstream LLM/VQA implementation.
- A minimal stable trigger API should be extracted from the research scripts before robot integration.

## Main Docs

- `docs/VIGIL_TRIGGER_INTEGRATION.md`
- `docs/VIGIL_CURRENT_STATUS.md`
- `docs/VIGIL_MODEL_ARCHITECTURE.md`
- `docs/VIGIL_EXPERIMENT_RESULTS.md`
- `docs/VIGIL_BROWSER_DEMO.md`
- `docs/VIGIL_DATA_COLLECTION_PROTOCOL.md`
