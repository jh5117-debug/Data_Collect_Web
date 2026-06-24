# VIGIL Browser Demo

This Gradio demo runs the trained two-stage VIGIL detector on browser microphone recordings or uploaded audio. It loads openWakeWord, the Stage 1 checkpoint, frozen Qwen3-ASR-1.7B, both Stage 2 verifier heads, thresholds, and validation-based model selection once at startup.

Run on one idle RTX 3090:

```bash
cd /home/hj/Data_Collect_Web
tmux new -d -s vigil_live_demo \
  "PATH=/home/hj/miniconda/envs/vigil-two-stage/bin:\$PATH \
   bash finetune/demo/run_demo.sh \
   6 \
   /home/hj/Data_Collect_Web/finetune/runs/20260624_075127_0fad4c7828149099_full"
```

Open through an SSH tunnel:

```bash
ssh -L 7860:127.0.0.1:7860 hal
```

Then open:

```text
http://127.0.0.1:7860
```

The app binds only to `127.0.0.1` and does not use Gradio `share=True`.

## Current HAL Session

As of the corrected Qwen transcript extraction run, the demo is running in:

```text
tmux: vigil_live_demo
python pid: 1099372
log: /home/hj/Data_Collect_Web/finetune/demo/logs/vigil_demo_20260624_205136_gpu6.log
url: http://127.0.0.1:7860
```

The selected GPU is physical GPU 6. Do not start a concurrent benchmark or training run on that GPU while the demo is active.

The default UI variant remains `Validation-selected`, which resolves to the validation-selected `stage2_bce` checkpoint from `model_selection.json`.

Qwen transcript extraction uses the shared extractor in:

```text
finetune/src/vigil_two_stage/qwen_text_result.py
```

The observed Qwen ASR path is `$[0].text` from `qwen_asr.inference.qwen3_asr.ASRTranscription`.

## File-Upload Validation

Local Gradio API file-upload validation passed on held-out files:

```text
report: /home/hj/Data_Collect_Web/finetune/demo/reports/VIGIL_LIVE_DEMO_FILE_UPLOAD_VALIDATION.md
P1: C000412 -> VIGIL DETECTED
P2: C000414 -> VIGIL DETECTED
P3: C000023 -> VIGIL DETECTED
P4: C000024 -> REJECTED
P4: C000423 -> REJECTED
P4: C000424 -> REJECTED
```

Browser microphone capture has not been manually validated on the user's laptop. A human must open the tunneled page and record audio before claiming microphone validation.

Try:

- VIGIL
- Hi VIGIL
- VIGIL, go back
- What's next, VIGIL?
- visual
- visible
- digital
- individual
- vigilant
- video

Temporary converted audio and windows are deleted after inference unless a debug directory is explicitly passed in code. Demo recordings, logs, checkpoints, and generated reports must not be committed.
