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

