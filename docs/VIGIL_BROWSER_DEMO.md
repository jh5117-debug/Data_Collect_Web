# VIGIL Browser Demo

## Branch Merged

The local browser assistant demo came from `research/vigil-browser-assistant-demo-20260627`.

## Local HAL Run Command

```bash
cd /home/hj/Data_Collect_Web
PATH=/home/hj/miniconda/envs/vigil-two-stage/bin:$PATH \
bash finetune/demo_live_assistant/scripts/run_demo.sh 6 \
  /home/hj/Data_Collect_Web/finetune/runs/20260624_075127_0fad4c7828149099_full
```

The server binds to `127.0.0.1:7861` on HAL by default.

## SSH Tunnel From Laptop

```bash
ssh -N -L 7862:127.0.0.1:7861 hj@130.149.110.182
```

Then open:

```text
http://127.0.0.1:7862
```

## User Flow

1. Enter a local name/profile.
2. Record a few onboarding VIGIL examples.
3. Run prototype calibration.
4. Start assistant listening.
5. Watch rolling transcript and VIGIL trigger state.
6. On trigger, the demo shows activation state and scores.

## Notes

- There is no downstream LLM or VQA response implemented yet.
- Qwen and openWakeWord are loaded for local inference and remain frozen.
- Local audio is ignored by Git and should not be committed.

## Troubleshooting

- If `127.0.0.1:7862` refuses connection on the laptop, the SSH tunnel is not running or the port is already occupied.
- If HAL `127.0.0.1:7861` refuses connection, the demo process is not running or failed during model load.
- If microphone permission fails, use Chrome on `127.0.0.1` or HTTPS.
- If inference returns 500, inspect `finetune/demo_live_assistant/logs/`, which is ignored by Git.
