# Shared Qwen Non-Regression Report

- Status: `blocked_by_runtime_interface`
- ASR smoke: `not_run_because_shared_path_blocked`
- VIGIL metric check: `not_run_because_shared_path_blocked`
- Current frozen Qwen combined WER: `0.02751646508258752`
- Current VIGIL F1: `0.9675190048`
- Blocker: The current public qwen_asr wrapper does not expose decoder-compatible audio hidden states and does not accept externally supplied audio hidden states for decoding.
