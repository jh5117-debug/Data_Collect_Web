# Transcript Parity Report

- Status: `blocked_by_runtime_interface`
- VIGIL examples transcribed with public path: `25`
- LibriSpeech examples transcribed with public path: `4`
- Shared transcript parity: `blocked`
- Blocker: The current public qwen_asr wrapper does not expose decoder-compatible audio hidden states and does not accept externally supplied audio hidden states for decoding.

Public transcripts were collected as a sanity check. Shared-path transcript parity is blocked because no project-owned one-forward path can pass hidden states into the Qwen decoder.
