# Final Shared Qwen Runtime Report

## Professor Question

Can one frozen Qwen3-ASR runtime provide both the continuous transcript and the Stage 2 VIGIL verifier features without a second audio encoder forward?

## Current System

The current clinical workflow uses a continuous frozen Qwen ASR branch for transcript and a parallel VIGIL trigger branch. Stage 2 uses frozen Qwen audio features with a small verifier head. Qwen weights remain frozen.

## Inspection And Attempt

- qwen-asr version: `0.0.6`
- Public transcribe result extraction path: `$[0].text`
- Stage 2 feature path: `FrozenQwenAudioAdapter -> qwen_asr.inference.utils.normalize_audio_input -> processor(text=[audio_token], audio=[audio]) -> model.thinker.get_audio_features(input_features, feature_attention_mask)`
- Call-counter combined path: `{'model_load_count': 1, 'transcribe_call_count': 1, 'generate_call_count': 1, 'get_audio_features_call_count': 2, 'thinker_forward_call_count': 6, 'decoder_call_count': 0, 'encoder_call_count': 8, 'patched_methods': ['generate', 'get_audio_features', 'thinker_forward', 'transcribe']}`
- Attempted shared status: `blocked_by_runtime_interface`

## Final Status

`blocked_by_runtime_interface`

The current public qwen_asr wrapper does not expose decoder-compatible audio hidden states and does not accept externally supplied audio hidden states for decoding. Therefore, we cannot yet prove one-forward shared Qwen-ASR. Current prototype still uses same frozen Qwen weights but one extra Qwen encoder forward for Stage 2 candidates.

## Transcript Parity

- Status: `blocked`

## Stage 2 Score Parity

- Status: `blocked`

## Metric Non-Regression

- Status: `blocked_by_runtime_interface`
- No shared-path LibriSpeech/VIGIL metric claim is made because the shared path is blocked.

## Cost

- Current extra encoder median cost: `13.663365971297026` ms per Stage 1 candidate.

## Chinese Notes

当前 qwen_asr 公共接口只返回转写文本，不返回可以复用给解码器和 Stage 2 的同一份 audio hidden states。因此现在不能声称一个 encoder forward 同时服务 ASR 和 VIGIL Stage 2。下一步需要上游接口暴露 decoder-compatible audio hidden states，或允许 decoder 接收外部传入的 hidden states。

## Exact Next Technical Step

Request or implement a project-owned Qwen wrapper API that returns decoder-compatible audio hidden states and accepts those same states for generation, then rerun call-counter, transcript parity, Stage 2 score parity, and non-regression checks.
