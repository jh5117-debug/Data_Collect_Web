# Shared Qwen Hidden-State Feasibility

Status: possible but not yet exposed by the current public `qwen_asr` wrapper.

Evidence: the public `transcribe(path, language=None)` call returns `list[ASRTranscription]` containing text and language metadata, not reusable hidden states. The Stage 2 feature path uses `model.thinker.get_audio_features` separately. A shared-hidden-state system would require changing or extending the continuous ASR runtime to expose audio hidden states safely. No verified shared-hidden-state inference path was implemented in this task.
