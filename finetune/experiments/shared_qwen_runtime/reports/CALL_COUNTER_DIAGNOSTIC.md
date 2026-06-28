# Call Counter Diagnostic

- Model load count: `1`
- Public transcribe counts: `{'model_load_count': 1, 'transcribe_call_count': 1, 'generate_call_count': 1, 'get_audio_features_call_count': 1, 'thinker_forward_call_count': 6, 'decoder_call_count': 0, 'encoder_call_count': 7, 'patched_methods': ['generate', 'get_audio_features', 'thinker_forward', 'transcribe']}`
- Separate Stage 2 feature counts: `{'model_load_count': 1, 'transcribe_call_count': 0, 'generate_call_count': 0, 'get_audio_features_call_count': 1, 'thinker_forward_call_count': 0, 'decoder_call_count': 0, 'encoder_call_count': 1, 'patched_methods': ['generate', 'get_audio_features', 'thinker_forward', 'transcribe']}`
- Public transcribe + separate Stage 2 counts: `{'model_load_count': 1, 'transcribe_call_count': 1, 'generate_call_count': 1, 'get_audio_features_call_count': 2, 'thinker_forward_call_count': 6, 'decoder_call_count': 0, 'encoder_call_count': 8, 'patched_methods': ['generate', 'get_audio_features', 'thinker_forward', 'transcribe']}`
- Attempted shared status: `blocked_by_runtime_interface`
- Attempted shared encoder calls: `7`
- Attempted shared decoder calls: `1`
- Can claim one encoder forward: `False`
