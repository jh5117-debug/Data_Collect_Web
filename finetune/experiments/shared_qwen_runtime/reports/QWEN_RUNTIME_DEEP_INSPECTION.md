# Qwen Runtime Deep Inspection

- qwen-asr version: `0.0.6`
- Qwen class: `qwen_asr.inference.qwen3_asr.Qwen3ASRModel`
- Source file: `/home/hj/miniconda/envs/vigil-two-stage/lib/python3.12/site-packages/qwen_asr/inference/qwen3_asr.py`
- Transcribe signature: `(self, audio: Union[str, Tuple[numpy.ndarray, int], List[Union[str, Tuple[numpy.ndarray, int]]]], context: Union[str, List[str]] = '', language: Union[str, List[Optional[str]], NoneType] = None, return_time_stamps: bool = False) -> List[qwen_asr.inference.qwen3_asr.ASRTranscription]`
- Loaded backend: `transformers`
- Model generate signature: `(input_ids: Optional[torch.Tensor] = None, max_new_tokens: int = 4096, eos_token_id: int | list[int] = [151645, 151643], **kwargs)`
- thinker.get_audio_features signature: `(input_features: torch.FloatTensor, feature_attention_mask: Optional[torch.LongTensor] = None, audio_feature_lengths: Optional[torch.LongTensor] = None)`
- Public methods exposing hidden/features: `[]`

## Code Path

- transcribe normalizes audio with normalize_audios
- transcribe splits audio into chunks
- transcribe calls _infer_asr(contexts, wavs, languages)
- _infer_asr_transformers builds processor(text=..., audio=...) inputs
- _infer_asr_transformers calls self.model.generate(**inputs)
- decoded text is parsed into ASRTranscription(language, text, time_stamps)

## Finding

- Hidden states accessible from public transcribe: `False`
- Decoder accepts external hidden states in public wrapper: `False`
- Stage 2 feature path: `FrozenQwenAudioAdapter -> qwen_asr.inference.utils.normalize_audio_input -> processor(text=[audio_token], audio=[audio]) -> model.thinker.get_audio_features(input_features, feature_attention_mask)`
- Exact blocker: The current public qwen_asr wrapper does not expose decoder-compatible audio hidden states and does not accept externally supplied audio hidden states for decoding.
