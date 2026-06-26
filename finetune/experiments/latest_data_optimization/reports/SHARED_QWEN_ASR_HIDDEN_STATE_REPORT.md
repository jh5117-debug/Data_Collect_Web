# Shared Qwen-ASR Hidden-State Report

## Status

- Status: `blocked_by_runtime_interface`
- Blocker: The installed qwen_asr public transcribe API accepts raw audio and returns ASRTranscription text, while the Stage 2 feature path calls model.thinker.get_audio_features. The public API does not expose reusable audio encoder hidden states and does not accept externally supplied hidden states for decoding. Signature inspected: (self, audio: Union[str, Tuple[numpy.ndarray, int], List[Union[str, Tuple[numpy.ndarray, int]]]], context: Union[str, List[str]] = '', language: Union[str, List[Optional[str]], NoneType] = None, return_time_stamps: bool = False) -> List[qwen_asr.inference.qwen3_asr.ASRTranscription]

## Evidence

- qwen-asr version: `0.0.6`
- Source file inspected: `/home/hj/miniconda/envs/vigil-two-stage/lib/python3.12/site-packages/qwen_asr/inference/qwen3_asr.py`
- Public transcribe signature: `(self, audio: Union[str, Tuple[numpy.ndarray, int], List[Union[str, Tuple[numpy.ndarray, int]]]], context: Union[str, List[str]] = '', language: Union[str, List[Optional[str]], NoneType] = None, return_time_stamps: bool = False) -> List[qwen_asr.inference.qwen3_asr.ASRTranscription]`
- Public methods exposing hidden/features: `[]`
- `transcribe` uses model generation internally: `True`

## Call Counter Diagnostic

- Model load count: `1`
- Counts after public transcribe: `{'transcribe': 1, 'generate': 1, 'thinker_forward': 6, 'get_audio_features': 1}`
- Counts after separate Stage 2 feature extraction: `{'transcribe': 1, 'generate': 1, 'thinker_forward': 6, 'get_audio_features': 2}`
- Transcript extraction path: `$[0].text`
- Stage 2 feature path: `model.thinker.get_audio_features`
- Hidden shape: `[26, 2048]`

## Cost Table

| Variant | Qwen copies | Encoder forwards | Transcript available? | Stage2 score available? | Median latency | Status |
|---|---:|---:|---:|---:|---:|---|
| Current prototype | 1 | extra encoder forward per Stage 1 candidate | yes | yes | 13.663365971297026 ms extra encoder median | working |
| Shared hidden-state prototype | 1 | 1 only if upstream exposes handoff | no verified handoff | no verified handoff | None | blocked_by_runtime_interface |

## Professor Wording

The current public Qwen ASR wrapper does not expose a reusable hidden-state handoff. Therefore the current prototype still needs an extra encoder forward for Stage 2 candidates. The measured median extra cost is around 13.66 ms per Stage 1 candidate from the latest compute report.
