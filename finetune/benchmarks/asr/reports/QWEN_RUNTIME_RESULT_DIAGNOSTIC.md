# Qwen Runtime Result Diagnostic

Utterance: `1188-133604-0004` (test-clean)
Audio: `/home/hj/Data_Collect_Web/finetune/benchmarks/asr/data/LibriSpeech/test-clean/1188/133604/1188-133604-0004.flac`
Reference: SOME OF THE TOUCHES INDEED WHEN THE TINT HAS BEEN MIXED WITH MUCH WATER HAVE BEEN LAID IN LITTLE DROPS OR PONDS SO THAT THE PIGMENT MIGHT CRYSTALLIZE HARD AT THE EDGE

## Runtime

qwen-asr version: `0.0.6`
qwen-asr file: `/home/hj/miniconda/envs/vigil-two-stage/lib/python3.12/site-packages/qwen_asr/__init__.py`
transformers: `4.57.6`
torch: `2.11.0+cu128` CUDA `12.8`
Selected physical GPU: `6` visible device: `NVIDIA GeForce RTX 3090`

## Official Call

Class: `qwen_asr.inference.qwen3_asr.Qwen3ASRModel`
Signature: `(audio: Union[str, Tuple[numpy.ndarray, int], List[Union[str, Tuple[numpy.ndarray, int]]]], context: Union[str, List[str]] = '', language: Union[str, List[Optional[str]], NoneType] = None, return_time_stamps: bool = False) -> List[qwen_asr.inference.qwen3_asr.ASRTranscription]`
Call used: `model.transcribe(str(audio_path), language=None)`

## Result Shape

Raw result type: `builtins.list`
List/tuple: `True` length `1`
First item type: `qwen_asr.inference.qwen3_asr.ASRTranscription`
Attributes: `language, text, time_stamps`
has `.text`: `True`; has `.transcript`: `False`; has `.language`: `True`
dataclass: `True`; `_asdict`: `False`; `model_dump`: `False`
Language: `English`
Text: Some of the touches, indeed, when the tint has been mixed with much water, have been laid in little drops or ponds, so that the pigment might crystallize hard at the edge.
Safe repr: `ASRTranscription(language='English', text='Some of the touches, indeed, when the tint has been mixed with much water, have been laid in little drops or ponds, so that the pigment might crystallize hard at the edge.', time_stamps=None)`

## Extraction Path

Implement extraction path: `[0].text`

Latency seconds: `2.69786976929754`
Peak GPU memory GB: `3.885563850402832`
