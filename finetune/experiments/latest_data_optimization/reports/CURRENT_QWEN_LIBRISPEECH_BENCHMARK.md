# Current Qwen LibriSpeech Benchmark

This benchmark evaluates the frozen continuous Qwen3-ASR branch in the VIGIL clinical workflow architecture. It does not evaluate the two-stage trigger detector.

The current method does not fine-tune Qwen. Qwen weights are frozen, so LibriSpeech measures the unchanged general ASR branch. VIGIL recall/FPR is a separate wake-word trigger metric. If future Qwen LoRA or SFT is performed, LibriSpeech must be rerun on that fine-tuned Qwen checkpoint.

| Qwen module | Qwen updated? | Benchmark | test-clean WER | test-other WER | Combined WER |
|---|---:|---|---:|---:|---:|
| Continuous frozen Qwen3-ASR-1.7B | No | LibriSpeech | 1.8411% | 3.6662% | 2.7516% |

## Verification

- Run: `finetune/benchmarks/asr/runs/20260624_185419_qwen3_asr_1_7b_fixed_text_extraction_baseline_full`
- Status: `verified`
- Successful predictions: `5559`
- Failures: `0`
- Malformed/object-repr hypotheses: `0`
- Extraction path: `['$[0].text']`
- Result type: `['qwen_asr.inference.qwen3_asr.ASRTranscription']`

The old approximately 40% WER run is invalid because it stored `ASRTranscription(...)` object representations instead of the `.text` transcript. It is not used here.
