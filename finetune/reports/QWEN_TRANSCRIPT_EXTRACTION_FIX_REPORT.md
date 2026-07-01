# Qwen Transcript Extraction Fix Report

## Bug

The previous Qwen transcript extractors accepted arbitrary `str(result)` output when a returned object was not a string, mapping, list, or tuple. The installed Qwen runtime returns `list[qwen_asr.inference.qwen3_asr.ASRTranscription]`, where the real transcript is the `.text` attribute. The old path stored structured object repr strings such as `ASRTranscription(language=..., text=..., time_stamps=None)`.

## Evidence

- Old run audit: `/home/hj/Data_Collect_Web/finetune/benchmarks/asr/reports/QWEN_OLD_RUN_EXTRACTION_AUDIT.md`
- Runtime diagnostic: `/home/hj/Data_Collect_Web/finetune/benchmarks/asr/reports/QWEN_RUNTIME_RESULT_DIAGNOSTIC.md`
- Old full run: `/home/hj/Data_Collect_Web/finetune/benchmarks/asr/runs/20260624_090118_qwen3_asr_1_7b_baseline_full`
- Old full predictions: 5559 rows, 5559 structured object repr hypotheses.

The actual Qwen return shape is:

```text
type(raw_result): builtins.list
len(raw_result): 1
type(raw_result[0]): qwen_asr.inference.qwen3_asr.ASRTranscription
text field: raw_result[0].text
extraction path: $[0].text
```

## Code Fix

Fix commit: `538f646 Fix structured Qwen ASR transcript extraction`

The shared extractor is:

```text
finetune/src/vigil_two_stage/qwen_text_result.py
```

It supports strings, bytes, mappings, sequences, object attributes, dataclasses, named tuples, and Pydantic-style `model_dump()` results. It records `text`, `extraction_path`, and `result_type`; it rejects empty outputs and clear Python object repr patterns. It never treats arbitrary `str(result)` as a successful transcript.

Updated call sites:

```text
finetune/benchmarks/asr/src/qwen_runner.py
finetune/benchmarks/asr/scripts/run_qwen_librispeech.py
finetune/scripts/run_qwen_text_baseline.py
finetune/demo/inference.py
```

## Tests

Added:

```text
finetune/tests/test_qwen_text_result.py
finetune/benchmarks/asr/tests/test_qwen_transcript_call_sites.py
```

Coverage includes plain strings, bytes, nested mappings, object attributes, dataclasses, named tuples, Pydantic-like objects, empty sequences, unsupported objects, object repr rejection, cycle detection, extraction-path recording, and the Qwen-style `FakeASRResult` regression.

## Invalid Historical Result

The old full LibriSpeech run is invalid for scientific reporting:

```text
/home/hj/Data_Collect_Web/finetune/benchmarks/asr/runs/20260624_090118_qwen3_asr_1_7b_baseline_full
```

Its old normalized WER `0.40069005613854497` must not be cited as ASR performance because it was scored against structured dataclass repr strings.

Invalid notices:

```text
/home/hj/Data_Collect_Web/finetune/benchmarks/asr/runs/20260624_090118_qwen3_asr_1_7b_baseline_full/INVALID_RESULT_NOTICE.md
/home/hj/Data_Collect_Web/finetune/benchmarks/asr/runs/20260624_090118_qwen3_asr_1_7b_baseline_full/invalid_result_notice.json
```

## Corrected Smoke Result

Run:

```text
/home/hj/Data_Collect_Web/finetune/benchmarks/asr/runs/20260624_185009_qwen3_asr_1_7b_fixed_text_extraction_smoke_smoke
```

Measured values:

```text
successful predictions: 64
failures: 0
duplicate IDs: 0
normalized WER: 0.037165082108902334
test-clean normalized WER: 0.015455950540958269
test-other normalized WER: 0.06470588235294118
normalized CER: 0.013495934959349594
SER: 0.3125
exact-match rate: 0.6875
malformed hypotheses: 0
extraction path: $[0].text
result type: qwen_asr.inference.qwen3_asr.ASRTranscription
```

## Corrected Full Result

Run:

```text
/home/hj/Data_Collect_Web/finetune/benchmarks/asr/runs/20260624_185419_qwen3_asr_1_7b_fixed_text_extraction_baseline_full
```

Measured values:

```text
successful predictions: 5559
failures: 0
duplicate IDs: 0
combined normalized WER: 0.02751646508258752
test-clean normalized WER: 0.018411442483262326
test-other normalized WER: 0.03666201784383776
raw WER: 0.9862560642019081
normalized CER: 0.009904616200632524
substitutions: 2337
deletions: 330
insertions: 220
reference words: 104919
sentence error rate: 0.29807519338010435
exact-match rate: 0.7019248066198956
total audio duration: 38682.05075s
total inference time: 4587.473365384154s
mean latency: 0.8252335609613517s
median latency: 0.694949496537447s
RTF: 0.1185943680967885
peak GPU memory: 4.068508625030518 GB
malformed hypotheses: 0
extraction path: $[0].text
result type: qwen_asr.inference.qwen3_asr.ASRTranscription
```

## VIGIL Baseline Impact

Corrected VIGIL Qwen baseline:

```text
/home/hj/Data_Collect_Web/finetune/runs/20260624_075127_0fad4c7828149099_full/baseline_qwen_exact_clip_fixed_text_extraction
```

Measured values:

```text
n: 93
precision: 1.0
recall: 0.6862745098039216
false-positive rate: 0.0
F1: 0.813953488372093
P1 recall: 0.6666666666666666
P2 recall: 0.7272727272727273
P3 recall: 0.6470588235294118
P4 false-positive rate: 0.0
Qwen trainable parameters: 0
peak GPU memory: 3.8581886291503906 GB
extraction path: $[0].text
result type: qwen_asr.inference.qwen3_asr.ASRTranscription
```

The old recall `0.6862745098039216` did not change because the old object repr strings still contained the transcript text. The old transcript artifacts remain invalid.

## Demo Status

The existing browser/upload demo was launched after the benchmark on physical GPU 6:

```text
tmux session: vigil_live_demo
python pid: 1099372
url: http://127.0.0.1:7860
log: /home/hj/Data_Collect_Web/finetune/demo/logs/vigil_demo_20260624_205136_gpu6.log
```

File-upload validation report:

```text
/home/hj/Data_Collect_Web/finetune/demo/reports/VIGIL_LIVE_DEMO_FILE_UPLOAD_VALIDATION.md
```

The validation-selected `stage2_bce` variant remains the default. Browser microphone capture was not human-validated.

## Remaining Limitations

- The corrected LibriSpeech numbers are for this installed backend, model revision, decoding path, and normalization; they are not claimed to exactly reproduce an upstream published result.
- Raw WER is punctuation/case-sensitive and should not be treated as the normalized ASR benchmark number.
- HAL file-upload validation does not prove the user's laptop browser microphone path.
