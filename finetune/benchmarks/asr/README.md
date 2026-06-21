# LibriSpeech Qwen ASR Benchmark

This benchmark measures the unchanged `Qwen/Qwen3-ASR-1.7B` model on the official LibriSpeech `test-clean` and `test-other` evaluation splits.

It is separate from the VIGIL two-stage wake-word experiment. It does not use openWakeWord, the VIGIL Stage 1 model, the VIGIL Stage 2 verifier, or any fine-tuning. The goal is a reusable baseline so we can compare future ASR model changes against the same public test set.

## Layout

```text
finetune/benchmarks/asr/
  configs/
  data/                 # generated, ignored
  downloads/            # generated, ignored
  logs/                 # generated, ignored
  manifests/            # generated JSONL, ignored
  runs/                 # generated benchmark outputs, ignored
  scripts/
  src/
  tests/
```

Generated data, predictions, logs and runs are ignored by Git.

## One-Time Setup

From a normal HAL SSH shell:

```bash
cd /home/hj/Data_Collect_Web
bash finetune/benchmarks/asr/scripts/bootstrap_asr_benchmark.sh
```

The scripts prefer:

```text
/home/hj/miniconda/envs/vigil-two-stage
```

The Hugging Face and PyTorch caches are kept under:

```text
/home/hj/Data_Collect_Web/finetune/cache/
```

## Download And Prepare LibriSpeech

```bash
cd /home/hj/Data_Collect_Web
bash finetune/benchmarks/asr/scripts/download_librispeech_eval.sh
PATH=/home/hj/miniconda/envs/vigil-two-stage/bin:$PATH \
  python finetune/benchmarks/asr/scripts/prepare_librispeech_manifest.py \
  --validate-audio \
  --expected-counts
```

The expected official counts are:

```text
test-clean: 2620
test-other: 2939
total: 5559
```

The manifest rows are deterministic and include absolute audio paths, reference text, duration, speaker/chapter IDs, and audio SHA-256.

## Smoke Benchmark

Use exactly one local RTX 3090. Example with physical GPU 6:

```bash
cd /home/hj/Data_Collect_Web
bash finetune/benchmarks/asr/scripts/run_librispeech_smoke.sh 6
```

Run it in tmux so it survives SSH disconnects:

```bash
tmux new -d -s librispeech_qwen_smoke \
  'cd /home/hj/Data_Collect_Web && bash finetune/benchmarks/asr/scripts/run_librispeech_smoke.sh 6'
```

Monitor:

```bash
tail -f "$(ls -1t /home/hj/Data_Collect_Web/finetune/benchmarks/asr/logs/librispeech_smoke_*_gpu6.log | head -1)"
```

The smoke set is 32 `test-clean` utterances plus 32 `test-other` utterances. Any smoke report must be treated as:

```text
SMOKE SUBSET — NOT FULL LIBRISPEECH RESULT
```

## Full Benchmark

```bash
cd /home/hj/Data_Collect_Web
bash finetune/benchmarks/asr/scripts/run_librispeech_full.sh \
  6 \
  Qwen/Qwen3-ASR-1.7B \
  qwen3_asr_1_7b_baseline
```

tmux:

```bash
tmux new -d -s librispeech_qwen_full \
  'cd /home/hj/Data_Collect_Web && bash finetune/benchmarks/asr/scripts/run_librispeech_full.sh 6 Qwen/Qwen3-ASR-1.7B qwen3_asr_1_7b_baseline'
```

## Strict Runtime Rules

The launchers fail instead of silently falling back when:

- `nvidia-smi` is unavailable
- the selected physical GPU is not an RTX 3090
- PyTorch does not see exactly one CUDA device
- the Qwen model is not `Qwen/Qwen3-ASR-1.7B` unless a future run intentionally passes another model
- the runner would need CPU execution

The runner uses `model.eval()` and `torch.inference_mode()`. Qwen weights are not updated.

## Outputs

Each run writes:

```text
config_resolved.yaml
environment.json
model_info.json
predictions.jsonl
failures.jsonl
progress.json
metrics_raw.json
metrics_normalized.json
per_split_metrics.json
per_speaker_metrics.json
error_analysis.csv
reproducibility.json
FINAL_REPORT.md
```

Predictions are appended incrementally and fsynced. Re-running with `--resume` skips successful utterances.

## Comparing Runs

```bash
PATH=/home/hj/miniconda/envs/vigil-two-stage/bin:$PATH \
  python finetune/benchmarks/asr/scripts/compare_asr_runs.py \
  --baseline /path/to/baseline_run \
  --candidate /path/to/candidate_run \
  --output /path/to/comparison/librispeech_compare
```

If both runs contain exactly the same successful utterance IDs, the comparison is formal. Otherwise it is clearly marked as intersection-only.

## How This Connects To VIGIL

This benchmark answers a different question from the VIGIL trigger pipeline:

- LibriSpeech benchmark: "How good is the unchanged Qwen ASR model on public English ASR?"
- VIGIL trigger pipeline: "Can we detect the VIGIL trigger before running the full ASR path?"

The benchmark gives us a public, reproducible ASR baseline before we change or fine-tune any model.

## Troubleshooting

- CUDA not visible: run from a normal HAL SSH shell, not from the Codex sandbox.
- First model load is slow: Qwen weights may be downloading into the local Hugging Face cache.
- Out of memory: verify only one process is using the selected RTX 3090 and keep batch size at 1.
- Missing LibriSpeech files: re-run `download_librispeech_eval.sh` and then `prepare_librispeech_manifest.py --validate-audio --expected-counts`.
- Partial run: re-run the same command; successful rows in `predictions.jsonl` are skipped when resume mode is enabled by the launcher.

Follow the official LibriSpeech/OpenSLR license terms for the downloaded dataset.
