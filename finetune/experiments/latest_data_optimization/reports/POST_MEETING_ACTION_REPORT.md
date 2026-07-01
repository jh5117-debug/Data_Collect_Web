# Post-Meeting VIGIL Action Report

## 1. What The Professor Asked

- Frame this as a VIGIL voice trigger module inside an ASR-based clinical workflow.
- Explain Stage 1 openWakeWord structure clearly.
- Redo few-shot onboarding as target-doctor-only personalization.
- Integrate the corrected LibriSpeech benchmark for the frozen continuous Qwen ASR branch.
- Try to reduce Stage 2 cost through shared Qwen-ASR hidden-state reuse, or document the blocker.

## 2. Corrected Clinical Workflow

```text
Microphone audio
  -> Continuous Qwen3-ASR branch
       -> full doctor-patient transcript for the medical report
  -> Parallel VIGIL trigger branch
       -> Stage 1 openWakeWord candidate detector
       -> Stage 2 frozen-Qwen-feature verifier
       -> enter assistant / VQA state when VIGIL is detected
```

## 3. Stage 1 openWakeWord Structure

- Stage 1 is not Qwen and not LoRA.
- It is a lightweight KWS front-end.
- Input is 16 kHz audio.
- Frozen front-end: official openWakeWord melspectrogram and embedding ONNX assets.
- Trainable head: `LayerNorm -> 2-layer GRU -> Linear`.
- Trainable Stage 1 head parameters: `56321`.
- openWakeWord feature extraction median/p95: `29.0040` / `44.7058` ms.
- Stage 1 head median/p95: `1.1985` / `1.2425` ms.
- Output is `p1`, a candidate probability. Stage 1 does not produce text.

## 4. Latest VIGIL Trigger Result

| Method | Recall | FPR | Precision | F1 |
|---|---:|---:|---:|---:|
| Base Qwen exact keyword | 0.6189 | 0.0000 | 1.0000 | 0.7514 |
| Optimized two-stage VIGIL trigger | 0.9409 | 0.0050 | 0.9957 | 0.9675 |

The VIGIL trigger branch improves wake-word F1 over exact transcript keyword matching while keeping false positives low.

## 5. Target-Doctor Few-Shot Result

| Setting | Recall | FPR | Precision | F1 | Delta F1 | Improved | Degraded | Unchanged |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 3-shot | 0.9494 | 0.0117 | 0.9880 | 0.9683 | 0.0423 | 13 | 2 | 18 |
| 5-shot | 0.9558 | 0.0118 | 0.9866 | 0.9710 | 0.0423 | 12 | 2 | 17 |

Conclusion: target-doctor-only support-based onboarding improved safely: `True`. The query is now only the target doctor's remaining clips.

## 6. LibriSpeech Benchmark For Frozen Qwen ASR

| Qwen module | Qwen updated? | Benchmark | test-clean WER | test-other WER | Combined WER |
|---|---:|---|---:|---:|---:|
| Continuous frozen Qwen3-ASR-1.7B | No | LibriSpeech | 1.8411% | 3.6662% | 2.7516% |

This benchmark measures general ASR ability of the frozen continuous Qwen branch. It is separate from VIGIL trigger recall/FPR. If Qwen is updated in the future, LibriSpeech must be rerun on that updated Qwen.

## 7. Shared-Qwen Status

- Status: `blocked_by_runtime_interface`
- Extra Qwen encoder cost median: `13.6634` ms per Stage 1 candidate.
- Exact blocker: The installed qwen_asr public transcribe API accepts raw audio and returns ASRTranscription text, while the Stage 2 feature path calls model.thinker.get_audio_features. The public API does not expose reusable audio encoder hidden states and does not accept externally supplied hidden states for decoding. Signature inspected: (self, audio: Union[str, Tuple[numpy.ndarray, int], List[Union[str, Tuple[numpy.ndarray, int]]]], context: Union[str, List[str]] = '', language: Union[str, List[Optional[str]], NoneType] = None, return_time_stamps: bool = False) -> List[qwen_asr.inference.qwen3_asr.ASRTranscription]

## 8. Next Step

- Validate the target-doctor onboarding gain on future blind doctors before treating it as a locked deployment claim.
- Test stronger adaptation methods only if they still avoid target negatives and keep Qwen/openWakeWord frozen.
- For shared Qwen, request or implement an upstream interface that returns decoder-compatible audio hidden states and accepts those states for ASR decoding.
- Keep the current extra encoder-forward prototype until one-forward reuse is proven by call counters.

## Speaking Script

In simple English: VIGIL is a voice trigger module for a clinical ASR workflow. Qwen keeps transcribing the full conversation. In parallel, Stage 1 cheaply proposes possible VIGIL events, and Stage 2 verifies them using frozen Qwen audio features. Qwen is not fine-tuned, so the LibriSpeech result is the frozen ASR branch. The shared-Qwen experiment shows that the current wrapper does not expose hidden states for reuse, so Stage 2 still needs an extra encoder forward.

中文备注：这个系统不是商业语音助手产品，而是临床 ASR 流程里的 VIGIL 触发模块。Qwen 连续转写完整对话；VIGIL 分支并行检测唤醒词。Stage 1 很小，只做候选检测；Stage 2 用冻结的 Qwen 音频特征做验证。当前没有微调 Qwen，所以 LibriSpeech 衡量的是冻结 Qwen ASR 的通用转写能力。shared-Qwen 目前被公开接口阻塞，还不能证明一次 encoder forward 同时服务 ASR 和 Stage 2。
