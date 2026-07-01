# Stage 1 openWakeWord Structure Report

## Simple English Summary

Stage 1 is a lightweight wake-word candidate detector. It is not Qwen, not LoRA, and it does not produce a transcript. It reads 16 kHz audio, uses the official openWakeWord audio feature extractor as a frozen front-end, and trains only a small VIGIL-specific PyTorch head. The head outputs `p1`, the probability that this audio window should become a VIGIL candidate. The candidate rule is `p1 >= theta_1`.

## Technical Summary

- Input: 16 kHz mono audio windows.
- Preprocessing: official openWakeWord `AudioFeatures` computes mel features and 96-dimensional speech embeddings with ONNX Runtime.
- Frozen component: openWakeWord `melspectrogram.onnx` and `embedding_model.onnx` assets are used as feature extractors; they are not trained by this project.
- Trainable component: `Stage1GRUClassifier`, implemented as `LayerNorm -> 2-layer GRU -> Linear`.
- Loss: weighted `BCEWithLogitsLoss` through the local wrapper used by `train_stage1.py`.
- Output: one trigger logit per window; `p1 = sigmoid(logit)`.
- Role: high-recall candidate detector that cheaply reduces how often Stage 2 must run.
- Difference from Qwen: Qwen3-ASR-1.7B produces transcript and high-dimensional audio states; Stage 1 has only a 56k trainable head and no decoder.

## Architecture Diagram

```text
Microphone 16 kHz audio
    -> openWakeWord preprocessing
    -> frozen openWakeWord melspectrogram ONNX
    -> frozen openWakeWord embedding ONNX
    -> embeddings [T, 96]
    -> trainable LayerNorm + 2-layer GRU + Linear
    -> p1 candidate score
    -> candidate if p1 >= theta_1
```

## Parameter Count

| Component | Frozen? | Parameters | Trainable? | Notes |
|---|---:|---:|---:|---|
| `melspectrogram.onnx` | yes | N/A (ONNX runtime asset) | no | official openWakeWord feature asset, 1062.5 KiB |
| `embedding_model.onnx` | yes | N/A (ONNX runtime asset) | no | official openWakeWord feature asset, 1297.0 KiB |
| Stage 1 LayerNorm | no | 192 | yes | normalizes 96-d openWakeWord embeddings |
| Stage 1 GRU | no | 56064 | yes | 2-layer unidirectional GRU, hidden size 64 |
| Stage 1 Linear | no | 65 | yes | maps final GRU state to one trigger logit |
| Stage 1 total head | no | 56321 | yes | verified from PyTorch module |

## Latency

| Component | n | Median ms | p95 ms | Source |
|---|---:|---:|---:|---|
| official openWakeWord feature extraction | 100 | 29.004013165831566 | 44.705767184495926 | latest compute report |
| Stage 1 head | 100 | 1.1985101737082005 | 1.242530532181263 | latest compute report |
| cached feature load + Stage 1 head | 100 | 1.218139659613371 | 1.2691989541053772 | latest compute report |
| full Stage 1 component-sum estimate | - | 30.202523339539766 | 45.94829771667719 | openWakeWord feature + head sum |

## Comparison To Continuous Qwen ASR

| Component | Primary role | Produces transcript? | Approx parameters | Runtime role |
|---|---|---:|---:|---|
| Stage 1 VIGIL head | cheap candidate detector | no | 56321 trainable head params | runs before Stage 2 |
| Qwen3-ASR-1.7B | continuous clinical ASR and Stage 2 audio features | yes | 2,038,052,480 frozen params | main transcript branch and verifier features |

Stage 1 is designed to be high-recall and lightweight. Stage 2 then uses frozen Qwen audio features to reject most false candidates.
