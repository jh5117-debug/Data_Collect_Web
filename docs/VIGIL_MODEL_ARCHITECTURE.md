# VIGIL Model Architecture

## Overall

```text
Microphone
  -> continuous Qwen ASR
       -> transcript / report
  -> VIGIL trigger branch
       -> Stage 1
       -> Stage 2
       -> trigger
```

The ASR branch is for transcription. The VIGIL branch is for wake-word detection.

## Stage 1 Candidate Detector

```text
16 kHz audio
  -> Mel-spectrogram
  -> frozen openWakeWord speech embedding backbone
  -> [T, 96] embeddings
  -> LayerNorm
  -> 2-layer GRU
  -> Linear
  -> p1 candidate probability
```

Stage 1 is intended to be high recall. It uses the frozen openWakeWord feature extractor and a trainable head with `56,321` trainable parameters. The measured full Stage 1 median latency is about `30.203 ms`.

## Stage 2 Verifier

```text
candidate audio
  -> frozen Qwen audio encoder
  -> Qwen audio features
  -> LayerNorm
  -> Linear projection
  -> temporal pooling
  -> embedding layer
  -> 128D normalized embedding z
  -> final classifier
  -> p2 trigger score
```

Stage 2 is the main verifier and the main location for doctor-specific similarity/adaptation. Qwen parameters are frozen.

## Few-Shot Personalization

```text
doctor 3/5 VIGIL clips
  -> Stage 2 embeddings
  -> doctor-specific prototype
  -> cosine(candidate, prototype)
  -> Stage 2 score calibration
```

The few-shot cosine method uses Stage 2 embeddings, not Stage 1 embeddings. Target doctor negative clips are not used for adaptation.

## Shared Qwen Runtime

The desired future path is to reuse Qwen hidden states from continuous ASR for Stage 2. The current public runtime path does not expose the needed verified hidden-state reuse, so Stage 2 currently performs its own frozen Qwen encoder forward.
