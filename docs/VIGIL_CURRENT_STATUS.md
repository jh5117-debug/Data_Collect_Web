# VIGIL Current Status

## Website And Recorder

The VIGIL Recorder web app is a browser-based collection tool with participant login, consent, prompt recording, playback, submit, admin summary, and export. Production collection uses Vercel for the frontend, Render for the backend, and Supabase Postgres plus Supabase Storage for raw browser uploads.

## Supabase, Render, And Vercel

- Supabase stores account/session/clip metadata and raw uploaded audio objects.
- Render hosts the FastAPI backend.
- Vercel hosts the React frontend and admin page.
- Online QC remains lightweight. WAV conversion, Qwen ASR review, semantic review, and final manifests are offline steps.

## Data Collection Prompt Groups

- P1: VIGIL only, positive.
- P2: phrase or sentence plus VIGIL, positive.
- P3: VIGIL plus phrase or sentence, positive.
- P4: negative examples such as visual/visible/digital, negative.

## Model Status

- Stage 1 is a high-recall candidate detector using a frozen openWakeWord feature extractor and a trainable LayerNorm -> 2-layer GRU -> Linear head.
- Stage 2 is a verifier using frozen Qwen audio features and a small trigger head.
- Qwen ASR and openWakeWord feature extractor weights are not fine-tuned.

## Current Metrics

| Result | Recall | FPR | Precision | F1 |
|---|---:|---:|---:|---:|
| Base Qwen exact keyword | 0.6189 | 0.0000 | 1.0000 | 0.7514 |
| Optimized two-stage | 0.9409 | 0.0050 | 0.9957 | 0.9675 |
| 5-shot Stage 2 cosine prototype | 0.95510 | 0.01176 | 0.98658 | 0.97059 |

LibriSpeech frozen-Qwen ASR preservation:

| Split | WER |
|---|---:|
| test-clean | 1.8411% |
| test-other | 3.6662% |
| combined | 2.7516% |

## Few-Shot Status

The clean target-doctor ablation shows the best few-shot method is `stage2_cosine_prototype` at 5-shot: F1 `0.97059`, recall `0.95510`, FPR `0.01176`, delta F1 `+0.04195`.

## Demo Status

The local HAL browser assistant demo supports name profile, onboarding recordings, prototype calibration, assistant listening, rolling transcript, and VIGIL trigger state. It is local-only and has no downstream LLM/VQA response implementation.

## Shared Qwen Status

The current system uses the same frozen Qwen weights conceptually, but shared hidden-state reuse from public `qwen_asr` is not verified. Stage 2 currently still needs an extra Qwen encoder forward.

## Next Actions

- Package a minimal Python trigger API for ROS 2 integration.
- Decide whether to keep Stage 2 cosine prototype or bounded positive-bias calibration as the default personalization path.
- Continue investigating Qwen hidden-state sharing only if runtime access allows it.
- Keep raw audio, private local data, checkpoints, and logs out of Git.
