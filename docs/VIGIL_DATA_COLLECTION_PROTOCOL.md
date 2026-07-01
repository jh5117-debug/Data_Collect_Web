# VIGIL Data Collection Protocol

## Prompt Groups

### Prompt 1: VIGIL Only

The participant says only the wake word:

```text
VIGIL
```

Label: positive.

### Prompt 2: Phrase/Sentence + VIGIL

The participant says a natural phrase or sentence ending with or followed by VIGIL:

```text
Hi VIGIL.
What is next, VIGIL?
```

Label: positive.

### Prompt 3: VIGIL + Phrase/Sentence

The participant says VIGIL followed by a command-like phrase:

```text
VIGIL, next.
VIGIL, go back.
```

Label: positive.

### Prompt 4: Negative Examples

The participant says confusing non-trigger words or phrases such as:

```text
visual
visible
digital
individual
residual
```

Label: negative.

## Labels And Transcripts

Positive/negative is the KWS trigger label. It is not the same thing as the ASR transcript.

- Qwen ASR training/evaluation format should use the transcript text only.
- Do not put `positive` or `negative` into the ASR transcript.
- KWS manifests use the trigger label separately from transcript text.
- Prompt 4 negatives may sound similar to VIGIL but should not activate the trigger.

## Supabase Export Workflow

1. Browser records raw audio through MediaRecorder.
2. Frontend sends accepted raw clips to the backend on session submit.
3. Backend stores raw audio in Supabase Storage and metadata in Supabase Postgres.
4. Admin export creates a ZIP containing prompts, metadata, raw audio, and manifests.
5. Offline conversion turns raw browser audio into WAV.
6. Offline Qwen ASR review and manual review create final clean manifests.
7. Training/evaluation code consumes the processed local manifests, not Supabase directly.

## Git Safety

Do not commit raw audio, exports, Supabase dumps, local conversion outputs, participant private names/emails, feature caches, checkpoints, model weights, logs, or predictions.
