# Qwen ASR Format Report

- Qwen ASR rows contain only `audio` and `text`.
- `text` is `language English<asr_text>` plus the transcript.
- Labels, prompt groups, phrase IDs, participant aliases, and splits are written only to KWS manifests.

| Split | Rows |
|---|---:|
| train | 1177 |
| val | 270 |
| test | 189 |
