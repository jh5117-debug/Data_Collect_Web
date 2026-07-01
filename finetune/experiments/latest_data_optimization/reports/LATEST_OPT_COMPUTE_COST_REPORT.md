# Latest Optimized Compute Cost Report

- Status: `ok`
- Device: `cuda:0`
- Deterministic subset size: `100`
- Qwen ASR transcript latency source: recorded transcript cache (`100` samples)
- Qwen audio encoder forward measured samples: `20`; status `ok`
- Stage2 F1 gain over Qwen exact: `0.21609202499900704`
- Stage2 F1 gain over Stage1-only: `0.011231176996303938`
- Stage1 candidates per hour on outer-test clips: `798.624263116061`
- Current System C uses one Qwen weight copy plus one extra audio-encoder forward per Stage1 candidate.

| Component | n | median ms | p95 ms |
|---|---:|---:|---:|
| stage1_head | 100 | 1.1985101737082005 | 1.242530532181263 |
| stage1_cached_feature_load_plus_head | 100 | 1.218139659613371 | 1.2691989541053772 |
| stage2_head | 100 | 1.4751083217561245 | 1.5391763299703598 |
| stage2_cached_qwen_feature_load_plus_head | 100 | 1.4943983405828476 | 1.5608668327331543 |
| official_openwakeword_feature_extraction | 100 | 29.004013165831566 | 44.705767184495926 |
| qwen_asr_transcript_from_recorded_cache | 100 | 250.64418883994222 | 385.60373708605766 |
| qwen_audio_encoder_forward | 20 | 13.663365971297026 | 16.2650840356946 |
