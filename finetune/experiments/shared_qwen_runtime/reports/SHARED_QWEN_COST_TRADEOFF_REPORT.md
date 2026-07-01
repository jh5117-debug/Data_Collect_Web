# Shared Qwen Cost Tradeoff Report

- Status: `blocked_by_runtime_interface`
- Extra encoder median cost per Stage 1 candidate: `13.663365971297026` ms

| variant | qwen_weight_copies | encoder_forwards | transcript | stage2_score | median_latency_ms | status |
|---|---|---|---|---|---|---|
| Current prototype | 1 | public transcribe path plus one extra get_audio_features call for Stage 2 candidates | yes | yes | 13.663365971297026 | working |
| Shared hidden-state prototype | 1 | 1 only if upstream exposes decoder-compatible hidden-state handoff | blocked | blocked | None | blocked_by_runtime_interface |
