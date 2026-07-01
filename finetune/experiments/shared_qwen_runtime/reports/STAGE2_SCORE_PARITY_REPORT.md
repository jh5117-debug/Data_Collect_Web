# Stage 2 Score Parity Report

- Status: `blocked_by_runtime_interface`
- Fixed VIGIL subset size: `25`
- Separate feature median latency: `15.986877493560314` ms
- Shared Stage 2 score parity: `blocked`
- Max absolute score difference: `None`
- Blocker: The current public qwen_asr wrapper does not expose decoder-compatible audio hidden states and does not accept externally supplied audio hidden states for decoding.

The separate production Stage 2 feature path was measured. A shared score was not computed because the runtime did not expose reusable decoder-compatible hidden states.
