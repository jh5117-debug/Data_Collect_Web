# Latest Compute/Accuracy Tradeoff

- Status: `partial_head_benchmark`
- Device: `cuda:0`
- Stage 1 parameters: `{'total': 56321, 'trainable': 56321, 'frozen': 0}`
- Stage 2 parameters: `{'total': 561922, 'trainable': 561922}`

| Component | Median ms | P95 ms | Peak allocated GB |
|---|---:|---:|---:|
| stage1_head | 0.8454932831227779 | 1.0746415704488745 | 0.012576580047607422 |
| stage2_head | 0.45637600123882294 | 0.48366887494921684 | 0.013738632202148438 |

Full Qwen ASR/audio-encoder forward latency remains a limitation.
