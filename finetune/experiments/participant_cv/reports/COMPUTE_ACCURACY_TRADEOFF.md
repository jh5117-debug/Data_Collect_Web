# Compute Accuracy Trade-Off

| System | Qwen Copies | Extra Encoder Forward | Recall | FPR | F1 | Peak VRAM | Median Latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen transcript keyword | 1 | 0 | 0.54640630845742 | 0.0 | 0.6789141371086982 | 3.8581886291503906 | 0.27646801294758916 |
| Stage 1 only + continuous ASR | 1 | 0 | 0.9055543469408414 | 0.07431277242252153 | 0.9246570588270571 | None | None |
| Two-stage BCE + continuous ASR | 1 | 1 | 0.7652259275562515 | 0.004651162790697674 | 0.8629426150555287 | None | None |
| Shared hidden-state feasibility | 1 | None | None | None | None | None | None |

Measured Qwen transcript median latency: `0.276468` seconds over `1026` balanced clips.
Stage 2 improves FPR over Stage 1-only but reduces recall in the current validation-selected BCE run. SupCon has higher recall than BCE at a higher FPR, but validation-selected method chose BCE.
Unavailable latency cells are left null rather than estimated.
