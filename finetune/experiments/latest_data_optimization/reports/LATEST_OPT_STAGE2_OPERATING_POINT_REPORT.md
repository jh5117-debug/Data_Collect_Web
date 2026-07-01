# Latest Optimized Stage 2 Operating Points

Thresholds are selected from fold validation predictions only. Outer-test rows are used once for reporting.

| Variant | Target recall | Dev recall | Dev FPR | Test recall | Test FPR | Test F1 |
|---|---:|---:|---:|---:|---:|---:|
| stage2_bce_supcon | 0.85 | 0.950909 | 0.000000 | 0.940785 | 0.004703 | 0.967427 |
| stage2_bce_supcon | 0.90 | 0.950909 | 0.000000 | 0.940785 | 0.004703 | 0.967427 |
| stage2_bce_supcon | 0.92 | 0.950909 | 0.000000 | 0.940785 | 0.004703 | 0.967427 |
| stage2_bce_supcon | 0.95 | 0.953576 | 0.025000 | 0.946379 | 0.007867 | 0.969015 |
| stage2_bce | 0.85 | 0.923325 | 0.001786 | 0.909746 | 0.006896 | 0.949111 |
| stage2_bce | 0.90 | 0.942909 | 0.010555 | 0.925677 | 0.012253 | 0.956115 |
| stage2_bce | 0.92 | 0.953576 | 0.012340 | 0.941261 | 0.015505 | 0.963743 |
| stage2_bce | 0.95 | 0.953576 | 0.012340 | 0.941261 | 0.015505 | 0.963743 |

## Meeting Modes

- Safe mode: `stage2_bce_supcon`, top_k `1`, method `threshold_only`.
- Balanced mode: `stage2_bce_supcon`, top_k `1`, method `threshold_only`, dev F1 `0.9748470428280082`, test F1 `0.9675190048375951`.
- High-recall mode: `stage2_bce_supcon`, top_k `1`, method `threshold_only`.
