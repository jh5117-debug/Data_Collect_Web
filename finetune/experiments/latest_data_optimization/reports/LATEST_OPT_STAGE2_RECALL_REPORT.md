# Latest Stage 2 Recall/FPR Optimization

Development-only search evaluated threshold-only, top-K expansion, BCE versus SupCon, and Stage1/Stage2 logit fusion.

- Selected variant: `stage2_bce_supcon`
- Selected method: `threshold_only`
- Selected top_k: `1`
- Fusion a/b/logit: `1.0` / `0.0` / `False`
- Development recall/FPR/F1: `0.9509283819628647` / `0.0` / `0.9748470428280082`
- Outer-test recall/FPR/F1: `0.9408602150537635` / `0.0049833887043189366` / `0.9675190048375951`
- Qwen exact outer-test F1: `0.7514269798385881`
- Stage1-only recomputed outer-test F1: `0.9562878278412912`

The selected configuration is deployment-safe if its development FPR is <= 0.02 and outer-test FPR is reported without tuning.
