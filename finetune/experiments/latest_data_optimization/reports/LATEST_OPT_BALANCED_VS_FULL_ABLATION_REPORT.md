# Latest Optimized Balanced Versus Full-Data Ablation

- Status: `partial_full_unbalanced_heads_only`
- Full-unbalanced clips/windows evaluated by heads: `1597` / `1615`
- Full-unbalanced Qwen exact status: full-unbalanced Qwen exact was not rerun; only balanced transcript cache exists for 1346 clips

| Dataset | Method | Recall | FPR | Precision | F1 | Participant-macro F1 |
|---|---|---:|---:|---:|---:|---:|
| Balanced max-100 | Stage1 only | 0.9556451612903226 | 0.053156146179401995 | 0.9569313593539704 | 0.9562878278412912 | None |
| Balanced max-100 | Selected Stage2 | 0.9408602150537635 | 0.0049833887043189366 | 0.9957325746799431 | 0.9675190048375951 | None |
| Full unbalanced | Stage1 only | 0.9590254706533776 | 0.05187319884726225 | 0.9600886917960089 | 0.9595567867036011 | 0.9224760100829628 |
| Full unbalanced | Selected Stage2 | 0.946843853820598 | 0.007204610951008645 | 0.9941860465116279 | 0.9699376063528077 | 0.9376236411304498 |

Primary result remains balanced max-100; this ablation is partial because full Qwen exact transcript cache was not generated.
