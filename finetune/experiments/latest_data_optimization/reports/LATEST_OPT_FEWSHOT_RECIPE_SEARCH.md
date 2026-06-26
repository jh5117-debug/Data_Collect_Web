# Latest Optimized Few-Shot Recipe Search

- Device: `cuda:0`
- Recipe selection uses each fold's validation participants only.
- Target support uses positive clips only; target negatives and query positives are not used for adaptation.
- Support-based recipe selected: `False`
- Claim: Real support-based onboarding was implemented and evaluated, but no safe improvement was found on the latest dataset.

| Fold | Selected method | Reason or delta |
|---:|---|---:|
| 0 | no_adaptation_zero_shot_fallback | no_safe_support_based_f1_improvement |
| 1 | no_adaptation_zero_shot_fallback | no_safe_support_based_f1_improvement |
| 2 | no_adaptation_zero_shot_fallback | no_safe_support_based_f1_improvement |
| 3 | no_adaptation_zero_shot_fallback | no_safe_support_based_f1_improvement |
| 4 | no_adaptation_zero_shot_fallback | no_safe_support_based_f1_improvement |
