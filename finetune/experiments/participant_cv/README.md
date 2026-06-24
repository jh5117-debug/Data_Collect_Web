# VIGIL Participant CV And Few-Shot Experiments

This package contains the participant-disjoint evaluation protocol requested for the VIGIL detector.

Primary reports:

- `reports/PARTICIPANT_DATA_AUDIT.md`
- `reports/BALANCED_MAX100_REPORT.md`
- `reports/FOLD_BALANCE_REPORT.md`
- `reports/ZERO_SHOT_5FOLD_REPORT.md`
- `reports/STAGE_ERROR_ANALYSIS.md`
- `reports/FEW_SHOT_ONBOARDING_REPORT.md`
- `reports/COMPUTE_ACCURACY_TRADEOFF.md`
- `reports/PROFESSOR_MEETING_SUMMARY.md`

Shared public protocol files use privacy-safe aliases only:

- `shared/participant_folds_5fold.json`
- `shared/shared_experiment_protocol.json`
- `shared/balanced_max100_summary.json`

Generated private data, logs, predictions, support sets, transcript caches, feature files, and run directories are ignored by Git.

Current limitation: the zero-shot worker is participant-disjoint but uses one development validation fold per outer fold. The stricter nested OOF refit procedure remains the next protocol-hardening step.
