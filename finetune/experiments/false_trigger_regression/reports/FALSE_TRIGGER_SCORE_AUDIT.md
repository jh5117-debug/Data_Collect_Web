# False-Trigger Score Audit

- Status: `ok`
- Reason: Scored decoded rosbag WAV windows with the current VIGIL two-stage detector.
- Diagnosis: `heldout_false_positive_model_or_threshold_issue`
- Scored rows: `4`

## Required Decision Logic

`final_trigger` must equal `stage1_accept AND stage2_accept`.

## Current Result

Score rows were available and audited.

| Case | Expected | Transcript hint | Stage 1 | Stage 2 | Final trigger | Feature hash | Embedding hash |
|---|---:|---|---:|---:|---|---|---|
| `false_positive_001_go` | 0 | `Go.` | 0.202083 | 0.846976 | `False` | `f87bf503b6b8a58c` | `49b58714741466ef` |
| `false_positive_002_joe` | 0 | `Joe.` | 0.995889 | 0.847932 | `True` | `b5ea982834709a0d` | `1f0737b0350867b0` |
| `false_positive_003_joke` | 0 | `Joke.` | 0.236179 | 0.847448 | `False` | `477304ed5e2eb64f` | `dbb69fc062ac1d3e` |
| `true_positive_001_vigil` | 1 | `VIGIL.` | 0.995428 | 0.846778 | `True` | `14c4e4716f99a151` | `e7cded0be7870d4c` |

## Constant Score Checks

- False accepts: `['false_positive_002_joe']`
- False rejects: `[]`
- Stage 2 negative accepts: `['false_positive_001_go', 'false_positive_002_joe', 'false_positive_003_joke']`
- Stage 2 constant check: `{'n': 4, 'constant': False, 'range': 0.0011541247367858887, 'mean': 0.8472836911678314, 'tolerance': 0.0001}`
- Feature hash check: `{'n': 4, 'unique': 4, 'identical': False, 'unique_hashes': ['14c4e4716f99a151', '477304ed5e2eb64f', 'b5ea982834709a0d', 'f87bf503b6b8a58c']}`
- Embedding hash check: `{'n': 4, 'unique': 4, 'identical': False, 'unique_hashes': ['1f0737b0350867b0', '49b58714741466ef', 'dbb69fc062ac1d3e', 'e7cded0be7870d4c']}`
