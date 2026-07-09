# Hard-Negative Retraining Plan

## Rules

- Do not retrain Qwen.
- Qwen ASR transcript target remains exact transcript text.
- KWS label for go / joe / joke / yo is `0`.
- Keep Shaw/Andy false-positive rosbags as held-out regression cases, not training data.
- Collect more speakers for these hard negatives before retraining.
- If model bias is confirmed, retrain or recalibrate the Stage 2 verifier head first.
- Stage 1 should remain high-recall; tune Stage 1 only if candidate rate is too high.

## Current Audit Evidence

- `Joe.` is a final false accept in Shaw's held-out false-positive bag.
- `Go.` and `Joke.` are rejected by the current cascade because Stage 1 rejects them.
- Stage 2 accepts all three negative windows in this tiny held-out set.
- Feature and Stage 2 embedding hashes differ across `Go.`, `Joe.`, `Joke.`, and `VIGIL.`, so this audit does not support stale cached features or identical audio windows.
- This does not mean retraining is complete; it identifies the next hard-negative regression target.

## Prompt 4 Hard Negatives To Add

- go
- go go
- joe
- joke
- yo
- yo yo
- hey yo
- hello
- no
- visual
- visible
- digital
- individual
- vigilant

## Regression Policy

The current rosbag examples should be replayed after any integration change, threshold change, Stage 2 retraining, or hard-negative data expansion. They should not be used to tune thresholds in this task.
