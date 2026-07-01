# Export Inspection

- ZIP SHA-256: `f0808053b27d2c5d0af834df5ecc3662af7dd47fbf3984691da27bfbd8b6628d`
- Metadata clip rows: 403
- Canonical samples: 403
- Rejected/inconsistent rows: 0
- Canonical `audio_raw/` files: 403
- Duplicate `raw_audio/` view files: 806
- Duplicate `by_prompt_group/` view files: 403
- Prompt groups: `{'P1_vigil_only': 31, 'P2_phrase_plus_vigil': 111, 'P3_vigil_plus_phrase': 76, 'P4_negative': 185}`
- Labels: `{1: 218, 0: 185}`

The parser uses `metadata/clips.jsonl` as the canonical table and resolves one `audio_raw/<clip_id>.*` source per clip.
