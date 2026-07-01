#!/usr/bin/env python3
from __future__ import annotations

import inspect
from pathlib import Path

from vigil_final.qwen_call_counter import QwenCallCounter
from vigil_final.shared_qwen import SharedQwenDiagnostic, validate_shared_status
from vigil_final.utils import write_json


def main() -> int:
    public_path = None
    audio_path = "model.thinker.get_audio_features"
    blocker = None
    try:
        import qwen_asr

        package_file = getattr(qwen_asr, "__file__", None)
        public_path = str(package_file)
        source_hint = inspect.getsource(qwen_asr)[:1000] if package_file else ""
    except Exception as exc:
        source_hint = ""
        blocker = f"could_not_import_qwen_asr: {exc}"
    status = "blocked_by_runtime_interface"
    required = "public transcribe would need to expose precomputed audio hidden states or return reusable audio encoder features"
    diagnostic = SharedQwenDiagnostic(
        status=status,
        public_transcribe_path=public_path,
        audio_feature_path=audio_path,
        blocker=blocker or "public wrapper returns transcript objects but does not expose reusable hidden-state handoff",
        required_interface=required,
        call_counts=QwenCallCounter(loaded_weight_instances=1, transcribe_calls=1, audio_encoder_forward_calls=2, lm_generation_calls=1).as_dict(),
    )
    validate_shared_status(diagnostic.status)
    out = diagnostic.to_json()
    out["source_hint"] = source_hint
    write_json("finetune/experiments/vigil_final/reports/shared_qwen_diagnostic.json", out)
    Path("finetune/experiments/vigil_final/reports/SHARED_QWEN_PROTOTYPE_REPORT.md").write_text(
        "# Shared-Qwen Prototype Report\n\n"
        f"- Status: `{status}`\n"
        f"- Blocker: {out['blocker']}\n"
        f"- Required interface: {required}\n"
        "- No shared hidden-state reuse is claimed without one-encoder-forward call-counter proof.\n",
        encoding="utf-8",
    )
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
