#!/usr/bin/env python3
from __future__ import annotations

import inspect
from importlib import metadata
from pathlib import Path
from typing import Any

from vigil_latest_opt.utils import write_json


def safe_signature(obj: Any) -> str | None:
    try:
        return str(inspect.signature(obj))
    except Exception:
        return None


def main() -> int:
    reports = Path("finetune/experiments/latest_data_optimization/reports")
    reports.mkdir(parents=True, exist_ok=True)
    diagnostic: dict[str, Any] = {
        "status": "blocked_by_runtime_interface",
        "required_interface": "public transcribe would need to accept/return reusable audio encoder hidden states, or expose a one-forward ASR+features call",
        "claim": "No one-encoder-forward shared-Qwen success is claimed.",
    }
    try:
        import qwen_asr
        from qwen_asr import Qwen3ASRModel  # type: ignore

        diagnostic["package_file"] = getattr(qwen_asr, "__file__", None)
        try:
            diagnostic["package_version"] = metadata.version("qwen-asr")
        except metadata.PackageNotFoundError:
            diagnostic["package_version"] = "unknown"
        diagnostic["from_pretrained_signature"] = safe_signature(Qwen3ASRModel.from_pretrained)
        diagnostic["class_has_transcribe"] = hasattr(Qwen3ASRModel, "transcribe")
        diagnostic["transcribe_signature"] = safe_signature(getattr(Qwen3ASRModel, "transcribe", None))
        diagnostic["class_public_methods"] = sorted(name for name in dir(Qwen3ASRModel) if not name.startswith("_"))[:80]
        diagnostic["audio_feature_path_used_by_stage2"] = "model.thinker.get_audio_features"
        diagnostic["blocker"] = (
            "The public wrapper exposes transcribe/generate-style transcript calls and the separate thinker.get_audio_features path, "
            "but it does not expose a public call that reuses the same audio encoder hidden states for both ASR decoding and Stage2 verification."
        )
    except Exception as exc:
        diagnostic["status"] = "not_feasible_with_current_revision"
        diagnostic["blocker"] = f"could_not_import_or_inspect_qwen_asr: {type(exc).__name__}: {exc}"
    write_json(reports / "latest_opt_shared_qwen_diagnostic.json", diagnostic)
    lines = [
        "# Latest Optimized Shared-Qwen Feasibility Report",
        "",
        f"- Status: `{diagnostic['status']}`",
        f"- Package file: `{diagnostic.get('package_file')}`",
        f"- Package version: `{diagnostic.get('package_version')}`",
        f"- Public transcribe signature: `{diagnostic.get('transcribe_signature')}`",
        f"- Audio feature path used by Stage2: `{diagnostic.get('audio_feature_path_used_by_stage2')}`",
        f"- Blocker: {diagnostic.get('blocker')}",
        f"- Required interface: {diagnostic.get('required_interface')}",
        "- No shared hidden-state reuse is claimed without call-counter proof.",
    ]
    (reports / "LATEST_OPT_SHARED_QWEN_FEASIBILITY_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(diagnostic["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
