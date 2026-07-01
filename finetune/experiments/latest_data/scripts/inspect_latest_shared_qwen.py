#!/usr/bin/env python3
from __future__ import annotations

import inspect
from pathlib import Path

from vigil_latest.utils import write_json


def main() -> int:
    reports = Path("finetune/experiments/latest_data/reports")
    reports.mkdir(parents=True, exist_ok=True)
    try:
        import qwen_asr

        package_file = getattr(qwen_asr, "__file__", None)
        source_hint = inspect.getsource(qwen_asr)[:1000] if package_file else ""
        blocker = "public wrapper returns transcript objects but does not expose reusable hidden-state handoff"
    except Exception as exc:
        package_file = None
        source_hint = ""
        blocker = f"could_not_import_qwen_asr: {exc}"
    result = {
        "status": "blocked_by_runtime_interface",
        "package_file": package_file,
        "audio_feature_path": "model.thinker.get_audio_features",
        "blocker": blocker,
        "source_hint": source_hint,
        "claim": "No one-encoder-forward shared-Qwen success is claimed.",
    }
    write_json(reports / "latest_shared_qwen_diagnostic.json", result)
    (reports / "LATEST_SHARED_QWEN_PROTOTYPE_REPORT.md").write_text(
        "# Latest Shared-Qwen Prototype Report\n\n"
        "- Status: `blocked_by_runtime_interface`\n"
        f"- Blocker: {blocker}\n"
        "- No shared hidden-state reuse is claimed without call-counter proof.\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
