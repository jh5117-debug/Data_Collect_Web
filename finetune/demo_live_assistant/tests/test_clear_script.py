from __future__ import annotations

from pathlib import Path


def test_clear_local_demo_data_script_has_path_guard() -> None:
    script = Path("finetune/demo_live_assistant/scripts/clear_local_demo_data.sh").read_text(encoding="utf-8")
    assert "finetune/demo_live_assistant/local_data" in script
    assert "Refusing to clear unsafe path" in script
