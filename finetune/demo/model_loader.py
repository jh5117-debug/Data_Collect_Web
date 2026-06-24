from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finetune.scripts.extract_openwakeword_features import OfficialOpenWakeWordExtractor
from vigil_two_stage.qwen_audio_adapter import FrozenQwenAudioAdapter
from vigil_two_stage.stage1_model import Stage1GRUClassifier
from vigil_two_stage.stage2_model import QwenVerifierHead
from vigil_two_stage.utils import read_json


VARIANT_DIRS = {
    "bce": "stage2_bce",
    "bce_supcon": "stage2_bce_supcon",
    "stage2_bce": "stage2_bce",
    "stage2_bce_supcon": "stage2_bce_supcon",
}


@dataclass
class Stage1Runtime:
    model: Stage1GRUClassifier
    theta: float
    input_dim: int


@dataclass
class Stage2Runtime:
    model: QwenVerifierHead
    theta: float
    input_dim: int
    variant_dir: str


@dataclass
class VigilRuntime:
    run_dir: Path
    device: torch.device
    openwakeword: OfficialOpenWakeWordExtractor
    qwen: FrozenQwenAudioAdapter
    stage1: Stage1Runtime
    stage2: dict[str, Stage2Runtime]
    selected_variant: str
    model_selection: dict[str, Any]


def require_single_rtx_3090() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing CPU fallback")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected exactly one visible CUDA device, got {torch.cuda.device_count()}")
    name = torch.cuda.get_device_name(0)
    if "RTX 3090" not in name:
        raise RuntimeError(f"visible CUDA device is not an RTX 3090: {name}")
    return torch.device("cuda:0")


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=False)


def load_stage1(run_dir: Path, device: torch.device) -> Stage1Runtime:
    ckpt = _load_checkpoint(run_dir / "stage1" / "checkpoint_best.pt")
    config = ckpt.get("config") or read_json(run_dir / "stage1" / "model_config.json")
    input_dim = int(ckpt.get("input_dim") or config["input_dim"])
    model = Stage1GRUClassifier(
        input_dim,
        hidden_size=int(config["gru_hidden_size"]),
        layers=int(config["gru_layers"]),
        dropout=float(config["dropout"]),
    )
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Stage 1 checkpoint key mismatch: missing={missing}, unexpected={unexpected}")
    threshold = read_json(run_dir / "stage1" / "threshold.json")
    model.to(device).eval()
    return Stage1Runtime(model=model, theta=float(threshold["threshold"]), input_dim=input_dim)


def _infer_stage2_dims(state: dict[str, torch.Tensor]) -> tuple[int, int, int]:
    input_dim = int(state["norm.weight"].shape[0])
    projection_dim = int(state["proj.weight"].shape[0])
    embedding_dim = int(state["embed.weight"].shape[0])
    return input_dim, projection_dim, embedding_dim


def load_stage2(run_dir: Path, variant: str, device: torch.device) -> Stage2Runtime:
    variant_dir = VARIANT_DIRS[variant]
    ckpt = _load_checkpoint(run_dir / variant_dir / "checkpoint_best.pt")
    state = ckpt["model_state"]
    input_dim, projection_dim, embedding_dim = _infer_stage2_dims(state)
    config = ckpt.get("config", {})
    if int(config.get("projection_dim", projection_dim)) != projection_dim:
        raise RuntimeError(f"{variant_dir} projection dimension mismatch")
    if int(config.get("embedding_dim", embedding_dim)) != embedding_dim:
        raise RuntimeError(f"{variant_dir} embedding dimension mismatch")
    model = QwenVerifierHead(input_dim, projection_dim=projection_dim, embedding_dim=embedding_dim)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"{variant_dir} checkpoint key mismatch: missing={missing}, unexpected={unexpected}")
    threshold = read_json(run_dir / variant_dir / "threshold.json")
    model.to(device).eval()
    return Stage2Runtime(model=model, theta=float(threshold["threshold"]), input_dim=input_dim, variant_dir=variant_dir)


def load_runtime(run_dir: Path | str) -> VigilRuntime:
    run_dir = Path(run_dir)
    device = require_single_rtx_3090()
    model_selection = read_json(run_dir / "model_selection.json")
    selected = str(model_selection.get("selected_variant") or "stage2_bce")
    if selected not in VARIANT_DIRS:
        raise RuntimeError(f"unsupported selected Stage 2 variant: {selected}")
    openwakeword = OfficialOpenWakeWordExtractor()
    stage1 = load_stage1(run_dir, device)
    stage2 = {
        "stage2_bce": load_stage2(run_dir, "stage2_bce", device),
        "stage2_bce_supcon": load_stage2(run_dir, "stage2_bce_supcon", device),
    }
    qwen = FrozenQwenAudioAdapter("Qwen/Qwen3-ASR-1.7B")
    qwen.load()
    integrity = qwen.integrity()
    if integrity.trainable_parameters != 0:
        raise RuntimeError(f"Qwen must remain frozen, got {integrity.trainable_parameters} trainable parameters")
    return VigilRuntime(
        run_dir=run_dir,
        device=device,
        openwakeword=openwakeword,
        qwen=qwen,
        stage1=stage1,
        stage2=stage2,
        selected_variant=selected,
        model_selection=model_selection,
    )

