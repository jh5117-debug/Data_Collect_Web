from __future__ import annotations

from typing import Any

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def reset_cuda_peak_memory(device: str | None = None) -> None:
    if torch is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def cuda_memory_summary(device: str | None = None) -> dict[str, Any]:
    if torch is None or not torch.cuda.is_available():
        return {"cuda_available": False}
    return {
        "cuda_available": True,
        "peak_allocated_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
        "peak_reserved_gb": torch.cuda.max_memory_reserved(device) / 1024**3,
        "current_allocated_gb": torch.cuda.memory_allocated(device) / 1024**3,
        "current_reserved_gb": torch.cuda.memory_reserved(device) / 1024**3,
    }
