from __future__ import annotations

import numpy as np
import torch


def score_prefixes(model: torch.nn.Module, features: np.ndarray, device: str = "cpu") -> dict[str, float | int]:
    model.eval()
    scores = []
    with torch.no_grad():
        for t in range(1, features.shape[0] + 1):
            x = torch.from_numpy(features[:t][None, :, :]).float().to(device)
            lengths = torch.tensor([t], dtype=torch.long, device=device)
            score = torch.sigmoid(model(x, lengths))[0].item()
            scores.append(score)
    idx = int(np.argmax(scores)) if scores else 0
    return {"max_score": float(scores[idx]) if scores else 0.0, "peak_frame_index": idx}
