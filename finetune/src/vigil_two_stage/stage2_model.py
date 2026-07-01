from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class MaskedAttentionPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.score(x).squeeze(-1)
        if mask is None:
            mask = torch.ones(logits.shape, dtype=torch.bool, device=x.device)
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1)
        weights = weights.masked_fill(~mask, 0.0)
        denom = weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        weights = weights / denom
        pooled = torch.bmm(weights.unsqueeze(1), x).squeeze(1)
        return pooled, weights


class QwenVerifierHead(nn.Module):
    def __init__(self, input_dim: int, projection_dim: int = 256, embedding_dim: int = 128):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, projection_dim)
        self.pool = MaskedAttentionPool(projection_dim)
        self.embed = nn.Linear(projection_dim, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, 1)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        x = F.gelu(self.proj(self.norm(hidden.float())))
        pooled, attn = self.pool(x, mask)
        embedding = F.normalize(self.embed(pooled), dim=-1)
        logit = self.classifier(embedding).squeeze(-1)
        return {"logit": logit, "embedding": embedding, "attention": attn}
