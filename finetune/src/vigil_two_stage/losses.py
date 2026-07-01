from __future__ import annotations

import torch
import torch.nn.functional as F


def bce_with_logits_loss(logits: torch.Tensor, labels: torch.Tensor, pos_weight: torch.Tensor | None = None) -> torch.Tensor:
    labels = labels.float()
    return F.binary_cross_entropy_with_logits(logits.float().view_as(labels), labels, pos_weight=pos_weight)


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    phrase_ids: list[str],
    *,
    temperature: float = 0.07,
    exclude_phrase: str = "background",
) -> torch.Tensor:
    if embeddings.numel() == 0:
        return embeddings.sum() * 0.0
    z = F.normalize(embeddings.float(), dim=-1)
    n = z.shape[0]
    if n <= 1:
        return z.sum() * 0.0
    sim = torch.matmul(z, z.T) / temperature
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    losses = []
    for i, pid in enumerate(phrase_ids):
        if pid == exclude_phrase:
            continue
        positive_mask = torch.tensor([j != i and phrase_ids[j] == pid for j in range(n)], dtype=torch.bool, device=z.device)
        if not bool(positive_mask.any()):
            continue
        denom_mask = ~eye[i]
        denom = torch.logsumexp(sim[i][denom_mask], dim=0)
        losses.append(-(sim[i][positive_mask] - denom).mean())
    if not losses:
        return z.sum() * 0.0
    return torch.stack(losses).mean()
