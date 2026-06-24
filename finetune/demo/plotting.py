from __future__ import annotations

from typing import Any


def stage1_score_plot(stage1_scores: list[dict[str, Any]], theta_1: float):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 3))
    starts = [row["start_sec"] for row in stage1_scores]
    scores = [row["score"] for row in stage1_scores]
    ax.plot(starts, scores, marker="o", linewidth=1.5, markersize=3)
    ax.axhline(theta_1, color="#b91c1c", linestyle="--", linewidth=1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Stage 1")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig

