from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any


def logit(value: float) -> float:
    value = min(max(float(value), 1e-6), 1.0 - 1e-6)
    return math.log(value / (1.0 - value))


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(value)))


def apply_positive_bias(score: float | None, bias: float) -> float | None:
    if score is None:
        return None
    return sigmoid(logit(float(score)) + max(0.0, float(bias)))


def bounded_positive_bias(support_scores: list[float], threshold: float, *, max_bias: float = 1.0, margin: float = 0.02) -> float:
    if not support_scores:
        return 0.0
    ordered = sorted(float(score) for score in support_scores)
    median = ordered[len(ordered) // 2]
    needed = logit(float(threshold)) - logit(median) + float(margin)
    return min(float(max_bias), max(0.0, needed))


@dataclass
class TriggerState:
    state: str = "IDLE"
    cooldown_seconds: float = 5.0
    last_trigger_time: float | None = None

    def reset(self) -> None:
        self.state = "IDLE"
        self.last_trigger_time = None

    def cooldown_active(self, now: float | None = None) -> bool:
        if self.last_trigger_time is None:
            return False
        now = time.time() if now is None else float(now)
        return now - self.last_trigger_time < self.cooldown_seconds

    def update(self, *, candidate: bool, trigger_detected: bool, now: float | None = None) -> dict[str, Any]:
        now = time.time() if now is None else float(now)
        if self.cooldown_active(now):
            self.state = "LISTENING"
            return {"assistant_state": self.state, "cooldown_active": True, "trigger_accepted": False}
        if trigger_detected:
            self.state = "ASSISTANT_STATE"
            self.last_trigger_time = now
            return {"assistant_state": self.state, "cooldown_active": False, "trigger_accepted": True}
        self.state = "VIGIL_CANDIDATE" if candidate else "LISTENING"
        return {"assistant_state": self.state, "cooldown_active": False, "trigger_accepted": False}
