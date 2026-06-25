from __future__ import annotations

from dataclasses import dataclass


@dataclass
class QwenCallCounter:
    loaded_weight_instances: int = 0
    transcribe_calls: int = 0
    audio_encoder_forward_calls: int = 0
    lm_generation_calls: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "loaded_weight_instances": self.loaded_weight_instances,
            "transcribe_calls": self.transcribe_calls,
            "audio_encoder_forward_calls": self.audio_encoder_forward_calls,
            "lm_generation_calls": self.lm_generation_calls,
        }

    def assert_shared_success(self) -> None:
        if self.loaded_weight_instances != 1:
            raise AssertionError("shared path requires exactly one loaded Qwen weight instance")
        if self.audio_encoder_forward_calls != 1:
            raise AssertionError("shared path requires exactly one audio encoder forward")
        if self.transcribe_calls != 1:
            raise AssertionError("shared path requires exactly one transcript path call")
