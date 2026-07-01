from __future__ import annotations

from vigil_final.latency import cuda_synchronized_timer, summarize_latencies
from vigil_final.qwen_call_counter import QwenCallCounter


def test_latency_summary_and_timer():
    with cuda_synchronized_timer() as timer:
        pass
    assert timer["wall_seconds"] >= 0
    summary = summarize_latencies([0.1, 0.2, 0.3])
    assert summary["median"] == 0.2


def test_qwen_copy_count_differs_from_forward_count():
    counter = QwenCallCounter(loaded_weight_instances=1, transcribe_calls=10, audio_encoder_forward_calls=15, lm_generation_calls=10)
    assert counter.loaded_weight_instances == 1
    assert counter.audio_encoder_forward_calls != counter.loaded_weight_instances


def test_shared_counter_requires_one_encoder_call_when_claimed():
    counter = QwenCallCounter(loaded_weight_instances=1, transcribe_calls=1, audio_encoder_forward_calls=1, lm_generation_calls=1)
    counter.assert_shared_success()
