from __future__ import annotations

from pathlib import Path

import numpy as np

from audio_processing import sliding_windows, temporary_audio_dir


def test_sliding_windows_uses_stride_and_final_padding():
    sample_rate = 10
    waveform = np.ones(25, dtype=np.float32)
    windows = sliding_windows(waveform, sample_rate, window_seconds=1.0, stride_seconds=0.5)
    assert [window.index for window in windows] == [0, 1, 2, 3]
    assert [window.start_sec for window in windows] == [0.0, 0.5, 1.0, 1.5]
    assert windows[-1].end_sec == 2.5
    assert windows[-1].padded is False


def test_sliding_windows_pads_short_audio():
    sample_rate = 10
    waveform = np.ones(5, dtype=np.float32)
    windows = sliding_windows(waveform, sample_rate, window_seconds=1.0, stride_seconds=0.5)
    assert len(windows) == 1
    assert windows[0].padded is True
    assert windows[0].waveform.shape[0] == 10
    assert float(windows[0].waveform[5:].sum()) == 0.0


def test_temporary_audio_dir_cleans_up():
    created: Path
    with temporary_audio_dir() as path:
        created = path
        (path / "x.wav").write_text("x", encoding="utf-8")
        assert path.exists()
    assert not created.exists()

