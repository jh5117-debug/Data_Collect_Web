import numpy as np

from vigil_two_stage.audio import fixed_window_bounds, materialize_window


def test_p1_window_centering():
    bounds = fixed_window_bounds("P1_vigil_only", 1000, 3000, 8000, 1000, 2.0)
    assert bounds[0][0] == 1000
    assert bounds[0][1] == 3000


def test_p2_end_window_logic():
    bounds = fixed_window_bounds("P2_phrase_plus_vigil", 1000, 5000, 8000, 1000, 2.0)
    assert bounds[0][:2] == (3000, 5000)


def test_p3_start_window_logic():
    bounds = fixed_window_bounds("P3_vigil_plus_phrase", 1000, 5000, 8000, 1000, 2.0)
    assert bounds[0][:2] == (1000, 3000)


def test_p4_negative_window_logic_and_padding():
    bounds = fixed_window_bounds("P4_negative", 1000, 7000, 8000, 1000, 2.0)
    assert len(bounds) >= 2
    audio = np.ones(1000, dtype=np.float32)
    win, left, right = materialize_window(audio, -500, 1500)
    assert len(win) == 2000
    assert left == 500
    assert right == 500
