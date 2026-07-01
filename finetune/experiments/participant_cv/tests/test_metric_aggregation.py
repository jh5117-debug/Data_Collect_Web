from __future__ import annotations

from vigil_participant_cv.aggregation import mean_std


def test_fold_metric_aggregation_calculates_mean_and_std():
    result = mean_std([1.0, 2.0, 3.0])
    assert result["mean"] == 2.0
    assert round(result["std"], 6) == 1.0
