"""Data checks, the typical day and the histogram, on series with known answers."""

import numpy as np
import pandas as pd

from strawberrywatch.workbench import checks, text


def series(values, start="2025-06-08 07:00"):
    index = pd.date_range(pd.Timestamp(start, tz="UTC"), periods=len(values), freq="15min")
    return pd.Series(values, index=index, dtype=float)


def test_missing_counts_96_a_day():
    values = series(np.arange(96 * 7, dtype=float))
    start, end = values.index[0], values.index[-1]
    assert checks.missing(values, start, end) == (0, 672)
    gappy = values.copy()
    gappy.iloc[10:30] = np.nan
    assert checks.missing(gappy, start, end) == (20, 672)
    assert checks.missing(gappy.dropna(), start, end) == (20, 672)


def test_stuck_runs():
    values = series([1, 2] * 10 + [5] * 8 + [6] * 7 + [7] * 12)
    assert checks.STUCK_RUN == 8
    assert checks.stuck(values) == 2
    assert checks.stuck(series([])) == 0


def test_hampel_flags_only_the_spikes():
    rng = np.random.default_rng(1)
    values = series(500 + rng.normal(0, 1, 96 * 4))
    values.iloc[[50, 200]] = [560, 430]
    flagged = checks.far(values)
    assert set(np.flatnonzero(flagged.to_numpy())) >= {50, 200}
    assert int(flagged.sum()) <= 4


def test_typical_day_matches_pandas():
    rng = np.random.default_rng(2)
    values = series(rng.normal(0, 1, 96 * 10))
    hours, median, low, high = checks.typical_day(values)
    local = values.index.tz_convert(text.LOCAL).hour
    grouped = values.groupby(local)
    assert hours == list(range(24))
    np.testing.assert_allclose(median, grouped.median().to_numpy())
    quartiles = grouped.quantile([0.25, 0.75]).unstack()
    np.testing.assert_allclose(low, quartiles[0.25].to_numpy())
    np.testing.assert_allclose(high, quartiles[0.75].to_numpy())


def test_histogram_matches_numpy_on_bin_multiples():
    values = series([521.0, 522.4, 530.0, 548.9, 551.2, 569.9])
    edges, counts = checks.histogram(values, 5)
    assert edges[0] == 520 and edges[-1] == 570
    assert all(e % 5 == 0 for e in edges)
    expected, _ = np.histogram(values.to_numpy(), bins=np.array(edges))
    assert counts == expected.tolist()
    edges, counts = checks.histogram(series([15.2, 15.7, 16.3]), 0.5)
    np.testing.assert_allclose(edges, [15.0, 15.5, 16.0, 16.5])
    assert counts == [1, 1, 1]
