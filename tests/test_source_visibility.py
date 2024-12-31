from contextlib import contextmanager

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@contextmanager
def assert_num_figures():
    n_fig_prev = plt.gcf().number
    yield
    n_fig_after = plt.gcf().number
    assert n_fig_prev < n_fig_after


def test_source_visibility():
    from radiotools.visibility import SourceVisibility

    sv = SourceVisibility(
        target="M87",
        date="2024-10-03",
        location="vlba",
        obs_length=4.0,
        frame="icrs",
        print_optimal_date=False,
    )

    with assert_num_figures():
        sv.plot()

    dates = sv.get_optimal_date()

    expected_dates = [
        pd.Timestamp("2024-10-03 17:05:56"),
        pd.Timestamp("2024-10-03 19:05:56"),
        pd.Timestamp("2024-10-03 21:05:56"),
    ]

    assert dates == expected_dates


def test_source_visibility_alt_restrictions():
    from radiotools.visibility import SourceVisibility

    alts = np.array([np.arange(0, 95, 5), np.arange(90, -5, -5)])

    opt_dates = [
        pd.Timestamp("2022-12-31 16:29:11"),
        pd.Timestamp("2022-12-31 22:29:11"),
        pd.Timestamp("2023-01-01 04:29:11"),
    ]

    results = []

    for i in np.arange(alts.shape[1]):
        try:
            results.append(
                SourceVisibility(
                    target="crab",
                    date="2022-12-31",
                    location="vla",
                    obs_length=12.0,
                    min_alt=alts[0][i],
                    max_alt=alts[1][i],
                ).get_optimal_date()
                == opt_dates
            )
        except ValueError:
            results.append(False)

    assert np.all(results[:5])
    assert np.all(np.logical_not(results[5:]))
