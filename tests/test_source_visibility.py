from contextlib import contextmanager

import matplotlib.pyplot as plt
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

    sv.plot()
    assert_num_figures()

    dates = sv.get_optimal_date()

    expected_dates = [
        pd.Timestamp("2024-10-03 17:05:56"),
        pd.Timestamp("2024-10-03 19:05:56"),
        pd.Timestamp("2024-10-03 21:05:56"),
    ]

    assert dates == expected_dates
