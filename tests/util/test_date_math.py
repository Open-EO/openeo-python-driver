import datetime as dt

import pandas as pd
import pytest
import time_machine

from openeo_driver.util.date_math import month_shift, simple_job_progress_estimation


def test_month_shift_date():
    assert month_shift(dt.date(2022, 8, 20), months=0) == dt.date(2022, 8, 20)
    assert month_shift(dt.date(2022, 8, 20), months=1) == dt.date(2022, 9, 20)
    assert month_shift(dt.date(2022, 8, 20), months=4) == dt.date(2022, 12, 20)
    assert month_shift(dt.date(2022, 8, 20), months=5) == dt.date(2023, 1, 20)
    assert month_shift(dt.date(2022, 8, 20), months=25) == dt.date(2024, 9, 20)

    assert month_shift(dt.date(2022, 8, 20), months=-1) == dt.date(2022, 7, 20)
    assert month_shift(dt.date(2022, 8, 20), months=-4) == dt.date(2022, 4, 20)
    assert month_shift(dt.date(2022, 8, 20), months=-7) == dt.date(2022, 1, 20)
    assert month_shift(dt.date(2022, 8, 20), months=-8) == dt.date(2021, 12, 20)
    assert month_shift(dt.date(2022, 8, 20), months=-9) == dt.date(2021, 11, 20)
    assert month_shift(dt.date(2022, 8, 20), months=-25) == dt.date(2020, 7, 20)


def test_month_shift_overflow_date():
    assert month_shift(dt.date(2022, 1, 31), months=1) == dt.date(2022, 2, 28)
    assert month_shift(dt.date(2022, 3, 31), months=-1) == dt.date(2022, 2, 28)


def test_month_shift_datetime():
    assert month_shift(dt.datetime(2022, 8, 20, 14, 15, 16), months=0) == dt.datetime(2022, 8, 20, 14, 15, 16)
    assert month_shift(dt.datetime(2022, 8, 20, 14, 15, 16), months=1) == dt.datetime(2022, 9, 20, 14, 15, 16)
    assert month_shift(dt.datetime(2022, 8, 20, 14, 15, 16), months=4) == dt.datetime(2022, 12, 20, 14, 15, 16)
    assert month_shift(dt.datetime(2022, 8, 20, 14, 15, 16), months=5) == dt.datetime(2023, 1, 20, 14, 15, 16)
    assert month_shift(dt.datetime(2022, 8, 20, 14, 15, 16), months=25) == dt.datetime(2024, 9, 20, 14, 15, 16)

    assert month_shift(dt.datetime(2022, 8, 20, 14, 15, 16), months=-1) == dt.datetime(2022, 7, 20, 14, 15, 16)
    assert month_shift(dt.datetime(2022, 8, 20, 14, 15, 16), months=-4) == dt.datetime(2022, 4, 20, 14, 15, 16)
    assert month_shift(dt.datetime(2022, 8, 20, 14, 15, 16), months=-7) == dt.datetime(2022, 1, 20, 14, 15, 16)
    assert month_shift(dt.datetime(2022, 8, 20, 14, 15, 16), months=-8) == dt.datetime(2021, 12, 20, 14, 15, 16)
    assert month_shift(dt.datetime(2022, 8, 20, 14, 15, 16), months=-9) == dt.datetime(2021, 11, 20, 14, 15, 16)
    assert month_shift(dt.datetime(2022, 8, 20, 14, 15, 16), months=-25) == dt.datetime(2020, 7, 20, 14, 15, 16)


def test_month_shift_overflow_datetime():
    assert month_shift(dt.datetime(2022, 1, 31, 14, 15, 16), months=1) == dt.datetime(2022, 2, 28, 14, 15, 16)
    assert month_shift(dt.datetime(2022, 3, 31, 14, 15, 16), months=-1) == dt.datetime(2022, 2, 28, 14, 15, 16)


def test_month_shift_pandas_timestamp():
    assert month_shift(pd.to_datetime("2022-08-20"), months=0) == pd.Timestamp(2022, 8, 20)
    assert month_shift(pd.to_datetime("2022-08-20"), months=1) == pd.Timestamp(2022, 9, 20)
    assert month_shift(pd.to_datetime("2022-08-20"), months=4) == pd.Timestamp(2022, 12, 20)
    assert month_shift(pd.to_datetime("2022-08-20"), months=5) == pd.Timestamp(2023, 1, 20)
    assert month_shift(pd.to_datetime("2022-08-20"), months=25) == pd.Timestamp(2024, 9, 20)

    assert month_shift(pd.to_datetime("2022-08-20"), months=-1) == pd.Timestamp(2022, 7, 20)
    assert month_shift(pd.to_datetime("2022-08-20"), months=-4) == pd.Timestamp(2022, 4, 20)
    assert month_shift(pd.to_datetime("2022-08-20"), months=-7) == pd.Timestamp(2022, 1, 20)
    assert month_shift(pd.to_datetime("2022-08-20"), months=-8) == pd.Timestamp(2021, 12, 20)
    assert month_shift(pd.to_datetime("2022-08-20"), months=-9) == pd.Timestamp(2021, 11, 20)
    assert month_shift(pd.to_datetime("2022-08-20"), months=-25) == pd.Timestamp(2020, 7, 20)


def test_month_shift_overflow_pandas_timestamp():
    assert month_shift(pd.to_datetime("2022-01-31"), months=1) == pd.Timestamp(2022, 2, 28)
    assert month_shift(pd.to_datetime("2022-03-31"), months=-1) == pd.Timestamp(2022, 2, 28)


@time_machine.travel("2024-12-06T12:00:00+00")
@pytest.mark.parametrize(
    "tzinfo",
    [
        None,  # Naive
        dt.timezone.utc,  # Explicit UTC
    ],
)
def test_simple_job_progress_estimation_basic(tzinfo):
    # Started 1 second ago
    assert simple_job_progress_estimation(
        dt.datetime(2024, 12, 6, 11, 59, 59, tzinfo=tzinfo),
        average_run_time=600,
    ) == pytest.approx(0.0, abs=0.01)
    # Started 5 minutes ago
    assert simple_job_progress_estimation(
        dt.datetime(2024, 12, 6, 11, 55, tzinfo=tzinfo),
        average_run_time=600,
    ) == pytest.approx(0.33, abs=0.01)
    # Long overdue
    assert simple_job_progress_estimation(
        dt.datetime(2024, 12, 5, tzinfo=tzinfo),
        average_run_time=600,
    ) == pytest.approx(1.0, abs=0.01)


@time_machine.travel("2024-12-06T12:00:00+00")
def test_simple_job_progress_estimation_negative():
    # OMG a job from the future.
    assert (
        simple_job_progress_estimation(
            started=dt.datetime(2024, 12, 8),
            average_run_time=600,
        )
        == 0.0
    )
