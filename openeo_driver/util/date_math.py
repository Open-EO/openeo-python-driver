import datetime as dt
from typing import Union

import pandas as pd


def now_utc() -> dt.datetime:
    """
    Current timezone-aware (UTC) datetime
    to be used instead of deprecated naive datetime.utcnow()
    """
    return dt.datetime.now(tz=dt.timezone.utc)


def month_shift(
        d: Union[dt.date, dt.datetime, pd.Timestamp],
        months: int = 1
) -> Union[dt.date, dt.datetime, pd.Timestamp]:
    """
    Shift a date with given amount of months.

    Month overflows are clipped. E.g. Jan 31 + 1 month -> Feb 28

    :param d: date/datetime/timestamp
    :param months: amount of months to shift (positive: to future, negative: to past)
    :return:
    """
    year = d.year
    month = d.month + months
    if not (1 <= month <= 12):
        year = year + ((month - 1) // 12)
        month = ((month - 1) % 12) + 1
    try:
        return d.replace(year=year, month=month)
    except ValueError:
        # Handle month overflow (e.g clip Feb 31 to 28)
        return month_shift(d=d.replace(day=1), months=months + 1) - dt.timedelta(days=1)


def simple_job_progress_estimation(started: dt.datetime, average_run_time: float) -> float:
    """
    Simple progress estimation,
    assuming job run time is distributed exponentially (with lambda = 1 / average run time)

    - estimated remaining run time = average run time
      (note that this is regardless of current run time so far,
      this is mathematical consequence of assuming an exponential distribution)
    - estimated total run time = current run time + average run time
    - estimated progress = current run time / (current run time + average run time)

    :param started: start time of the job
    :param average_run_time: average run time of jobs in seconds
    :return: progress as a fraction in range [0, 1]
    """
    # TODO: also support string input?
    # TODO: also support other timezones than UTC or naive?

    if started.tzinfo is None:
        # Convert naive to UTC
        started = started.replace(tzinfo=dt.timezone.utc)

    now = dt.datetime.now(tz=dt.timezone.utc)
    elapsed = (now - started).total_seconds()
    if elapsed <= 0 or average_run_time <= 0:
        return 0.0
    progress = elapsed / (elapsed + average_run_time)
    return progress
