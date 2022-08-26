import datetime as dt
from typing import Union

import pandas as pd


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
