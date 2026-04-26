"""Temporal aggregation process implementations."""
import calendar
import logging
from datetime import timedelta
from typing import List, Tuple

import pandas as pd

from openeo_driver.datacube import DriverDataCube
from openeo_driver.dry_run import DryRunDataTracer
from openeo_driver.errors import ProcessParameterInvalidException
from openeo_driver.processes import ProcessArgs
from openeo_driver.processgraph.registry import ENV_DRY_RUN_TRACER, process
from openeo_driver.util.date_math import month_shift
from openeo_driver.utils import EvalEnv

_log = logging.getLogger(__name__)


@process
def aggregate_temporal(args: ProcessArgs, env: EvalEnv) -> DriverDataCube:
    data_cube = args.get_required("data", expected_type=DriverDataCube)
    intervals = args.get_required("intervals")
    reduce_pg = args.get_deep("reducer", "process_graph", expected_type=dict)
    labels = args.get_optional("labels", default=None)
    dimension = args.get_optional(
        "dimension",
        default=lambda: data_cube.metadata.temporal_dimension.name,
        validator=ProcessArgs.validator_one_of(data_cube.get_dimension_names()),
    )
    context = args.get_optional("context", default=None)

    return data_cube.aggregate_temporal(
        intervals=intervals, labels=labels, reducer=reduce_pg, dimension=dimension, context=context
    )


@process
def aggregate_temporal_period(args: ProcessArgs, env: EvalEnv) -> DriverDataCube:
    data_cube = args.get_required("data", expected_type=DriverDataCube)
    period = args.get_required("period")
    reduce_pg = args.get_deep("reducer", "process_graph", expected_type=dict)
    dimension = args.get_optional(
        "dimension",
        default=lambda: data_cube.metadata.temporal_dimension.name,
        validator=ProcessArgs.validator_one_of(data_cube.get_dimension_names()),
    )
    context = args.get_optional("context", default=None)

    dry_run_tracer: DryRunDataTracer = env.get(ENV_DRY_RUN_TRACER)
    if dry_run_tracer:
        intervals = []
    else:
        temporal_extent = data_cube.metadata.temporal_dimension.extent
        start = temporal_extent[0]
        end = temporal_extent[1]
        intervals = _period_to_intervals(start, end, period)

    return data_cube.aggregate_temporal(intervals=intervals, labels=None, reducer=reduce_pg, dimension=dimension,
                                        context=context)


def _period_to_intervals(start, end, period) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    if start.tzinfo:
        start = start.tz_convert(None)
    if end.tzinfo:
        end = end.tz_convert(None)

    if "day" == period:
        offset = timedelta(days=1)
        start_dates = pd.date_range(start - offset, end, freq="D", inclusive="left")
        end_dates = pd.date_range(start, end + offset, freq="D", inclusive="left")
        intervals = zip(start_dates, end_dates)
    elif "week" == period:
        offset = timedelta(weeks=1)
        start_dates = pd.date_range(start - offset, end, freq="W", inclusive="left")
        end_dates = pd.date_range(start, end + offset, freq="W", inclusive="left")
        intervals = zip(start_dates, end_dates)
    elif "dekad" == period:
        offset = timedelta(days=10)
        start_dates = pd.date_range(start - offset, end, freq="MS", inclusive="left")
        ten_days = pd.Timedelta(days=10)
        first_dekad_month = [(date, date + ten_days) for date in start_dates]
        second_dekad_month = [(date + ten_days, date + ten_days + ten_days) for date in start_dates]
        end_month = [date + pd.Timedelta(days=calendar.monthrange(date.year, date.month)[1]) for date in start_dates]
        third_dekad_month = list(zip([date + ten_days + ten_days for date in start_dates], end_month))
        intervals = (first_dekad_month + second_dekad_month + third_dekad_month)
        intervals.sort(key=lambda t: t[0])
    elif "month" == period:
        periods = pd.period_range(start, end, freq="M")
        intervals = [(p.to_timestamp(), month_shift(p.to_timestamp(), months=1)) for p in periods]
    elif "season" == period:
        season_start = month_shift(start, months=-(start.month % 3))
        periods = pd.period_range(season_start, end, freq="3M")
        intervals = [(p.to_timestamp(), month_shift(p.to_timestamp(), months=3)) for p in periods]
    elif "tropical-season" == period:
        season_start = month_shift(start, months=-((start.month - 5) % 6))
        periods = pd.period_range(season_start, end, freq="6M")
        intervals = [(p.to_timestamp(), month_shift(p.to_timestamp(), months=6)) for p in periods]
    elif "year" == period:
        offset = timedelta(weeks=52)
        start_dates = pd.date_range(
            start - offset, end, freq="A-DEC", inclusive="left"
        ) + timedelta(days=1)
        end_dates = pd.date_range(
            start, end + offset, freq="A-DEC", inclusive="left"
        ) + timedelta(days=1)
        intervals = zip(start_dates, end_dates)
    else:
        raise ProcessParameterInvalidException('period', 'aggregate_temporal_period',
                                               'No support for a period of type: ' + str(period))
    intervals = [i for i in intervals if i[0] < end]
    _log.info(f"aggregate_temporal_period input: [{start},{end}] - {period} intervals: {intervals}")
    return intervals
