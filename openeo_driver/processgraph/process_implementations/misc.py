"""Miscellaneous process implementations."""
import datetime
import logging
import time
from typing import Union

from dateutil.relativedelta import relativedelta
from openeo.util import rfc3339

from openeo_driver.datacube import DriverDataCube
from openeo_driver.dry_run import DryRunDataTracer
from openeo_driver.errors import FeatureUnsupportedException, OpenEOApiException, ProcessParameterInvalidException
from openeo_driver.processes import ProcessArgs
from openeo_driver.processgraph.registry import (
    ENV_DRY_RUN_TRACER,
    ENV_FINAL_RESULT,
    ENV_SAVE_RESULT,
    NoPythonImplementationError,
    non_standard_process,
    process,
    process_registry_100,
    process_registry_2xx,
)
from openeo_driver.processes import ProcessSpec
from openeo_driver.save_result import NullResult
from openeo_driver.specs import read_spec
from openeo_driver.utils import EvalEnv

_log = logging.getLogger(__name__)


@process
def constant(args: ProcessArgs, env: EvalEnv):
    return args.get_required("x")


@process_registry_100.add_function(spec=read_spec("openeo-processes/1.x/proposals/inspect.json"))
@process_registry_2xx.add_function(spec=read_spec("openeo-processes/2.x/proposals/inspect.json"))
def inspect(args: ProcessArgs, env: EvalEnv):
    data = args.get_required("data")
    message = args.get_optional("message", default="")
    code = args.get_optional("code", default="User")
    level = args.get_optional("level", default="info")
    if message:
        _log.log(level=logging.getLevelName(level.upper()), msg=message)
    data_message = str(data)
    if isinstance(data, DriverDataCube):
        data_message = str(data.metadata)
    _log.log(level=logging.getLevelName(level.upper()), msg=data_message)
    return data


@non_standard_process(
    ProcessSpec("sleep", description="Sleep for given amount of seconds (and just pass-through given data).")
        .param('data', description="Data to pass through.", schema={}, required=False)
        .param('seconds', description="Number of seconds to sleep.", schema={"type": "number"}, required=True)
        .returns("Original data", schema={})
)
def sleep(args: ProcessArgs, env: EvalEnv):
    data = args.get_required("data")
    seconds = args.get_required("seconds", expected_type=(int, float))
    dry_run_tracer: DryRunDataTracer = env.get(ENV_DRY_RUN_TRACER)
    if not dry_run_tracer:
        _log.info("Sleeping {s} seconds".format(s=seconds))
        time.sleep(seconds)
    return data


@non_standard_process(
    ProcessSpec("discard_result", description="Discards given data. Used for side-effecting purposes.")
        .param('data', description="Data to discard.", schema={}, required=False)
        .returns("Nothing", schema={})
)
def discard_result(args: ProcessArgs, env: EvalEnv):
    return NullResult()


@process_registry_100.add_function(spec=read_spec("openeo-processes/1.x/proposals/date_shift.json"))
@process_registry_2xx.add_function(spec=read_spec("openeo-processes/2.x/proposals/date_shift.json"))
def date_shift(args: ProcessArgs, env: EvalEnv) -> str:
    date = rfc3339.parse_date_or_datetime(args.get_required("date", expected_type=str))
    value = int(args.get_required("value", expected_type=int))
    unit = args.get_enum("unit", options={"year", "month", "week", "day", "hour", "minute", "second", "millisecond"})
    if unit == "millisecond":
        raise FeatureUnsupportedException(message="Millisecond unit is not supported in date_shift")
    shifted = date + relativedelta(**{unit + "s": value})
    if type(date) is datetime.date and type(shifted) is datetime.datetime:
        shifted = shifted.date()
    return rfc3339.normalize(shifted)


@process_registry_2xx.add_function(spec=read_spec("openeo-processes/2.x/proposals/date_between.json"))
def date_between(args: ProcessArgs, env: EvalEnv) -> bool:
    raise NoPythonImplementationError


@process_registry_100.add_simple_function(name="if")
@process_registry_2xx.add_simple_function(name="if")
def if_(value: Union[bool, None], accept, reject=None):
    return accept if value else reject
