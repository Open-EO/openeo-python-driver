"""
Minimal, numpy-only implementation of a selection of openEO processes.

Adapted from the (unmaintained) `openeo-processes-python` project, release 0.0.4.
Importing this package registers the implementations; use `get_process`/`has_process`
to look them up by openEO process id (e.g. "and", "add", "array_element").
For backwards compatibility with `openeo_processes`, the functions are also available
as attributes with their Python names (e.g. `and_`, `not_`, `add`).
"""

from openeo_driver.vendor.processes_python.utils import (  # noqa: F401
    get_process,
    has_process,
    process,
    eval_datatype,
    isnull,
)
from openeo_driver.vendor.processes_python.logic import *  # noqa: F401,F403
from openeo_driver.vendor.processes_python.arrays import *  # noqa: F401,F403
from openeo_driver.vendor.processes_python.comparison import *  # noqa: F401,F403
from openeo_driver.vendor.processes_python.math import *  # noqa: F401,F403

