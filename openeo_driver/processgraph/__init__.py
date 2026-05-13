# Re-export everything from definitions for backward compatibility.
# The original openeo_driver/processgraph.py has been moved to
# openeo_driver/processgraph/definitions.py.
from openeo_driver.processgraph.definitions import (
    ProcessGraphFlatDict,
    ProcessDefinition,
    get_process_definition_from_url,
    extract_default_job_options_from_process_graph,
)
