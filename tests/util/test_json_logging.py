import json
import logging

import pytest

from openeo_driver.util.logging import get_logging_config


@pytest.mark.parametrize(
    "detail, expected_truncation",
    [
        pytest.param("short", False, id="short"),
        pytest.param("x" * 34_000, True, id="ascii"),
        pytest.param("\U0001f30a" * 8_500, True, id="unicode"),
    ],
)
def test_json_logging_limits_exception_trace_size(caplog, detail, expected_truncation):
    logger = logging.getLogger(__name__)
    with caplog.at_level(logging.ERROR, logger=logger.name):
        try:
            raise ValueError(f"{detail} (end of exception)")
        except ValueError:
            logger.exception("Batch job failed", extra={"job_id": "job-123"})
    (record,) = caplog.records
    config = get_logging_config()["formatters"]["json"]
    formatter = config["()"](fmt=config["format"])

    log_data = json.loads(formatter.format(record))

    assert log_data["message"] == "Batch job failed"
    assert log_data["job_id"] == "job-123"
    trace = log_data["exc_info"]
    assert len(trace.encode("utf-8")) <= 24 * 1024
    full_trace = logging.Formatter().formatException(record.exc_info)
    assert logging.Formatter().format(record).endswith(full_trace)
    if expected_truncation:
        assert trace.startswith("Traceback (most recent call last):")
        assert "\n... traceback truncated ...\n" in trace
        assert trace.endswith(" (end of exception)")
        assert "\ufffd" not in trace
    else:
        assert trace == full_trace
