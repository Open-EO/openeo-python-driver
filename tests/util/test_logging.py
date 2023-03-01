import json
import logging
import re
import traceback
from typing import List

import flask
import pytest
from re_assert import Matches

from openeo_driver.util.logging import (
    LOGGING_CONTEXT_BATCH_JOB,
    LOGGING_CONTEXT_FLASK,
    BatchJobLoggingFilter,
    FlaskRequestCorrelationIdLogging,
    FlaskUserIdLogging,
    just_log_exceptions,
    user_id_trim,
)

from ..conftest import enhanced_logging


def test_filter_flask_request_correlation_id_logging():
    with enhanced_logging(format="[%(req_id)s] %(message)s", context=LOGGING_CONTEXT_FLASK) as logs:
        app = flask.Flask(__name__)
        log = logging.getLogger(__name__)

        log.info("Setting up app")

        @app.before_request
        def before_request():
            FlaskRequestCorrelationIdLogging.before_request()

        @app.route("/hello")
        def hello():
            log.warning("Watch out!")
            return "Hello world"

        with app.test_client() as client:
            client.get("/hello")

    logs = [l for l in logs.getvalue().split("\n")]
    assert "[no-request] Setting up app" in logs
    assert "[123-456] Watch out!" in logs


def test_filter_flask_user_id_logging():
    with enhanced_logging(format="[%(user_id)s] %(message)s", context=LOGGING_CONTEXT_FLASK) as logs:
        app = flask.Flask(__name__)
        log = logging.getLogger(__name__)

        log.info("Setting up app")

        @app.route("/public")
        def public():
            log.info("public stuff")
            return "Hello world"

        @app.route("/private")
        def private():
            FlaskUserIdLogging.set_user_id("john")
            log.info("private stuff")
            return "Hello John"

        with app.test_client() as client:
            client.get("/public")
            client.get("/private")
            client.get("/public")

    logs = [l for l in logs.getvalue().split("\n") if "stuff" in l]
    assert logs == ["[None] public stuff", "[john] private stuff", "[None] public stuff"]


def test_filter_batch_job_logging():
    with enhanced_logging(json=True, context=LOGGING_CONTEXT_BATCH_JOB) as logs:
        BatchJobLoggingFilter.reset()
        log = logging.getLogger(__name__)

        log.info("Some set up")
        BatchJobLoggingFilter.set("user_id", "j0hnD03")
        BatchJobLoggingFilter.set("job_id", "job-42")
        log.info("Doing the work")

    logs = [json.loads(l) for l in logs.getvalue().strip().split("\n")]
    assert logs == [
        {"message": "Some set up"},
        {"message": "Doing the work", "user_id": "j0hnD03", "job_id": "job-42"},
    ]


def test_json_logging_exc_info():
    logger = logging.getLogger(__name__)

    def foo():
        return 4 / 0

    def bar():
        try:
            foo()
        except Exception:
            logger.info("hmm that didn't work", exc_info=True)

    with enhanced_logging(json=True, context="test") as logs:
        bar()

    logs = [json.loads(l) for l in logs.getvalue().strip().split("\n")]
    assert logs == [
        {
            "message": "hmm that didn't work",
            "exc_info": Matches(
                r"Traceback \(most recent call last\):.*"
                r"  File .*/test_logging.py.*, in bar.*"
                r"    foo().*"
                r"  File .*/test_logging.py.*, in foo.*"
                r"    return 4 / 0.*"
                "ZeroDivisionError: division by zero",
                flags=re.DOTALL,
            ),
        }
    ]


def test_json_logging_stack_info():
    logger = logging.getLogger(__name__)

    def foo():
        logger.info("Hello", stack_info=True)

    def bar():
        foo()

    with enhanced_logging(json=True, context="test") as logs:
        bar()
    logs = [json.loads(l) for l in logs.getvalue().strip().split("\n")]
    assert logs == [
        {
            "message": "Hello",
            "stack_info": Matches(
                r"Stack \(most recent call last\):.*"
                r"  File .*/test_logging.py.*, in test_json_logging_stack_info.*"
                r"    bar\(\).*"
                r"  File .*/test_logging.py.*, in bar.*"
                r"    foo\(\).*"
                r"  File .*/test_logging.py.*, in foo.*"
                r'    logger.info\("Hello", stack_info=True\)',
                flags=re.DOTALL,
            ),
        }
    ]


def test_user_id_trim():
    assert user_id_trim("pol") == "pol"
    assert user_id_trim("536e61f6fb8489946ab99ed3a028") == "536e61f6..."


def _decode_json_lines(lines: List[str]) -> List[dict]:
    result = []
    for line in lines:
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return result


def test_setup_logging_capture_warnings(pytester):
    pytester.makepyfile(
        main="""
        import logging
        import warnings
        from openeo_driver.util.logging import get_logging_config, setup_logging

        _log = logging.getLogger(__name__)

        def main():
            setup_logging(
                get_logging_config(root_handlers=["stderr_json"]),
                capture_warnings=True,
            )
            warnings.warn("Attention please")
            _log.info("That's all")

        if __name__ == "__main__":
            main()
    """
    )
    result = pytester.runpython("main.py")

    records = _decode_json_lines(result.errlines)
    assert any(
        r["levelname"] == "WARNING" and "UserWarning: Attention please" in r["message"]
        for r in records
    )
    assert any(r["levelname"] == "INFO" and "That's all" in r["message"] for r in records)


def test_setup_logging_capture_threading_exceptions(pytester):
    pytester.makepyfile(
        main="""
        import logging
        import threading
        from openeo_driver.util.logging import get_logging_config, setup_logging

        _log = logging.getLogger(__name__)

        def work():
            return 4 / 0

        def main():
            setup_logging(
                get_logging_config(root_handlers=["stderr_json"]),
                capture_threading_exceptions=True,
            )

            thread = threading.Thread(target=work)
            thread.start()
            thread.join()
            _log.info("That's all")

        if __name__ == "__main__":
            main()
    """
    )
    result = pytester.runpython("main.py")

    records = _decode_json_lines(result.errlines)
    assert any(
        r["levelname"] == "ERROR" and "ZeroDivisionError in thread" in r["message"]
        for r in records
    )
    assert any(r["levelname"] == "INFO" and "That's all" in r["message"] for r in records)


@pytest.mark.skipif(
    any("pydevd.py" in f.filename for f in traceback.extract_stack()[:5]),
    reason="When running test in PyDev debugger, pydev catches the exceptions we want to leave uncaught."
)
def test_setup_logging_capture_unhandled_exceptions(pytester):
    pytester.makepyfile(
        main="""
        import logging
        import threading
        from openeo_driver.util.logging import get_logging_config, setup_logging

        def main():
            setup_logging(
                get_logging_config(root_handlers=["stderr_json"]),
                capture_unhandled_exceptions=True,
            )
            x = 4 / 0

        if __name__ == "__main__":
            main()
    """
    )
    result = pytester.runpython("main.py")

    records = _decode_json_lines(result.errlines)
    assert any(
        r["levelname"] == "ERROR" and "Unhandled ZeroDivisionError exception" in r["message"]
        for r in records
    )


def test_just_log_exceptions_default(caplog):
    with just_log_exceptions():
        x = 4 / 0

    expected = (
        "openeo_driver.util.logging",
        logging.ERROR,
        "In context 'untitled': caught ZeroDivisionError('division by zero')",
    )
    assert caplog.record_tuples == [expected]


def test_just_log_exceptions_name(caplog):
    with just_log_exceptions(name="mathzz"):
        x = 4 / 0

    expected = (
        "openeo_driver.util.logging",
        logging.ERROR,
        "In context 'mathzz': caught ZeroDivisionError('division by zero')",
    )
    assert caplog.record_tuples == [expected]


def test_just_log_exceptions_logger(caplog):
    log = logging.getLogger("foo.dothetest")

    with just_log_exceptions(log=log):
        x = 4 / 0

    expected = (
        "foo.dothetest",
        logging.ERROR,
        "In context 'untitled': caught ZeroDivisionError('division by zero')",
    )
    assert caplog.record_tuples == [expected]


def test_just_log_exceptions_logger_method(caplog):
    log = logging.getLogger("foo.dothetest")
    with just_log_exceptions(log=log.warning):
        x = 4 / 0

    expected = (
        "foo.dothetest",
        logging.WARNING,
        "In context 'untitled': caught ZeroDivisionError('division by zero')",
    )
    assert caplog.record_tuples == [expected]


@pytest.mark.parametrize(
    ["level"],
    [(logging.INFO,), ("INFO",)],
)
def test_just_log_exceptions_log_level(caplog, level):
    caplog.set_level(logging.INFO)
    with just_log_exceptions(log=level):
        x = 4 / 0

    expected = (
        "openeo_driver.util.logging",
        logging.INFO,
        "In context 'untitled': caught ZeroDivisionError('division by zero')",
    )
    assert caplog.record_tuples == [expected]
