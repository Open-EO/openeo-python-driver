import json
import logging
import re
import textwrap
import traceback
from typing import List, Union, Optional

import flask
import pytest
import re_assert
from re_assert import Matches

from openeo_driver.testing import DictSubSet, caplog_with_custom_formatter
from openeo_driver.util.logging import (
    LOGGING_CONTEXT_BATCH_JOB,
    LOGGING_CONTEXT_FLASK,
    BatchJobLoggingFilter,
    FlaskRequestCorrelationIdLogging,
    ContextBasedExtraInjectingFilter,
    FlaskUserIdLogging,
    just_log_exceptions,
    user_id_trim,
)

from ..conftest import enhanced_logging


_log = logging.getLogger(__name__)


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


def test_filter_flask_request_correlation_id_logging_overriding():
    """Don't overwrite existing request id."""
    with enhanced_logging(format="[%(req_id)s] %(message)s", context=LOGGING_CONTEXT_FLASK) as logs:
        app = flask.Flask(__name__)
        log = logging.getLogger(__name__)

        log.info("Setting up app")

        @app.before_request
        def before_request():
            FlaskRequestCorrelationIdLogging.before_request()

        @app.route("/hello")
        def hello():
            log.warning("Watch out!", extra={FlaskRequestCorrelationIdLogging.LOG_RECORD_ATTR: "hello123"})
            return "Hello world"

        with app.test_client() as client:
            client.get("/hello")

    logs = [l for l in logs.getvalue().split("\n")]
    assert "[no-request] Setting up app" in logs
    assert "[hello123] Watch out!" in logs


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


def _decode_json_lines(lines: Union[List[str], str], strict: bool = True) -> List[dict]:
    """Decode JSON lines data (one JSON dump per line)."""
    if isinstance(lines, str):
        lines = lines.strip("\n").split("\n")
    result = []
    for line in lines:
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError as e:
            if strict:
                raise ValueError(f"Failed to JSON-decode {line!r}") from e
            else:
                _log.warning(f"JSON decode failed on {line=}")
    return result


def _strip_jenkins_distutils_hack_warnings(stderr: Union[str, List[str]]) -> Union[str, List[str]]:
    """
    Strip weird errors from stderr on jenkins due to old setuptools version
    Also see https://github.com/Open-EO/openeo-python-driver/issues/230 and GDD-2879
    """
    # TODO: eliminate this workaround
    as_list = isinstance(stderr, list)
    if as_list:
        stderr = "\n".join(stderr)
    stderr = re.sub(
        r"Error processing line 1 of\s+/[\w/.-]+/distutils-precedence\.pth:"
        r"\s+Traceback.*AttributeError: module '_distutils_hack' has no attribute 'ensure_shim'"
        r"\s+Remainder of file ignored\n?",
        repl="",
        string=stderr,
        flags=re.DOTALL,
    )
    if as_list:
        stderr = stderr.split("\n") if stderr else []
    return stderr


def test_strip_jenkins_distutils_hack_warnings():
    stderr = textwrap.dedent(
        """
        Foo
        Error processing line 1 of
        /usr/local/lib/python3.8/site-packages/distutils-precedence.pth:

          Traceback (most recent call last):
            File "/usr/lib64/python3.8/site.py", line 169, in addpackage
              exec(line)
            File "<string>", line 1, in <module>
          AttributeError: module '_distutils_hack' has no attribute 'ensure_shim'

        Remainder of file ignored
        bar
    """
    )
    assert _strip_jenkins_distutils_hack_warnings(stderr) == "\nFoo\nbar\n"

def test_setup_logging_capture_warnings(pytester):
    script = """
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
    pytester.makepyfile(main=script)
    result = pytester.runpython("main.py")

    records = _decode_json_lines(_strip_jenkins_distutils_hack_warnings(result.errlines))
    assert any(
        r["levelname"] == "WARNING" and "UserWarning: Attention please" in r["message"]
        for r in records
    )
    assert any(r["levelname"] == "INFO" and "That's all" in r["message"] for r in records)


def test_setup_logging_default(pytester):
    script = """
        import logging
        import warnings
        from openeo_driver.util.logging import get_logging_config, setup_logging

        _log = logging.getLogger(__name__)

        def main():
            setup_logging(get_logging_config())
            _log.warning("Hello warning")
            _log.info("Hello info")

        if __name__ == "__main__":
            main()
    """
    pytester.makepyfile(main=script)
    result = pytester.runpython("main.py")

    assert result.outlines == []
    assert _strip_jenkins_distutils_hack_warnings(result.errlines) == [
        re_assert.Matches(r"\[[0-9-]+ [0-9:,]+\] \d+ WARNING in __main__:9 Hello warning"),
        re_assert.Matches(r"\[[0-9-]+ [0-9:,]+\] \d+ INFO in __main__:10 Hello info"),
    ]


def test_setup_logging_basic(pytester):
    script = """
        import logging
        import warnings
        from openeo_driver.util.logging import get_logging_config, setup_logging, LOG_HANDLER_STDERR_BASIC

        _log = logging.getLogger(__name__)

        def main():
            logging_config = get_logging_config(root_handlers=[LOG_HANDLER_STDERR_BASIC])
            setup_logging(logging_config)
            _log.warning("Hello warning")
            _log.info("Hello info")

        if __name__ == "__main__":
            main()
        """
    pytester.makepyfile(main=script)
    result = pytester.runpython("main.py")

    assert result.outlines == []
    assert _strip_jenkins_distutils_hack_warnings(result.errlines) == [
        re_assert.Matches(r"\[[0-9-]+ [0-9:,]+\] \d+ WARNING in __main__:10 Hello warning"),
        re_assert.Matches(r"\[[0-9-]+ [0-9:,]+\] \d+ INFO in __main__:11 Hello info"),
    ]


def test_setup_logging_stderr_json(pytester):
    script = """
        import logging
        import warnings
        from openeo_driver.util.logging import get_logging_config, setup_logging, LOG_HANDLER_STDERR_JSON

        _log = logging.getLogger(__name__)

        def main():
            logging_config = get_logging_config(root_handlers=[LOG_HANDLER_STDERR_JSON])
            setup_logging(logging_config)
            _log.warning("Hello warning")
            _log.info("Hello info")

        if __name__ == "__main__":
            main()
        """
    pytester.makepyfile(main=script)
    result = pytester.runpython("main.py")

    assert result.outlines == []
    stderr_records = _decode_json_lines(_strip_jenkins_distutils_hack_warnings(result.errlines))
    assert stderr_records == [
        DictSubSet({"levelname": "WARNING", "message": "Hello warning", "filename": "main.py", "lineno": 10}),
        DictSubSet({"levelname": "INFO", "message": "Hello info", "filename": "main.py", "lineno": 11}),
    ]


def test_setup_logging_file_json(pytester, tmp_path):
    log_file = tmp_path / "openeo.log"
    script = f"""
        import logging
        import warnings
        from openeo_driver.util.logging import get_logging_config, setup_logging, LOG_HANDLER_FILE_JSON

        _log = logging.getLogger(__name__)

        def main():
            logging_config = get_logging_config(
                root_handlers=[LOG_HANDLER_FILE_JSON],
                log_file={str(log_file)!r}
            )
            setup_logging(logging_config)
            _log.warning("Hello warning")
            _log.info("Hello info")

        if __name__ == "__main__":
            main()
        """
    pytester.makepyfile(main=script)
    result = pytester.runpython("main.py")

    assert result.outlines == []
    assert _strip_jenkins_distutils_hack_warnings(result.errlines) == []
    assert _decode_json_lines(log_file.read_text()) == [
        DictSubSet({"levelname": "WARNING", "message": "Hello warning", "filename": "main.py", "lineno": 13}),
        DictSubSet({"levelname": "INFO", "message": "Hello info", "filename": "main.py", "lineno": 14}),
    ]


def test_setup_logging_rotating_file_json(pytester, tmp_path):
    log_file = tmp_path / "openeo.log"
    script = f"""
        import logging
        import warnings
        from openeo_driver.util.logging import get_logging_config, setup_logging, LOG_HANDLER_ROTATING_FILE_JSON

        _log = logging.getLogger(__name__)

        def main():
            logging_config = get_logging_config(
                root_handlers=[LOG_HANDLER_ROTATING_FILE_JSON],
                log_file={str(log_file)!r},
                rotating_file_max_bytes=1024,
            )
            setup_logging(logging_config)

            for i in range(10):
                _log.warning(f"Hello warning {{i}}")

        if __name__ == "__main__":
            main()
        """
    pytester.makepyfile(main=script)
    result = pytester.runpython("main.py")

    assert result.outlines == []
    assert _strip_jenkins_distutils_hack_warnings(result.errlines) == []

    # Log file has rolled over
    assert _decode_json_lines(log_file.read_text())[0] == DictSubSet(
        {"message": re_assert.Matches("Hello warning [3-9]")},
    )
    # Backup has first log entry
    backup = log_file.with_suffix(".log.1")
    assert backup.exists()
    assert _decode_json_lines(backup.read_text())[0] == DictSubSet(
        {"message": "Hello warning 0"},
    )


def test_setup_logging_capture_threading_exceptions(pytester):
    script = """
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
    pytester.makepyfile(main=script)
    result = pytester.runpython("main.py")

    records = _decode_json_lines(_strip_jenkins_distutils_hack_warnings(result.errlines))
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
    script = """
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
    pytester.makepyfile(main=script)
    result = pytester.runpython("main.py")

    records = _decode_json_lines(_strip_jenkins_distutils_hack_warnings(result.errlines))
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


def test_just_log_exceptions_invalid_logger(caplog):
    caplog.set_level(logging.INFO)
    not_a_logger = None
    with just_log_exceptions(log=not_a_logger):
        raise RuntimeError("Nope")

    expected = (
        "openeo_driver.util.logging",
        logging.ERROR,
        "Failed to do `just_log_exceptions` with log=None: 'NoneType' object is not callable",
    )
    assert caplog.record_tuples == [expected]


def test_just_log_exceptions_extra(caplog):
    class Formatter:
        def format(self, record: logging.LogRecord):
            foo = getattr(record, "foo", None)
            return f"[Foo:{foo}] {record.levelname} {record.message}"

    with caplog_with_custom_formatter(caplog=caplog, format=Formatter()):
        with just_log_exceptions(extra={"foo": "bar"}):
            raise RuntimeError("Nope")

    expected = "[Foo:bar] ERROR In context 'untitled': caught RuntimeError('Nope')\n"
    assert caplog.text == expected


class TestContextBasedExtraInjectingFilter:
    def _build_script(self, src: str, json_fields: Optional[List[str]] = None) -> str:
        """Helper to build script with some common imports and logging setup"""
        json_format = " ".join(f"%({f})s" for f in json_fields or ["levelname", "message"])
        setup = f"""
            import logging
            import pythonjsonlogger.jsonlogger
            from openeo_driver.util.logging import ContextBasedExtraInjectingFilter

            root_loger = logging.getLogger()
            root_loger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            handler.setFormatter(pythonjsonlogger.jsonlogger.JsonFormatter({json_format!r}))
            handler.addFilter(ContextBasedExtraInjectingFilter())
            root_loger.addHandler(handler)
        """
        return textwrap.dedent(setup) + "\n" + textwrap.dedent(src)

    def test_basic(self, pytester):
        script = self._build_script(
            """
            def do_work():
                log = logging.getLogger("bar")
                log.warning("Working here!")

            logger = logging.getLogger("foo")
            logger.info("Hello")
            with ContextBasedExtraInjectingFilter.with_extra_logging(job_id="job-42"):
                logger.info("Let's do some work")
                do_work()
            logger.info("kthxbye")
            """,
            json_fields=["levelname", "name", "message"],
        )
        pytester.makepyfile(main=script)
        result = pytester.runpython("main.py")

        records = _decode_json_lines(_strip_jenkins_distutils_hack_warnings(result.errlines))
        assert records == [
            {"levelname": "INFO", "name": "foo", "message": "Hello"},
            {"levelname": "INFO", "name": "foo", "message": "Let's do some work", "job_id": "job-42"},
            {"levelname": "WARNING", "name": "bar", "message": "Working here!", "job_id": "job-42"},
            {"levelname": "INFO", "name": "foo", "message": "kthxbye"},
        ]

    def test_nesting(self, pytester):
        """
        Test that nesting context is supported: extra fields are added/removed properly
        """
        script = self._build_script(
            """
            logger = logging.getLogger("foo")
            logger.info("Hello")
            with ContextBasedExtraInjectingFilter.with_extra_logging(user="john"):
                jobs = ["job-123", "job-456"]
                logger.info(f"Found jobs {jobs}")
                for job_id in jobs:
                    with ContextBasedExtraInjectingFilter.with_extra_logging(job_id=job_id):
                        logger.info(f"Handling {job_id}")
            logger.info("kthxbye")
            """,
            json_fields=["message"],
        )
        pytester.makepyfile(main=script)
        result = pytester.runpython("main.py")

        records = _decode_json_lines(_strip_jenkins_distutils_hack_warnings(result.errlines))
        assert records == [
            {"message": "Hello"},
            {"message": "Found jobs ['job-123', 'job-456']", "user": "john"},
            {"message": "Handling job-123", "user": "john", "job_id": "job-123"},
            {"message": "Handling job-456", "user": "john", "job_id": "job-456"},
            {"message": "kthxbye"},
        ]

    def test_nesting_override(self, pytester):
        """
        Test that overriding by nesting context is supported
        """
        script = self._build_script(
            """
            logger = logging.getLogger("foo")
            with ContextBasedExtraInjectingFilter.with_extra_logging(user="alice"):
                logger.info("Who knows Alice?")
                with ContextBasedExtraInjectingFilter.with_extra_logging(user="bob"):
                    logger.info("Greetings from Bob")
                logger.info(f"That was fun")
            logger.info("kthxbye")
            """,
            json_fields=["message"],
        )
        pytester.makepyfile(main=script)
        result = pytester.runpython("main.py")

        records = _decode_json_lines(_strip_jenkins_distutils_hack_warnings(result.errlines))
        assert records == [
            {"message": "Who knows Alice?", "user": "alice"},
            {"message": "Greetings from Bob", "user": "bob"},
            {"message": "That was fun", "user": "alice"},
            {"message": "kthxbye"},
        ]

    def test_error_handling(self, pytester):
        """
        Test that extra logging fields are properly cleaned up when an exception occurs in context body
        """
        script = self._build_script(
            """
            logger = logging.getLogger("foo")

            def main():
                logger.info("Hello")
                for job_id in ["job-123", "job-0"]:
                    with ContextBasedExtraInjectingFilter.with_extra_logging(job_id=job_id):
                        logger.info(f"Handling {job_id}")
                        try:
                            rate = 4 / int(job_id.split("-")[1])
                            logger.info(f"rate: {rate:.2f}")
                        except Exception:
                            logger.error(f"Something went wrong with {job_id}")
                            raise
                logger.info("kthxbye")

            if __name__ == "__main__":
                try:
                    main()
                except Exception:
                    logger.error("main failed")
        """
        )
        pytester.makepyfile(main=script)
        result = pytester.runpython("main.py")

        records = _decode_json_lines(_strip_jenkins_distutils_hack_warnings(result.errlines))
        assert records == [
            {"levelname": "INFO", "message": "Hello"},
            {"levelname": "INFO", "message": "Handling job-123", "job_id": "job-123"},
            {"levelname": "INFO", "message": "rate: 0.03", "job_id": "job-123"},
            {"levelname": "INFO", "message": "Handling job-0", "job_id": "job-0"},
            # This error log contains "job_id" field
            {"levelname": "ERROR", "message": "Something went wrong with job-0", "job_id": "job-0"},
            # This error log doesn't (it's cleaned up properly)
            {"levelname": "ERROR", "message": "main failed"},
        ]

    # TODO: test threading behaviour
