import contextlib
import functools
import logging
import logging.config
import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Callable

import flask
import pythonjsonlogger.jsonlogger

import openeo.udf.debug
from openeo_driver.utils import generate_unique_id

_log = logging.getLogger(__name__)

LOGGING_CONTEXT_FLASK = "flask"
LOGGING_CONTEXT_BATCH_JOB = "batch_job"

LOG_FORMAT_BASIC = "[%(asctime)s] %(process)s %(levelname)s in %(name)s:%(lineno)s %(message)s"

# This fake `format` string is the JsonFormatter way to list expected fields in json records
# Note: JsonFormatter will output `extra` properties like 'job_id' even if they're not included in
# its `format` string (https://github.com/madzak/python-json-logger/issues/97)
JSON_LOGGER_DEFAULT_FORMAT = "%(message)s %(levelname)s %(name)s %(created)s %(filename)s %(lineno)s %(process)s"


# Basic (like default logging): simple text format (on stderr)
LOG_HANDLER_STDERR_BASIC = "basic"
# Text format logging on stdout
LOG_HANDLER_STDOUT_BASIC = "stdout_basic"
# Flask-style WSGI logging (text format on stderr)
LOG_HANDLER_STDERR_WSGI = "wsgi"
# JSON-logging on stderr
LOG_HANDLER_STDERR_JSON = "stderr_json"
# JSON-logging on stdout
LOG_HANDLER_STDOUT_JSON = "stdout_json"
# JSON-logging to a file
LOG_HANDLER_FILE_JSON = "file_json"
# JSON-logging to a rotating/rolling file
LOG_HANDLER_ROTATING_FILE_JSON = "rotating_file_json"


def get_logging_config(
    *,
    root_handlers: Optional[List[str]] = None,
    loggers: Optional[Dict[str, dict]] = None,
    handler_default_level: str = "DEBUG",
    context: str = LOGGING_CONTEXT_FLASK,
    root_level: str = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    log_dir: Optional[Union[str, Path]] = None,
    rotating_file_max_bytes: int = 10 * 1024 * 1024,
    rotating_file_backup_count: int = 1,
) -> dict:
    """Construct logging config dict to be loaded with `logging.config.dictConfig`"""

    # Merge log levels per logger with some defaults
    default_loggers = {
        "gunicorn": {"level": "INFO"},
        "werkzeug": {"level": "INFO"},
        "kazoo": {"level": "WARN"},
        "py4j": {"level": "INFO"},
        "openeo.udf.debug": {"level": "DEBUG"}
    }
    loggers = {**default_loggers, **(loggers or {})}

    if context == LOGGING_CONTEXT_FLASK:
        json_filters = [
            "FlaskRequestCorrelationIdLogging",
            "FlaskUserIdLogging",
        ]
    elif context == LOGGING_CONTEXT_BATCH_JOB:
        json_filters = ["BatchJobLoggingFilter"]
    else:
        json_filters = []

    if not log_file:
        if not log_dir:
            # TODO: this LOG_DIRS env var is originally something YARN specific (see 1b441cae)
            #       can we eliminate env var usage here to improve traceability and predictability?
            log_dir = os.environ.get("LOG_DIRS", ".").split(",")[0]
        log_file = Path(log_dir) / "openeo_python.log"

    config = {
        "version": 1,
        "root": {
            "level": root_level,
            # TODO: `get_logging_config` is also used outside of WSGI contexts: use LOG_HANDLER_STDERR_BASIC by default
            "handlers": (root_handlers or [LOG_HANDLER_STDERR_WSGI]),
        },
        "loggers": loggers,
        "handlers": {
            # Simple basic handler: basic format on stderr
            LOG_HANDLER_STDERR_BASIC: {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
                "level": handler_default_level,
                "formatter": "basic",
            },
            LOG_HANDLER_STDOUT_BASIC: {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
                "level": handler_default_level,
                "formatter": "basic",
            },
            LOG_HANDLER_STDERR_WSGI: {
                "class": "logging.StreamHandler",
                "stream": "ext://flask.logging.wsgi_errors_stream",
                "level": handler_default_level,
                "formatter": "basic"
            },
            # TODO: remove this legacy, deprecated handler
            "json": {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
                "level": handler_default_level,
                "filters": json_filters,
                "formatter": "json",
            },
            LOG_HANDLER_STDERR_JSON: {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
                "level": handler_default_level,
                "filters": json_filters,
                "formatter": "json",
            },
            LOG_HANDLER_STDOUT_JSON: {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
                "level": handler_default_level,
                "filters": json_filters,
                "formatter": "json",
            },
            LOG_HANDLER_FILE_JSON: {
                "class": "logging.FileHandler",
                "filename": log_file,
                "level": handler_default_level,
                "filters": json_filters,
                "formatter": "json",
            },
            LOG_HANDLER_ROTATING_FILE_JSON: {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": log_file,
                "level": handler_default_level,
                "filters": json_filters,
                "formatter": "json",
                "maxBytes": rotating_file_max_bytes,
                "backupCount": rotating_file_backup_count,
            },
            # TODO: allow adding custom handlers (e.g. rotating file)
        },
        "filters": {
            "FlaskRequestCorrelationIdLogging": {"()": FlaskRequestCorrelationIdLogging},
            "FlaskUserIdLogging": {"()": FlaskUserIdLogging},
            "BatchJobLoggingFilter": {"()": BatchJobLoggingFilter},
        },
        "formatters": {
            "basic": {
                "()": UtcFormatter,
                "format": LOG_FORMAT_BASIC,
            },
            "json": {
                "()": pythonjsonlogger.jsonlogger.JsonFormatter,
                # This fake `format` string is the way to list expected fields in json records
                "format": JSON_LOGGER_DEFAULT_FORMAT,
            },
        },
        # Keep existing loggers alive (e.g. werkzeug, gunicorn, ...)
        "disable_existing_loggers": False,
    }
    return config


def setup_logging(
        config: Optional[dict] = None,
        force=False,
        capture_warnings=True,
        capture_threading_exceptions=True,
        capture_unhandled_exceptions=True,
):
    if capture_warnings is not None:
        logging.captureWarnings(capture=capture_warnings)

    if config is None:
        config = get_logging_config()
    if not logging.getLogger().handlers or force:
        logging.config.dictConfig(config)

    is_openeo_debug_enabled = os.environ.get("OPENEO_LOGGING_THRESHOLD", "INFO").lower() == "debug"
    if is_openeo_debug_enabled:
        _log.log(level=_log.getEffectiveLevel(), msg="setup_logging")
        for logger_name in config.get("loggers", {}).keys():
            show_log_level(logger=logger_name)
        _log.debug(f"root handlers: {logging.getLogger().handlers}")

    if capture_threading_exceptions:
        if hasattr(threading, "excepthook"):
            # From Python 3.8
            threading.excepthook = _threading_excepthook
        else:
            _log.warning("No support for capturing threading exceptions")

    if capture_unhandled_exceptions:
        _log.debug(f"Overriding sys.excepthook with {_sys_excepthook} (was {sys.excepthook})")
        sys.excepthook = _sys_excepthook


def show_log_level(logger: Union[logging.Logger, str]):
    """Helper to show (effective) threshold log level of a logger."""
    if isinstance(logger, str):
        logger = logging.getLogger(logger)
    level = logger.getEffectiveLevel()
    msg = f"Effective log level of {logger.name!r}: {logging.getLevelName(level)}"
    logger.log(level=level, msg=msg)


def _threading_excepthook(args):
    # Based on threading.excepthook default implementation
    if args.thread is not None:
        name = args.thread.name
    else:
        name = threading.get_ident()
    _log.error(f"Exception {args.exc_type.__name__} in thread {name}: {args.exc_value!r}", exc_info=True)


def _sys_excepthook(exc_type, exc_value, traceback):
    _log.error(f"Unhandled {exc_type.__name__} exception: {exc_value!r}", exc_info=(exc_type, exc_value, traceback))


class UtcFormatter(logging.Formatter):
    """Log formatter that uses UTC instead of local time."""
    # based on https://docs.python.org/3/howto/logging-cookbook.html#formatting-times-using-utc-gmt-via-configuration
    converter = time.gmtime


class FlaskRequestCorrelationIdLogging(logging.Filter):
    """
    Python logging plugin to include a Flask request correlation id
    automatically in log records.

    Usage instructions:

        - Add `before_request` handler to Flask app, e.g.:

            @app.before_request
            def before_request():
                RequestCorrelationIdLogging.before_request()

        - Add filter to relevant logging handler, e.g.:

            handler.addFilter(RequestCorrelationIdLogging())

        - Use "req_id" field in logging formatter, e.g.:

            formatter = logging.Formatter("[%(req_id)s] %(message)s")
            handler.setFormatter(formatter)
    """

    FLASK_G_ATTR = "request_correlation_id"
    LOG_RECORD_ATTR = "req_id"

    @classmethod
    def _build_request_id(cls) -> str:
        """Generate/extract request correlation id."""
        # TODO: get correlation id "from upstream/context" (e.g. nginx headers)
        return generate_unique_id(prefix="r")

    @classmethod
    def before_request(cls):
        """Flask `before_request` handler: store request correlation id in Flask request global `g`."""
        setattr(flask.g, cls.FLASK_G_ATTR, cls._build_request_id())

    @classmethod
    def get_request_id(cls) -> str:
        """Get request correlation id as stored in Flask request global `g`."""
        if flask.has_request_context():
            return flask.g.get(cls.FLASK_G_ATTR, "n/a")
        else:
            return "no-request"

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter a log record (logging.Filter API)."""
        if not hasattr(record, self.LOG_RECORD_ATTR):
            setattr(record, self.LOG_RECORD_ATTR, self.get_request_id())
        return True


def user_id_trim(user_id: str, size=8) -> str:
    """Trim user id (to reduce logging volume and for a touch of user_id obfuscation)."""
    if len(user_id) > size:
        user_id = user_id[:size] + '...'
    return user_id


class FlaskUserIdLogging(logging.Filter):
    """
    Python logging plugin to include a user id automatically in log records in Flask context.
    """

    FLASK_G_ATTR = "current_user_id"
    LOG_RECORD_ATTR = "user_id"

    @classmethod
    def set_user_id(cls, user_id: str):
        """Store user id in Flask request global `g`."""
        _log.debug(f"{cls} storing user id {user_id!r} on {flask.g}")
        setattr(flask.g, cls.FLASK_G_ATTR, user_id)

    @classmethod
    def get_user_id(cls) -> Union[str, None]:
        """Get user id as stored in Flask request global `g`."""
        if flask._app_ctx_stack.top:
            return flask.g.get(cls.FLASK_G_ATTR, None)

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter a log record (logging.Filter API)."""
        setattr(record, self.LOG_RECORD_ATTR, self.get_user_id())
        return True


class BatchJobLoggingFilter(logging.Filter):
    """
    Python logging plugin to inject data (such as user id and batch job id) in a batch job context.
    Internally stores/manages data as global class data, assuming the data is specific
    for the current (batch job) process and does not change once set.
    """

    data = {}

    @classmethod
    def set(cls, field: str, value: str):
        cls.data[field] = value

    @classmethod
    def reset(cls):
        cls.data = {}

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter a log record (logging.Filter API)."""
        for field, value in self.data.items():
            setattr(record, field, value)
        return True


@contextlib.contextmanager
def just_log_exceptions(
    log: Union[logging.Logger, Callable, str, int] = logging.ERROR,
    name: Optional[str] = "untitled",
    extra: Optional[dict] = None,
):
    """
    Context manager to catch any exception (if any) and just log them.

    :param log: one of
        - a `logging.Logger` instance,
        - a callable (e.g. use a logging method like `my_logger.warning` to define both logger and level)
        - a log level as string or int (e.g. logging constant like `logging.WARNING`)
    :param name: name of the context (to be used as reference in error log)
    """
    if isinstance(log, logging.Logger):
        log = log.error
    elif isinstance(log, int):
        log = functools.partial(_log.log, log)
    elif isinstance(log, str):
        log = functools.partial(_log.log, logging.getLevelName(log))
    try:
        yield
    except Exception as e:
        try:
            log(f"In context {name!r}: caught {e!r}", extra=extra, exc_info=True)
        except Exception as e:
            _log.error(
                f"Failed to do `just_log_exceptions` with {log=}: {e}", exc_info=True
            )
