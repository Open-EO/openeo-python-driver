import logging
import logging.config
import os
import sys
import threading
import time
from pathlib import Path
from typing import List, Dict, Optional, Union

import flask
import pythonjsonlogger.jsonlogger

from openeo_driver.utils import generate_unique_id

_log = logging.getLogger(__name__)

LOGGING_CONTEXT_FLASK = "flask"
LOGGING_CONTEXT_BATCH_JOB = "batch_job"

# This fake `format` string is the JsonFormatter way to list expected fields in json records
# Note: JsonFormatter will output `extra` properties like 'job_id' even if they're not included in
# its `format` string (https://github.com/madzak/python-json-logger/issues/97)
JSON_LOGGER_DEFAULT_FORMAT = "%(message)s %(levelname)s %(name)s %(created)s %(filename)s %(lineno)s %(process)s"


def get_logging_config(
        root_handlers: Optional[List[str]] = None,
        loggers: Optional[Dict[str, dict]] = None,
        handler_default_level: str = "DEBUG",
        context: str = LOGGING_CONTEXT_FLASK,
        root_level: str = "INFO",
) -> dict:
    """Construct logging config dict to be loaded with `logging.config.dictConfig`"""

    # Merge log levels per logger with some defaults
    default_loggers = {
        "gunicorn": {"level": "INFO"},
        "werkzeug": {"level": "INFO"},
        "kazoo": {"level": "WARN"},
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

    log_dirs = os.environ.get("LOG_DIRS")
    log_dir = log_dirs.split(",")[0] if log_dirs is not None else "."
    log_file = Path(log_dir) / "openeo_python.log"

    config = {
        "version": 1,
        "root": {
            "level": root_level,
            "handlers": (root_handlers or ["wsgi"]),
        },
        "loggers": loggers,
        "handlers": {
            "wsgi": {
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
            "stderr_json": {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
                "level": handler_default_level,
                "filters": json_filters,
                "formatter": "json",
            },
            "stdout_json": {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
                "level": handler_default_level,
                "filters": json_filters,
                "formatter": "json",
            },
            "file_json": {
                "class": "logging.FileHandler",
                "filename": log_file,
                "level": handler_default_level,
                "filters": json_filters,
                "formatter": "json",
            },
        },
        "filters": {
            "FlaskRequestCorrelationIdLogging": {"()": FlaskRequestCorrelationIdLogging},
            "FlaskUserIdLogging": {"()": FlaskUserIdLogging},
            "BatchJobLoggingFilter": {"()": BatchJobLoggingFilter},
        },
        "formatters": {
            "basic": {
                "()": UtcFormatter,
                "format": "[%(asctime)s] %(process)s %(levelname)s in %(name)s: %(message)s",
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

    _log.log(level=_log.getEffectiveLevel(), msg="setup_logging")
    for logger_name in config.get("loggers", {}).keys():
        show_log_level(logger=logger_name)
    _log.info(f"root handlers: {logging.getLogger().handlers}")

    if capture_threading_exceptions:
        if hasattr(threading, "excepthook"):
            # From Python 3.8
            threading.excepthook = _threading_excepthook
        else:
            _log.warning("No support for capturing threading exceptions")

    if capture_unhandled_exceptions:
        _log.info(f"Overriding sys.excepthook with {_sys_excepthook} (was {sys.excepthook})")
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
        if flask._app_ctx_stack.top is None:
            return "no-request"
        else:
            return flask.g.get(cls.FLASK_G_ATTR, "n/a")

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter a log record (logging.Filter API)."""
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
