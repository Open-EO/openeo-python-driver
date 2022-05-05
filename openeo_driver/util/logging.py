import logging
import logging.config
import threading
import time
import uuid
from typing import List, Dict, Optional, Union

import flask
import pythonjsonlogger.jsonlogger

_log = logging.getLogger(__name__)

LOGGING_CONTEXT_FLASK = "flask"
LOGGING_CONTEXT_BATCH_JOB = "batch_job"


def get_logging_config(
        root_handlers: Optional[List[str]] = None,
        loggers: Optional[Dict[str, dict]] = None,
        handler_default_level: str = "DEBUG",
        context: str = LOGGING_CONTEXT_FLASK,
) -> dict:
    """Construct logging config dict to be loaded with `logging.config.dictConfig`"""

    # Merge log levels per logger with some defaults
    default_loggers = {
        "gunicorn": {"level": "INFO"},
        "openeo": {"level": "INFO"},
        "openeo_driver": {"level": "INFO"},
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

    config = {
        "version": 1,
        "root": {
            "level": "INFO",
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
            "json": {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
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
                "format": "%(message)s %(levelname)s %(name)s %(created)s %(filename)s %(lineno)s %(process)s",
            },
        },
        # Keep existing loggers alive (e.g. werkzeug, gunicorn, ...)
        "disable_existing_loggers": False,
    }
    return config


def setup_logging(config: Optional[dict] = None, force=False, capture_warnings=True, capture_threading_exceptions=True):
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
    _log.error(f"Exception {args.exc_type} in thread {name}", exc_info=True)


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
        return str(uuid.uuid4())

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
        _log.info(f"{cls} storing user id {user_id} on {flask.g}")
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
