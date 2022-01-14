import logging
import uuid

import flask


class RequestCorrelationIdLogging(logging.Filter):
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
