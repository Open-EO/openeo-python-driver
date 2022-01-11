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
    def get_correlation_id(cls):
        """Generate/extract request correlation id."""
        # TODO: get correlation id "from upstream/context" (e.g. nginx headers)
        return str(uuid.uuid4())

    @classmethod
    def before_request(cls):
        """Flask `before_request` handler: store correlation id in Flask request global `g`."""
        setattr(flask.g, cls.FLASK_G_ATTR, cls.get_correlation_id())

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter a log record (logging.Filter API)."""
        if flask._app_ctx_stack.top is None:
            corr_id = "no-request"
        else:
            corr_id = flask.g.get(self.FLASK_G_ATTR, "n/a")
        if corr_id:
            setattr(record, self.LOG_RECORD_ATTR, corr_id)

        return True
