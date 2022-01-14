import logging
from unittest import mock

import flask

from openeo_driver.util.logging import RequestCorrelationIdLogging


def test_request_correlation_id_logging(caplog):
    caplog.set_level(logging.INFO)
    caplog.handler.addFilter(RequestCorrelationIdLogging())
    caplog.handler.setFormatter(logging.Formatter("[%(req_id)s] %(message)s"))

    app = flask.Flask(__name__)
    log = logging.getLogger(__name__)

    log.info("Setting up app")

    @app.before_request
    def before_request():
        RequestCorrelationIdLogging.before_request()

    @app.route("/hello")
    def hello():
        log.warning("Watch out!")
        return "Hello world"

    with mock.patch.object(RequestCorrelationIdLogging, "_build_request_id", return_value="1234-5678"):
        with app.test_client() as client:
            client.get("/hello")

    assert "[no-request] Setting up ap" in caplog.text
    assert "[1234-5678] Watch out!" in caplog.text
