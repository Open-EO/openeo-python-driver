import json
import logging

import flask

from openeo_driver.util.logging import FlaskUserIdLogging, FlaskRequestCorrelationIdLogging, BatchJobLoggingFilter, \
    LOGGING_CONTEXT_FLASK, LOGGING_CONTEXT_BATCH_JOB
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
