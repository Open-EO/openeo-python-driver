import contextlib
import io
import logging
import os
import time
from typing import Optional
from unittest import mock

import flask
import pytest
import pythonjsonlogger.jsonlogger

import openeo_driver.config.load
import openeo_driver.dummy.dummy_config
from openeo_driver.backend import UserDefinedProcesses
from openeo_driver.config import OpenEoBackendConfig
from openeo_driver.dummy.dummy_backend import DummyBackendImplementation, DummyProcessing
from openeo_driver.testing import UrllibMocker, config_overrides
from openeo_driver.util.logging import (
    LOGGING_CONTEXT_BATCH_JOB,
    LOGGING_CONTEXT_FLASK,
    FlaskRequestCorrelationIdLogging,
    GlobalExtraLoggingFilter,
    FlaskUserIdLogging,
)
from openeo_driver.views import build_app

pytest_plugins = "pytester"


def pytest_configure(config):
    # Isolate tests from the host machine’s timezone
    os.environ["TZ"] = "UTC"
    time.tzset()

    # Load dummy OpenEoBackendConfig by default
    os.environ["OPENEO_BACKEND_CONFIG"] = openeo_driver.dummy.dummy_config.__file__


@pytest.fixture
def backend_config_overrides() -> Optional[dict]:
    # No overrides by default
    return None


@pytest.fixture
def backend_config(backend_config_overrides) -> OpenEoBackendConfig:
    """
    Fixture to get the default OpenEoBackendConfig and optionally override some fields
    during the lifetime of a test through parameterization of the `backend_config_overrides` fixture.
    """
    if backend_config_overrides is None:
        yield openeo_driver.config.load.get_backend_config()
    else:
        with config_overrides(**backend_config_overrides):
            yield openeo_driver.config.load.get_backend_config()


@pytest.fixture
def dummy_processing() -> DummyProcessing:
    return DummyProcessing()


@pytest.fixture
def backend_implementation(backend_config, dummy_processing) -> DummyBackendImplementation:
    return DummyBackendImplementation(config=backend_config, processing=dummy_processing)


@pytest.fixture
def udp_registry(backend_implementation) -> UserDefinedProcesses:
    return backend_implementation.user_defined_processes


# TODO: move this to dummy config file
TEST_APP_CONFIG = dict(
    TESTING=True,
    SERVER_NAME='oeo.net',
)


@pytest.fixture
def flask_app(backend_implementation) -> flask.Flask:
    app = build_app(
        backend_implementation=backend_implementation,
        # error_handling=False,
    )
    app.config.from_mapping(TEST_APP_CONFIG)
    return app


@pytest.fixture
def client(flask_app):
    return flask_app.test_client()


@pytest.fixture
def urllib_mock() -> UrllibMocker:
    with UrllibMocker().patch() as mocker:
        yield mocker


@contextlib.contextmanager
def enhanced_logging(
    level=logging.INFO,
    json=False,
    format=None,
    request_ids=("123-456", "234-567", "345-678", "456-789", "567-890"),
    context: Optional[str] = None,
    enable_global_extra_logging: bool = False,
):
    """Set up logging with additional injection of request id, user id, ...."""
    root_logger = logging.getLogger()
    orig_root_level = root_logger.level

    out = io.StringIO()
    handler = logging.StreamHandler(out)
    handler.setLevel(level)
    if json:
        formatter = pythonjsonlogger.jsonlogger.JsonFormatter(format)
    else:
        formatter = logging.Formatter(format)
    handler.setFormatter(formatter)
    if context == LOGGING_CONTEXT_FLASK:
        handler.addFilter(FlaskRequestCorrelationIdLogging())
        handler.addFilter(FlaskUserIdLogging())
    if context == LOGGING_CONTEXT_BATCH_JOB or enable_global_extra_logging:
        handler.addFilter(GlobalExtraLoggingFilter())
    root_logger.addHandler(handler)
    root_logger.setLevel(level)
    try:
        with mock.patch.object(FlaskRequestCorrelationIdLogging, "_build_request_id", side_effect=request_ids):
            yield out
    finally:
        root_logger.removeHandler(handler)
        root_logger.setLevel(orig_root_level)


TEST_SWIFT_URL = "https://s3.example.test"


@pytest.fixture
def swift_url(monkeypatch):
    """
    For real environments this is used as the fallback endpoint for doing S3 requests.
    """
    monkeypatch.setenv("SWIFT_URL", TEST_SWIFT_URL)
