import flask
import pytest

from openeo_driver.backend import OpenEoBackendImplementation, UserDefinedProcesses
from openeo_driver.dummy.dummy_backend import DummyBackendImplementation
from openeo_driver.views import build_app


@pytest.fixture(scope="module")
def backend_implementation() -> OpenEoBackendImplementation:
    return DummyBackendImplementation()


@pytest.fixture
def udp_registry(backend_implementation) -> UserDefinedProcesses:
    return backend_implementation.user_defined_processes


@pytest.fixture(scope="module")
def flask_app(backend_implementation) -> flask.Flask:
    app = build_app(
        backend_implementation=backend_implementation,
        # error_handling=False
    )
    app.config.from_mapping(
        OPENEO_TITLE="openEO Unit Test Dummy Backend",
        TESTING=True,
        SERVER_NAME='oeo.net'
    )
    return app


@pytest.fixture
def client(flask_app):
    return flask_app.test_client()
