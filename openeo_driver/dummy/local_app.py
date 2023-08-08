"""
Script to start a local server. This script can serve as the entry-point for doing spark-submit.
"""

import logging
import os
import sys

from openeo_driver.dummy.dummy_backend import DummyBackendImplementation
from openeo_driver.server import run_gunicorn
from openeo_driver.util.logging import get_logging_config, setup_logging, show_log_level
from openeo_driver.views import build_app

_log = logging.getLogger('openeo-dummy-local')


def create_app():
    # "create_app" factory for Flask Application discovery
    # see https://flask.palletsprojects.com/en/2.1.x/cli/#application-discovery
    app = build_app(backend_implementation=DummyBackendImplementation())
    return app


if __name__ == '__main__':
    setup_logging(get_logging_config(
        # root_handlers=["stderr_json"],
        loggers={
            "openeo": {"level": "DEBUG"},
            "openeo_driver": {"level": "DEBUG"},
            "flask": {"level": "DEBUG"},
            "werkzeug": {"level": "DEBUG"},
            "kazoo": {"level": "WARN"},
            "gunicorn": {"level": "INFO"},
        },
    ))
    _log.info(repr({"pid": os.getpid(), "interpreter": sys.executable, "version": sys.version, "argv": sys.argv}))

    app = create_app()
    show_log_level(app.logger)

    run_gunicorn(
        app=app,
        threads=4,
        host="127.0.0.1",
        port=8080
    )
