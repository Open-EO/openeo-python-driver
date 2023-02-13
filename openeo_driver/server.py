import datetime
import logging
import logging.config
from typing import List

import flask
import gunicorn.app.base
import gunicorn.glogging

from openeo.util import rfc3339
from openeo_driver.util.logging import show_log_level
from openeo_driver.utils import get_package_versions

_log = logging.getLogger(__name__)


def build_backend_deploy_metadata(packages: List[str]) -> dict:
    """Build deploy metadata (version listing of provided python packages)"""
    return {
        "date": rfc3339.utcnow(),
        "versions": get_package_versions(packages),
    }


def run_gunicorn(app: flask.Flask, threads: int, host: str, port: int, on_started=lambda: None):
    """Run Flask app as gunicorn application."""

    # TODO move this meta logging out of this function?
    show_log_level(app.logger)

    # note the use of 1 worker and multiple threads
    # we were seeing strange py4j errors when executing multiple requests in parallel
    # this seems to be related by the type and configuration of worker that gunicorn uses, aiohttp also gave very bad results
    options = {
        'bind': '%s:%s' % (host, port),
        'workers': 1,
        'threads': threads,
        'worker_class': 'gthread',
        'timeout': 1000,
        # Override default gunicorn logger class so that gunicorn logging follows our global logging config
        "logger_class": ConformingGunicornLogger,
    }

    _log.info(f"StandaloneApplication options: {options}")

    def when_ready(server) -> None:
        _log.info(f"when_ready: {server}")

        show_log_level("gunicorn.error")
        show_log_level("flask")

        on_started()

    _log.info("Creating StandaloneApplication")
    application = StandaloneApplication(app, when_ready, options)

    _log.info("Running StandaloneApplication")
    application.run()


class StandaloneApplication(gunicorn.app.base.BaseApplication):
    def __init__(self, app, when_ready, options=None):
        self.options = options or {}
        self.application = app
        self.when_ready = when_ready
        super(StandaloneApplication, self).__init__()

    def load_config(self):
        config = dict((k, v) for k, v in self.options.items() if k in self.cfg.settings and v is not None)
        config['when_ready'] = self.when_ready
        for key, value in config.items():
            self.cfg.set(key.lower(), value)

    def load(self):
        return self.application


class ConformingGunicornLogger(gunicorn.glogging.Logger):
    """
    Override the default Gunicorn Logger so that it conforms to the global
    logging config (handlers, formatters, filters, ...).

    Use this class for the "logger_class" option (https://docs.gunicorn.org/en/stable/settings.html#logger-class).
    It assumes the logging is already set up before the gunicorn application
    is started (``BaseApplication.run()``) as described in https://docs.gunicorn.org/en/stable/custom.html).

    Based on, inspired by:
    - https://stackoverflow.com/questions/41087790/how-to-override-gunicorns-logging-config-to-use-a-custom-formatter
    - https://github.com/benoitc/gunicorn/issues/1572
    """

    def __init__(self, cfg):
        if logging.getLogger("gunicorn.access").isEnabledFor(logging.INFO) and not cfg.accesslog:
            # Although the actual accesslog value is not really used,
            # access logging will be skipped if it is empty/None:
            cfg.set("accesslog", "_dummy")
        super().__init__(cfg=cfg)

    def setup(self, cfg):
        self.error_log.propagate = True
        self.access_log.propagate = True
