import datetime
import logging
import logging.config
from typing import List

import flask
import gunicorn.app.base

from openeo.util import rfc3339
from openeo_driver.util.logging import show_log_level
from openeo_driver.utils import get_package_versions

_log = logging.getLogger(__name__)


def build_backend_deploy_metadata(packages: List[str]) -> dict:
    return {
        'date': rfc3339.normalize(datetime.datetime.utcnow()),
        'versions': get_package_versions(packages)
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
        'loglevel': 'DEBUG',
        'accesslog': '-',
        'errorlog': '-'
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
