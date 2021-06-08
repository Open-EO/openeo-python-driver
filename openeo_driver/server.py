import datetime
import logging
import logging.config
import time
from typing import List, Dict

import flask
import gunicorn.app.base
import pkg_resources

from openeo.util import rfc3339

_log = logging.getLogger(__name__)


class UtcLogFormatter(logging.Formatter):
    """Log formatter that uses UTC instead of local time."""
    # based on https://docs.python.org/3/howto/logging-cookbook.html#formatting-times-using-utc-gmt-via-configuration
    converter = time.gmtime


def setup_logging(root_level="INFO", loggers: Dict[str, dict] = None, show_loggers: List[str] = None):
    """
    Set up logging for flask app
    """
    # Based on https://flask.palletsprojects.com/en/2.0.x/logging/
    config = {
        'version': 1,
        'formatters': {
            'utclogformatter': {
                '()': UtcLogFormatter,
                'format': '[%(asctime)s] %(process)s %(levelname)s in %(name)s: %(message)s',
            }
        },
        'handlers': {
            'wsgi': {
                'class': 'logging.StreamHandler',
                'stream': 'ext://flask.logging.wsgi_errors_stream',
                'formatter': 'utclogformatter'
            }
        },
        'root': {
            'level': root_level,
            'handlers': ['wsgi']
        },
    }

    # Merge log levels per logger with some defaults
    loggers_defaults = {
        'openeo': {'level': 'INFO'},
        'openeo_driver': {'level': 'DEBUG'},
        'werkzeug': {'level': 'INFO'},
        'kazoo': {'level': 'WARN'},
    }
    config["loggers"] = {**loggers_defaults, **(loggers or {})}

    logging.config.dictConfig(config)

    for name in {"openeo_driver"}.union(show_loggers):
        show_log_level(logging.getLogger(name))


def show_log_level(logger: logging.Logger):
    """Helper to show threshold log level of a logger."""
    level = logger.getEffectiveLevel()
    logger.log(level, 'Logger {n!r}: effective level {t}'.format(n=logger.name, t=logging.getLevelName(level)))


def build_backend_deploy_metadata(packages: List[str]) -> dict:
    version_info = {}
    for package in packages:
        try:
            version_info[package] = str(pkg_resources.get_distribution(package))
        except pkg_resources.DistributionNotFound:
            version_info[package] = "n/a"
    return {
        'date': rfc3339.normalize(datetime.datetime.utcnow()),
        'versions': version_info
    }


def run_gunicorn(app: flask.Flask, threads: int, host: str, port: int, on_started=lambda: None):
    """Run Flask app as gunicorn application."""

    # TODO move this meta logging out of this function?
    app.logger.setLevel('DEBUG')
    app.logger.info('App info logging enabled!')
    app.logger.debug('App debug logging enabled!')

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

        logging.getLogger('gunicorn.error').info('Gunicorn info logging enabled!')
        logging.getLogger('flask').info('Flask info logging enabled!')

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
