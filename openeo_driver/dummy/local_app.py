"""
Script to start a local server. This script can serve as the entry-point for doing spark-submit.
"""

from datetime import datetime
import logging
from logging.config import dictConfig

dictConfig({
    'version': 1,
    'formatters': {'default': {
        'format': '[%(asctime)s] %(process)s %(levelname)s in %(name)s: %(message)s',
    }},
    'handlers': {'wsgi': {
        'class': 'logging.StreamHandler',
        'stream': 'ext://flask.logging.wsgi_errors_stream',
        'formatter': 'default'
    }},
    'root': {
        'level': 'INFO',
        'handlers': ['wsgi']
    },
    'loggers': {
        'werkzeug': {'level': 'DEBUG'},
        'flask': {'level': 'DEBUG'},
        'openeo': {'level': 'DEBUG'},
        'openeo_driver': {'level': 'DEBUG'},
        'kazoo': {'level': 'WARN'},
    }
})

import os
import sys

import gunicorn.app.base
from gunicorn.six import iteritems

_log = logging.getLogger('openeo-dummy-local')


def show_log_level(logger: logging.Logger):
    """Helper to show threshold log level of a logger."""
    level = logger.getEffectiveLevel()
    logger.log(level, 'Logger {n!r} level: {t}'.format(n=logger.name, t=logging.getLevelName(level)))


class StandaloneApplication(gunicorn.app.base.BaseApplication):

    def __init__(self, app, options=None):
        self.options = options or {}
        self.application = app
        super(StandaloneApplication, self).__init__()

    def load_config(self):
        config = dict([(key, value) for key, value in iteritems(self.options)
                       if key in self.cfg.settings and value is not None])
        for key, value in iteritems(config):
            self.cfg.set(key.lower(), value)

    def load(self):
        return self.application


if __name__ == '__main__':
    _log.info(repr({"pid": os.getpid(), "interpreter": sys.executable, "version": sys.version, "argv": sys.argv}))

    options = {
        'bind': '%s:%s' % ("127.0.0.1", 8080),
        'workers': 1,
        'threads': 4,
        'worker_class': 'gthread',  # 'gaiohttp',
        'timeout': 1000,
        'loglevel': 'DEBUG',
        'accesslog': '-',
        'errorlog': '-'  # ,
        # 'certfile': 'test.pem',
        # 'keyfile': 'test.key'
    }

    from openeo_driver.views import app
    from flask_cors import CORS

    CORS(app)

    show_log_level(logging.getLogger('openeo'))
    show_log_level(logging.getLogger('openeo_driver'))
    show_log_level(app.logger)

    app.config['OPENEO_BACKEND_VERSION'] = "local-42"
    app.config['OPENEO_TITLE'] = 'Local Dummy'
    app.config['OPENEO_DESCRIPTION'] = 'Local openEO API using dummy backend'
    application = StandaloneApplication(app, options)

    application.run()
