"""
Script to start a local server. This script can serve as the entry-point for doing spark-submit.
"""

from openeo_driver import server
from openeo_driver.server import show_log_level
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

_log = logging.getLogger('openeo-dummy-local')

if __name__ == '__main__':
    _log.info(repr({"pid": os.getpid(), "interpreter": sys.executable, "version": sys.version, "argv": sys.argv}))

    from openeo_driver.views import app

    show_log_level(logging.getLogger('openeo'))
    show_log_level(logging.getLogger('openeo_driver'))
    show_log_level(app.logger)

    server.run(
        title="'Local Dummy",
        description="Local openEO API using dummy backend",
        deploy_metadata=None,
        backend_version="local-42",
        threads=4,
        host="127.0.0.1",
        port=8080
    )
