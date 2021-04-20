import logging
import os
from typing import Union

import gunicorn.app.base


def show_log_level(logger: logging.Logger):
    """Helper to show threshold log level of a logger."""
    level = logger.getEffectiveLevel()
    logger.log(level, 'Logger {n!r} level: {t}'.format(n=logger.name, t=logging.getLevelName(level)))


def run(title: str, description: str, deploy_metadata: Union[dict, None], backend_version: str, threads: int, host: str, port: int, on_started=lambda: None) -> None:
    """ Starts a web server exposing OpenEO-GeoPySpark, bound to a public IP. """

    from openeo_driver.views import app

    app.logger.setLevel('DEBUG')
    app.config['OPENEO_BACKEND_VERSION'] = backend_version
    app.config['OPENEO_TITLE'] = title
    app.config['OPENEO_DESCRIPTION'] = description
    app.config['OPENEO_BACKEND_DEPLOY_METADATA'] = deploy_metadata
    app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024  # bytes
    app.config['SIGNED_URL'] = os.getenv('SIGNED_URL')
    app.config['SIGNED_URL_SECRET'] = os.getenv('SIGNED_URL_SECRET')
    app.config['SIGNED_URL_EXPIRATION'] = os.getenv('SIGNED_URL_EXPIRATION')

    app.logger.info('App info logging enabled!')
    app.logger.debug('App debug logging enabled!')

    #note the use of 1 worker and multiple threads
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

    print(options)

    def when_ready(server) -> None:
        print(server)

        logging.getLogger('gunicorn.error').info('Gunicorn info logging enabled!')
        logging.getLogger('flask').info('Flask info logging enabled!')

        on_started()

    application = StandaloneApplication(app, when_ready, options)
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
