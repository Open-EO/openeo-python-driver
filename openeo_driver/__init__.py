from flask import Flask
app = Flask(__name__)
app.config['APPLICATION_ROOT'] = '/openeo'

from ._version import __version__
import openeo_driver.views