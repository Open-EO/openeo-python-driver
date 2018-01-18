from flask import Flask
app = Flask(__name__)
app.config['APPLICATION_ROOT'] = '/openeo'
import openeo_driver.views