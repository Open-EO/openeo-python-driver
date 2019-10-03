## openEO Python Driver

[![Status](https://img.shields.io/badge/Status-proof--of--concept-yellow.svg)]() [![Build Status](https://travis-ci.org/Open-EO/openeo-python-driver.svg?branch=master)](https://travis-ci.org/Open-EO/openeo-python-driver)


Python version: 3.6 or higher

This Python package provides a Flask based REST frontend for openEO backend drivers.
It implements the general REST request handling of the openEO API and dispatches the real work to a pluggable openEO backend driver (such as the [openEO GeoPySpark driver](https://github.com/Open-EO/openeo-geopyspark-driver)).


### Running locally

Note: make sure that the git submodules are checked out and updated properly (e.g. run `git submodule update --init`).

For development, you can run the service using Flask:

    export FLASK_APP=openeo_driver.views
    export FLASK_DEBUG=1 
    flask run

For production, a gunicorn server script is available:

    python openeo_driver/server.py
