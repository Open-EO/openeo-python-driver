## OpenEO Geopyspark Driver

[![Status](https://img.shields.io/badge/Status-proof--of--concept-yellow.svg)]()

Python version: 3.5

This driver implements the GeoPySpark/Geotrellis specific backend for OpenEO.

It does this by implementing a direct (non-REST) version of the OpenEO client API on top 
of [GeoPySpark](https://github.com/locationtech-labs/geopyspark/). 

A REST service based on Flask translates incoming calls to this local API.

### Running locally

Note: make sure that the git submodules are checked out and updated properly (e.g. run `git submodule update --init`).

For development, you can run the service using Flask:

    export FLASK_APP=openeo_driver.views
    export FLASK_DEBUG=1 
    flask run

For production, a gunicorn server script is available:

    python openeo_driver/server.py
