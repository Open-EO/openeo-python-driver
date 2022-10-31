# openEO Python Driver

[![Status](https://img.shields.io/badge/Status-proof--of--concept-yellow.svg)]()


Python version: 3.8 or higher

This Python package provides a Flask based REST frontend for openEO backend drivers.
It implements the general REST request handling of the openEO API and dispatches the real work to a pluggable openEO backend driver (such as the [openEO GeoPySpark driver](https://github.com/Open-EO/openeo-geopyspark-driver)).

## Installation

- Clone this repo
- Check out the git submodules

        git submodule update --init
        
- Set up a virtual environment and install `openeo_driver` and it's dependencies.
    - basic install:
    
            pip install . --extra-index-url https://artifactory.vgt.vito.be/api/pypi/python-openeo/simple
    
    - If you plan to do development/testing, install it in editable mode
      and include additional dependencies:
        
            pip install -e .[dev] --extra-index-url https://artifactory.vgt.vito.be/api/pypi/python-openeo/simple

## Running local dummy service

For development, you can run a dummy service using Flask:

    export FLASK_APP=openeo_driver.dummy.local_app
    export FLASK_DEBUG=1 
    flask run

Now, visit http://127.0.0.1:5000/openeo/1.1.0/
