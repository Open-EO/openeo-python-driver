#!/usr/bin/env bash
SPARK_HOME=$(find_spark_home.py) FLASK_APP=openeo_driver/__init__.py FLASK_DEBUG=1 flask run