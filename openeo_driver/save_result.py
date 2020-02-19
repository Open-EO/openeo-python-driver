import os
import warnings
from abc import ABC
from pathlib import Path
from tempfile import mkstemp
from zipfile import ZipFile
from typing import Union

import numpy as np
from flask import send_from_directory, jsonify
from shapely.geometry import GeometryCollection, mapping

from openeo_driver.utils import replace_nan_values


class SaveResult(ABC):
    """
    A class that generates a Flask response.
    """

    def __init__(self, format: str = None, options: dict = None):
        self.format = format and format.lower()
        self.options = options or {}

    def set_format(self, format: str, options: dict = None):
        self.format = format.lower()
        self.options = options or {}

    def create_flask_response(self):
        """
        Returns a Flask compatible response.

        :return: A response that can be handled by Flask
        """
        pass


class ImageCollectionResult(SaveResult):

    def __init__(self, imagecollection, format: str, options: dict):
        super().__init__(format=format, options=options)
        self.imagecollection = imagecollection

    def create_flask_response(self):
        filename = self.imagecollection.download(None, bbox="", time="", format=self.format, **self.options)
        return send_from_directory(os.path.dirname(filename), os.path.basename(filename))


class JSONResult(SaveResult):

    def __init__(self, data: dict, format: str = "json", options: dict = None):
        super().__init__(format=format, options=options)
        self.data = data

    def get_data(self):
        return self.data

    def prepare_for_json(self):
        return replace_nan_values(self.get_data())

    def create_flask_response(self):
        return jsonify(self.prepare_for_json())


class AggregatePolygonResult(JSONResult):
    """
    Container for timeseries result of `aggregate_polygon` process (aka "zonal stats")

    Expects internal representation of timeseries as nested structure:

        dict mapping timestamp (str) to:
            a list, one item per polygon:
                a list, one float per band

    """

    def __init__(self, timeseries: dict, regions: GeometryCollection):
        super().__init__(data=timeseries)
        self._regions = regions

    def get_data(self):
        if self.format in ('covjson', 'coveragejson'):
            return self.to_covjson()
        # By default, keep original (proprietary) result format
        return self.data

    def to_covjson(self) -> dict:
        """
        Convert internal timeseries structure to Coverage JSON structured dict
        """

        # Convert GeometryCollection to list of GeoJSON Polygon coordinate arrays
        polygons = [p["coordinates"] for p in mapping(self._regions)["geometries"]]

        # TODO make sure timestamps are ISO8601 (https://covjson.org/spec/#temporal-reference-systems)
        timestamps = sorted(self.data.keys())

        # Count bands in timestamp data
        # TODO get band count and names from metadata
        band_counts = set(len(polygon_data) for ts_data in self.data.values() for polygon_data in ts_data)
        band_counts.discard(0)
        if len(band_counts) != 1:
            raise ValueError("Multiple band counts in data: {c}".format(c=band_counts))
        band_count = band_counts.pop()

        # build parameter value array (one for each band)
        actual_timestamps = []
        param_values = [[] for _ in range(band_count)]
        for ts in timestamps:
            ts_data = self.data[ts]
            if len(ts_data) != len(polygons):
                warnings.warn("Expected {e} polygon results, but got {g}".format(e=len(polygons), g=len(ts_data)))
                continue
            if all(len(polygon_data) != band_count for polygon_data in ts_data):
                # Skip timestamps without any complete data
                # TODO: also skip timestamps with only NaNs?
                continue
            actual_timestamps.append(ts)
            for polygon_data in ts_data:
                if len(polygon_data) == band_count:
                    for b, v in enumerate(polygon_data):
                        param_values[b].append(v)
                else:
                    for b in range(band_count):
                        param_values[b].append(np.nan)

        domain = {
            "type": "Domain",
            "domainType": "MultiPolygonSeries",
            "axes": {
                "t": {"values": actual_timestamps},
                "composite": {
                    "dataType": "polygon",
                    "coordinates": ["x", "y"],
                    "values": polygons,
                },
            },
            "referencing": [
                {
                    "coordinates": ["x", "y"],
                    "system": {
                        "type": "GeographicCRS",
                        "id": "http://www.opengis.net/def/crs/OGC/1.3/CRS84",  # TODO check CRS
                    }
                },
                {
                    "coordinates": ["t"],
                    "system": {"type": "TemporalRS", "calendar": "Gregorian"}
                }
            ]
        }
        parameters = {
            "band{b}".format(b=band): {
                "type": "Parameter",
                "observedProperty": {"label": {"en": "Band {b}".format(b=band)}},
                # TODO: also add unit properties?
            }
            for band in range(band_count)
        }
        shape = (len(actual_timestamps), len(polygons))
        ranges = {
            "band{b}".format(b=band): {
                "type": "NdArray",
                "dataType": "float",
                "axisNames": ["t", "composite"],
                "shape": shape,
                "values": param_values[band],
            }
            for band in range(band_count)
        }
        return {
            "type": "Coverage",
            "domain": domain,
            "parameters": parameters,
            "ranges": ranges
        }


class MultipleFilesResult(SaveResult):
    def __init__(self, format: str, *files: Path):
        super().__init__(format=format)
        self.files = list(files)

    def reduce(self, output_file: Union[str, Path], delete_originals: bool):
        with ZipFile(output_file, "w") as zip_file:
            for file in self.files:
                zip_file.write(filename=file, arcname=file.name)

        if delete_originals:
            for file in self.files:
                file.unlink()

    def create_flask_response(self):
        from flask import current_app

        _, temp_file_name = mkstemp(suffix=".zip", dir=Path.cwd())
        temp_file = Path(temp_file_name)

        def stream_and_remove():
            self.reduce(temp_file, delete_originals=True)

            with open(temp_file, "rb") as f:
                yield from f

            temp_file.unlink()

        resp = current_app.response_class(stream_and_remove(), mimetype="application/zip")
        resp.headers.set('Content-Disposition', 'attachment', filename=temp_file.name)

        return resp
