import fiona
from shapely.geometry import shape, box
from shapely.geometry.base import BaseGeometry
from urllib.parse import urlparse
import requests
from datetime import datetime, timedelta
import os
import json
from typing import Iterable, List, Dict
from functools import reduce


class DelayedVector:
    def __init__(self, path: str):
        self.path = path
        self._downloaded_shapefile = None

    @property
    def geometries(self) -> Iterable[BaseGeometry]:
        if self.path.startswith("http"):
            if DelayedVector._is_shapefile(self.path):
                local_shp_file = self._download_shapefile(self.path)
                geometries = DelayedVector._read_shapefile_geometries(local_shp_file)
            else:  # it's GeoJSON
                geojson = requests.get(self.path).json()
                geometries = DelayedVector._read_geojson_geometries(geojson)
        else:  # it's a file on disk
            if self.path.endswith(".shp"):
                geometries = DelayedVector._read_shapefile_geometries(self.path)
            else:  # it's GeoJSON
                with open(self.path, 'r') as f:
                    geojson = json.load(f)
                    geometries = DelayedVector._read_geojson_geometries(geojson)

        return geometries

    @property
    def bounds(self) -> (float, float, float, float):
        # FIXME: code duplication
        if self.path.startswith("http"):
            if DelayedVector._is_shapefile(self.path):
                local_shp_file = self._download_shapefile(self.path)
                bounds = DelayedVector._read_shapefile_bounds(local_shp_file)
            else:  # it's GeoJSON
                geojson = requests.get(self.path).json()
                # FIXME: can be cached
                bounds = DelayedVector._read_geojson_bounds(geojson)
        else:  # it's a file on disk
            if self.path.endswith(".shp"):
                bounds = DelayedVector._read_shapefile_bounds(self.path)
            else:  # it's GeoJSON
                with open(self.path, 'r') as f:
                    geojson = json.load(f)
                    bounds = DelayedVector._read_geojson_bounds(geojson)

        return bounds

    @staticmethod
    def _is_shapefile(path: str) -> bool:
        return DelayedVector._filename(path).endswith(".shp")

    @staticmethod
    def _filename(path: str) -> str:
        return urlparse(path).path.split("/")[-1]

    def _download_shapefile(self, shp_url: str) -> str:
        if self._downloaded_shapefile:
            return self._downloaded_shapefile

        def expiring_download_directory():
            now = datetime.now()
            now_hourly_truncated = now - timedelta(minutes=now.minute, seconds=now.second, microseconds=now.microsecond)
            hourly_id = hash(shp_url + str(now_hourly_truncated))
            return "/data/projects/OpenEO/download_%s" % hourly_id

        def save_as(src_url: str, dest_path: str):
            with open(dest_path, 'wb') as f:
                f.write(requests.get(src_url).content)

        download_directory = expiring_download_directory()
        shp_file = download_directory + "/" + DelayedVector._filename(shp_url)

        try:
            os.mkdir(download_directory)

            shx_file = shp_file.replace(".shp", ".shx")
            dbf_file = shp_file.replace(".shp", ".dbf")

            shx_url = shp_url.replace(".shp", ".shx")
            dbf_url = shp_url.replace(".shp", ".dbf")

            save_as(shp_url, shp_file)
            save_as(shx_url, shx_file)
            save_as(dbf_url, dbf_file)
        except FileExistsError:
            pass

        self._downloaded_shapefile = shp_file
        return self._downloaded_shapefile

    @staticmethod
    def _read_shapefile_geometries(shp_path: str) -> List[BaseGeometry]:
        # FIXME: returned as a list for safety but possible to return as an iterable?
        with fiona.open(shp_path) as collection:
            return [shape(record['geometry']) for record in collection]

    @staticmethod
    def _read_shapefile_bounds(shp_path: str) -> List[BaseGeometry]:
        with fiona.open(shp_path) as collection:
            return collection.bounds

    @staticmethod
    def _as_geometry_collection(feature_collection: Dict) -> Dict:
        geometries = (feature['geometry'] for feature in feature_collection['features'])

        return {
            'type': 'GeometryCollection',
            'geometries': geometries
        }

    @staticmethod
    def _read_geojson_geometries(geojson: Dict) -> Iterable[BaseGeometry]:
        if geojson['type'] == 'FeatureCollection':
            geojson = DelayedVector._as_geometry_collection(geojson)

        if geojson['type'] == 'GeometryCollection':
            geometries = (shape(geometry) for geometry in geojson['geometries'])
        else:
            geometry = shape(geojson)
            geometries = [geometry]

        return geometries

    @staticmethod
    def _read_geojson_bounds(geojson: Dict) -> (float, float, float, float):
        if geojson['type'] == 'FeatureCollection':
            geojson = DelayedVector._as_geometry_collection(geojson)

        if geojson['type'] == 'GeometryCollection':
            geometries = (shape(geometry) for geometry in geojson['geometries'])
            individual_bboxes = (box(*geometry.bounds) for geometry in geometries)
            bounds = reduce(lambda combined, individual: combined.union(individual), individual_bboxes).bounds
        else:
            geometry = shape(geojson)
            bounds = geometry.bounds

        return bounds
