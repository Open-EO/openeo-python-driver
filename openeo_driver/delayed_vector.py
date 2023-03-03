import json
import logging
import os
import tempfile
from datetime import datetime, timedelta
from typing import Iterable, List, Dict
from urllib.parse import urlparse

import fiona
import geopandas as gpd
import pyproj
import requests
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

from openeo_driver.errors import OpenEOApiException
from openeo_driver.util.geometry import (
    reproject_bounding_box,
    validate_geojson_coordinates,
    reproject_geometry,
)

_log = logging.getLogger(__name__)


class DelayedVector:
    """
        Represents the result of a read_vector process.

        A DelayedVector essentially wraps a reference to a vector file (a path); it's delayed in that it does not load
        geometries into memory until needed to avoid MemoryErrors.

        DelayedVector.path contains the path.
        DelayedVector.geometries loads the vector file into memory so don't do that if it contains a lot of geometries
        (use path instead); DelayedVector.bounds should be safe to use.
    """

    def __init__(self, path: str):
        # TODO: support pathlib too?
        self.path = path
        self._downloaded_shapefile = None
        self._crs = None
        self._area = None

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, self.path)

    def __str__(self):
        return self.path

    def __eq__(self, other):
        return isinstance(other, type(self)) and self.path == other.path

    def _load_geojson_url(self, url: str) -> dict:
        _log.info(f"Loading GeoJSON from {url!r}")
        resp = requests.get(url)
        content_type = resp.headers.get("content-type")
        content_length = resp.headers.get("content-length")
        _log.info(
            f"GeoJSON response: status:{resp.status_code!r}"
            f" content-type:{content_type!r} content-length:{content_length!r}"
        )
        resp.raise_for_status()
        try:
            geojson = resp.json()
            validate_geojson_coordinates(geojson)
            return geojson
        except json.JSONDecodeError as e:
            message = f"Failed to parse GeoJSON from URL {url!r} (content-type={content_type!r}, content-length={content_length!r}): {e!r}"
            # TODO: use generic client error? https://github.com/Open-EO/openeo-api/issues/456
            raise OpenEOApiException(status_code=400, message=message)

    @property
    def crs(self) -> pyproj.CRS:
        if self._crs is None:
            if self.path.startswith("http"):
                if DelayedVector._is_shapefile(self.path):
                    local_shp_file = self._download_shapefile(self.path)
                    self._crs = DelayedVector._read_shapefile_crs(local_shp_file)
                else:  # it's GeoJSON
                    geojson = self._load_geojson_url(url=self.path)
                    # FIXME: can be cached
                    self._crs = DelayedVector._read_geojson_crs(geojson)
            else:  # it's a file on disk
                if self.path.endswith(".shp"):
                    self._crs = DelayedVector._read_shapefile_crs(self.path)
                else:  # it's GeoJSON
                    with open(self.path, 'r') as f:
                        geojson = json.load(f)
                        self._crs = DelayedVector._read_geojson_crs(geojson)
        return self._crs

    @property
    def geometries(self) -> Iterable[BaseGeometry]:
        if self.path.startswith("http"):
            if DelayedVector._is_shapefile(self.path):
                local_shp_file = self._download_shapefile(self.path)
                geometries = DelayedVector._read_shapefile_geometries(local_shp_file)
            else:  # it's GeoJSON
                geojson = self._load_geojson_url(url=self.path)
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
    def geometries_wgs84(self) -> Iterable[BaseGeometry]:
        """
        Returns the geometries in WGS84.
        """
        geometries = self.geometries
        wgs84 = pyproj.CRS("EPSG:4326")
        if self.crs != wgs84:
            geometries = [
                reproject_geometry(geometry, from_crs=self.crs, to_crs=wgs84)
                for geometry in geometries
            ]
        return geometries

    @property
    def area(self):
        if(self._area == None):
            df = self.as_geodataframe()
            latlonbounds = reproject_bounding_box(dict(zip(["west", "south", "east", "north"], self.bounds)),self.crs,"EPSG:4326")

            equal_area_crs = pyproj.Proj(
                proj='aea',
                lat_1=latlonbounds['south'],
                lat_2=latlonbounds['north'])
            transformed_geometry = df.geometry.to_crs(equal_area_crs.crs)
            self._area = transformed_geometry.area.sum()
        return self._area

    def as_geodataframe(self):
        """
        Loads the vector collection and returns a geopandas GeoDataFrame.
        @return:
        """
        return gpd.GeoDataFrame(geometry=list(self.geometries),crs=self.crs)

    @property
    def bounds(self) -> (float, float, float, float):
        # FIXME: code duplication
        if self.path.startswith("http"):
            if DelayedVector._is_shapefile(self.path):
                local_shp_file = self._download_shapefile(self.path)
                bounds = DelayedVector._read_shapefile_bounds(local_shp_file)
            else:  # it's GeoJSON
                geojson = self._load_geojson_url(url=self.path)
                # FIXME: can be cached
                bounds = DelayedVector._read_geojson_bounds(geojson)
        else:  # it's a file on disk
            if self.path.endswith(".shp"):
                bounds = DelayedVector._read_shapefile_bounds(self.path)
            else:  # it's GeoJSON
                with open(self.path, 'r') as f:
                    geojson = json.load(f)
                    validate_geojson_coordinates(geojson)
                    bounds = DelayedVector._read_geojson_bounds(geojson)

        return bounds

    @staticmethod
    def from_json_dict(geojson:dict):
        with tempfile.NamedTemporaryFile(suffix=".json.tmp", delete=False,mode='w') as temp_file:
            json.dump(geojson,temp_file.file)
            return DelayedVector(temp_file.name)

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
            prj_file = shp_file.replace(".shp", ".prj")

            shx_url = shp_url.replace(".shp", ".shx")
            dbf_url = shp_url.replace(".shp", ".dbf")
            prj_url = shp_url.replace(".shp", ".prj")

            save_as(shp_url, shp_file)
            save_as(shx_url, shx_file)
            save_as(dbf_url, dbf_file)
            save_as(prj_url, prj_file)
        except FileExistsError:
            pass

        self._downloaded_shapefile = shp_file
        return self._downloaded_shapefile

    @staticmethod
    def _read_shapefile_geometries(shp_path: str) -> List[BaseGeometry]:
        # FIXME: returned as a list for safety but possible to return as an iterable?
        with fiona.open(shp_path) as collection:
            [validate_geojson_coordinates(record["geometry"]) for record in collection]
            return [shape(record["geometry"]) for record in collection]

    @staticmethod
    def _read_shapefile_bounds(shp_path: str) -> List[BaseGeometry]:
        with fiona.open(shp_path) as collection:
            return collection.bounds

    @staticmethod
    def _read_shapefile_crs(shp_path: str) -> pyproj.CRS:
        """

        @param shp_path:
        @return: CRS as a proj4 dict
        """
        with fiona.open(shp_path) as collection:
            return collection.crs

    @staticmethod
    def _as_geometry_collection(feature_collection: Dict) -> Dict:
        # TODO #71 #114 Deprecate/avoid usage of GeometryCollection
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
            bounds = gpd.GeoSeries(shape(f["geometry"]) for f in geojson["features"]).total_bounds
        elif geojson['type'] == 'GeometryCollection':
            bounds = gpd.GeoSeries(shape(g) for g in geojson['geometries']).total_bounds
        else:
            geometry = shape(geojson)
            bounds = geometry.bounds

        return tuple(bounds)

    @staticmethod
    def _read_geojson_crs(geojson: Dict) -> pyproj.CRS:
        #so actually geojson has no crs, it's always lat lon, need to check what gdal does...
        crs = geojson.get('crs', {}).get("properties", {}).get("name")

        # TODO: what's the deal with this deprecated "init"?
        if crs == None:
            return pyproj.CRS({'init': 'epsg:4326'})
        elif crs.startswith("urn:ogc:"):
            return pyproj.CRS(crs)
        else:
            return pyproj.CRS({'init': crs})
