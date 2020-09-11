import tempfile

import fiona
import geopandas as gpd
import pyproj
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from urllib.parse import urlparse
import requests
from datetime import datetime, timedelta
import os
import json
from typing import Iterable, List, Dict


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

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, self.path)

    def __str__(self):
        return self.path
    
    @property
    def crs(self) -> pyproj.CRS:
        if self._crs is None:
            if self.path.startswith("http"):
                if DelayedVector._is_shapefile(self.path):
                    local_shp_file = self._download_shapefile(self.path)
                    self._crs = DelayedVector._read_shapefile_crs(local_shp_file)
                else:  # it's GeoJSON
                    geojson = requests.get(self.path).json()
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
            return [shape(record['geometry']) for record in collection]

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
        crs = geojson.get('crs',{}).get("properties",{}).get("name",None)
        if crs==None:
            return pyproj.CRS({'init': 'epsg:4326'})
        else:
            return pyproj.CRS({'init': crs})
