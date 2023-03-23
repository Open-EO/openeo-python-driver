import abc
import inspect
import logging
import zipfile
from pathlib import Path
from typing import List, Union, Optional, Dict, Any, Tuple, Sequence
import io

import geopandas as gpd
import numpy as np
import pyproj
import shapely.geometry
import shapely.geometry.base
import shapely.ops
import xarray
from pyproj import CRS
import requests

from openeo import ImageCollection
from openeo.metadata import CollectionMetadata
from openeo.util import ensure_dir
from openeo_driver.datastructs import SarBackscatterArgs, ResolutionMergeArgs, StacAsset
from openeo_driver.errors import FeatureUnsupportedException, InternalException
from openeo_driver.util.geometry import GeometryBufferer, validate_geojson_coordinates
from openeo_driver.util.ioformats import IOFORMATS
from openeo_driver.util.utm import area_in_square_meters
from openeo_driver.utils import EvalEnv

log = logging.getLogger(__name__)


class DriverDataCube(ImageCollection):
    """Base class for "driver" side raster data cubes."""

    # TODO cut the openeo.ImageCollection chord (https://github.com/Open-EO/openeo-python-client/issues/100)

    def __init__(self, metadata: CollectionMetadata = None):
        self.metadata = CollectionMetadata.get_or_create(metadata)

    def __eq__(self, o: object) -> bool:
        if o.__class__ == self.__class__:
            if o.metadata == self.metadata:
                return True
        return False

    def _not_implemented(self):
        """Helper to raise a NotImplemented exception containing method name"""
        raise NotImplementedError("DataCube method not implemented: {m!r}".format(m=inspect.stack()[1].function))

    def filter_temporal(self, start: str, end: str) -> 'DriverDataCube':
        self._not_implemented()

    def filter_bbox(self, west, south, east, north, crs=None, base=None, height=None) -> 'DriverDataCube':
        self._not_implemented()

    def filter_spatial(self, geometries) -> 'DriverDataCube':
        self._not_implemented()

    def filter_bands(self, bands) -> 'DriverDataCube':
        self._not_implemented()

    def apply(self, process) -> 'DriverDataCube':
        self._not_implemented()

    def apply_kernel(self, kernel: list, factor=1, border=0, replace_invalid=0) -> 'DriverDataCube':
        self._not_implemented()

    def apply_neighborhood(self, process, size: List[dict], overlap: List[dict], env: EvalEnv) -> 'DriverDataCube':
        self._not_implemented()

    def apply_dimension(self, process, dimension: str, target_dimension: str=None, context:dict = None, env: EvalEnv = None) -> 'DriverDataCube':
        self._not_implemented()

    def apply_tiles_spatiotemporal(self, process, context={}) -> 'DriverDataCube':
        self._not_implemented()

    def reduce_dimension(self, reducer, dimension: str, context: Any, env: EvalEnv) -> 'DriverDataCube':
        self._not_implemented()

    def chunk_polygon(self, reducer, chunks, mask_value: float, env: EvalEnv, context={}) -> 'DriverDataCube':
        self._not_implemented()

    def add_dimension(self, name: str, label, type: str = "other") -> 'DriverDataCube':
        self._not_implemented()

    def drop_dimension(self, name: str) -> 'DriverDataCube':
        self._not_implemented()

    def dimension_labels(self, dimension: str) -> 'DriverDataCube':
        self._not_implemented()

    def rename_dimension(self, source: str, target: str) -> 'DriverDataCube':
        self._not_implemented()

    def rename_labels(self, dimension: str, target: list, source: list = None) -> 'DriverDataCube':
        self._not_implemented()

    def reduce_bands(self, process) -> 'DriverDataCube':
        # TODO #47: remove this non-standard process
        self._not_implemented()

    def mask(self, mask: 'DriverDataCube', replacement=None) -> 'DriverDataCube':
        self._not_implemented()

    def mask_polygon(self, mask: shapely.geometry.MultiPolygon, replacement=None, inside: bool = False) -> 'DriverDataCube':
        self._not_implemented()

    def merge_cubes(self, other: 'DriverDataCube', overlap_resolver) -> 'DriverDataCube':
        self._not_implemented()

    def resample_cube_spatial(self, target: 'DriverDataCube', method: str = 'near') -> 'DriverDataCube':
        self._not_implemented()

    def aggregate_temporal(self, intervals: list, reducer, labels: list = None,
                           dimension: str = None, context:dict = None) -> 'DriverDataCube':
        self._not_implemented()

    def aggregate_spatial(
            self,
            geometries: Union[shapely.geometry.base.BaseGeometry, str, "DriverVectorCube"],
            reducer: dict,
            target_dimension: str = "result",
    ) -> Union["AggregatePolygonResult", "AggregatePolygonSpatialResult", "DriverVectorCube"]:
        # TODO: drop `target_dimension`? see https://github.com/Open-EO/openeo-processes/issues/366
        self._not_implemented()


    def timeseries(self, x, y, srs="EPSG:4326") -> dict:
        # TODO #47: remove this non-standard process
        self._not_implemented()

    def ndvi(self, nir: str = "nir", red: str = "red", target_band: str = None) -> 'DriverDataCube':
        self._not_implemented()

    def save_result(self, filename: str, format: str, format_options: dict = None) -> str:
        self._not_implemented()

    def atmospheric_correction(self, method: str = None) -> 'DriverDataCube':
        self._not_implemented()

    def sar_backscatter(self, args: SarBackscatterArgs) -> 'DriverDataCube':
        self._not_implemented()

    def resolution_merge(self, args: ResolutionMergeArgs) -> 'DriverDataCube':
        self._not_implemented()


class VectorCubeError(InternalException):
    code = "VectorCubeError"

    def __init__(self, message="Unspecified VectorCube error"):
        super(VectorCubeError, self).__init__(message=message)


class DriverVectorCube:
    """
    Base class for driver-side 'vector cubes'

    Internal design has two components:
    - a GeoPandas dataframe for holding the GeoJSON-style properties (possibly heterogeneously typed or sparse/free-style)
    - optional xarray DataArray for holding the data cube data (homogeneously typed and rigorously indexed/gridded)
    These components are "joined" on the GeoPandas dataframe's index and DataArray first dimension
    """
    DIM_GEOMETRIES = "geometries"
    FLATTEN_PREFIX = "vc"

    def __init__(
        self,
        geometries: gpd.GeoDataFrame,
        cube: Optional[xarray.DataArray] = None,
        flatten_prefix: str = FLATTEN_PREFIX,
    ):
        """

        :param geometries:
        :param cube:
        :param flatten_prefix: prefix for column/field/property names when flattening the cube
        """
        # TODO #114 EP-3981: lazy loading (like DelayedVector)?
        if cube is not None:
            if cube.dims[0] != self.DIM_GEOMETRIES:
                log.error(f"First cube dim should be {self.DIM_GEOMETRIES!r} but got dims {cube.dims!r}")
                raise VectorCubeError("Cube's first dimension is invalid.")
            if not geometries.index.equals(cube.indexes[cube.dims[0]]):
                log.error(f"Invalid VectorCube components {geometries.index!r} != {cube.indexes[cube.dims[0]]!r}")
                raise VectorCubeError("Incompatible vector cube components")
        self._geometries: gpd.GeoDataFrame = geometries
        self._cube = cube
        self._flatten_prefix = flatten_prefix

    def with_cube(self, cube: xarray.DataArray, flatten_prefix: str = FLATTEN_PREFIX) -> "DriverVectorCube":
        """Create new vector cube with same geometries but new cube"""
        log.info(f"Creating vector cube with new cube {cube.name!r}")
        return type(self)(
            geometries=self._geometries, cube=cube, flatten_prefix=flatten_prefix
        )

    @classmethod
    def from_fiona(
        cls,
        paths: List[Union[str, Path]],
        driver: Optional[str] = None,
        options: Optional[dict] = None,
    ) -> "DriverVectorCube":
        """Factory to load vector cube data using fiona/GeoPandas."""
        if len(paths) != 1:
            # TODO #114 EP-3981: support multiple paths
            raise FeatureUnsupportedException(message="Loading a vector cube from multiple files is not supported")
        # TODO #114 EP-3981: lazy loading like/with DelayedVector
        # note for GeoJSON: will consider Feature.id as well as Feature.properties.id
        if "parquet" == driver:
            return cls.from_parquet(paths=paths)
        else:
            return cls(geometries=gpd.read_file(paths[0], driver=driver))

    @classmethod
    def from_parquet(cls, paths: List[Union[str, Path]]):
        if len(paths) != 1:
            # TODO #114 EP-3981: support multiple paths
            raise FeatureUnsupportedException(
                message="Loading a vector cube from multiple files is not supported"
            )

        location = paths[0]
        if isinstance(location, str) and location.startswith("http"):
            resp = requests.get(location, stream=True)
            resp.raw.decode_content = True
            location = io.BytesIO(resp.raw.read())
        df = gpd.read_parquet(location)
        log.info(f"Read geoparquet from {location} crs {df.crs} length {len(df)}")

        if "OGC:CRS84" in str(df.crs) or "WGS 84 (CRS84)" in str(df.crs):
            # workaround for not being able to decode ogc:crs84
            df.crs = CRS.from_epsg(4326)
        return cls(geometries=df)

    @classmethod
    def from_geojson(cls, geojson: dict) -> "DriverVectorCube":
        """Construct vector cube from GeoJson dict structure"""
        validate_geojson_coordinates(geojson)
        # TODO support more geojson types?
        if geojson["type"] in {"Polygon", "MultiPolygon", "Point", "MultiPoint"}:
            features = [{"type": "Feature", "geometry": geojson, "properties": {}}]
        elif geojson["type"] in {"Feature"}:
            features = [geojson]
        elif geojson["type"] in {"GeometryCollection"}:
            # TODO #71 #114 Deprecate/avoid usage of GeometryCollection
            log.error("Input GeoJSON of deprecated type 'GeometryCollection', please use a FeatureCollection or another type of Multi geometry.")
            features = [{"type": "Feature", "geometry": g, "properties": {}} for g in geojson["geometries"]]
        elif geojson["type"] in {"FeatureCollection"}:
            features = geojson
        else:
            raise FeatureUnsupportedException(
                f"Can not construct DriverVectorCube from {geojson.get('type', type(geojson))!r}"
            )
        return cls(geometries=gpd.GeoDataFrame.from_features(features))

    @classmethod
    def from_geometry(
        cls,
        geometry: Union[
            shapely.geometry.base.BaseGeometry,
            Sequence[shapely.geometry.base.BaseGeometry],
        ],
    ):
        """Construct vector cube from a shapely geometry (list)"""
        if isinstance(geometry, shapely.geometry.base.BaseGeometry):
            geometry = [geometry]
        return cls(geometries=gpd.GeoDataFrame(geometry=geometry))

    def _as_geopandas_df(self) -> gpd.GeoDataFrame:
        """Join geometries and cube as a geopandas dataframe"""
        # TODO: avoid copy?
        df = self._geometries.copy(deep=True)
        if self._cube is not None:
            assert self._cube.dims[0] == self.DIM_GEOMETRIES
            # TODO: better way to combine cube with geometries
            # Flatten multiple (non-geometry) dimensions from cube to new properties in geopandas dataframe
            if self._cube.dims[1:]:
                stacked = self._cube.stack(prop=self._cube.dims[1:])
                log.info(f"Flattened cube component of vector cube to {stacked.shape[1]} properties")
                for p in stacked.indexes["prop"]:
                    name = "~".join(str(x) for x in [self._flatten_prefix] + list(p))
                    # TODO: avoid column collisions?
                    df[name] = stacked.sel(prop=p)
            else:
                df[self._flatten_prefix] = self._cube

        return df

    def to_geojson(self) -> dict:
        """Export as GeoJSON FeatureCollection."""
        return shapely.geometry.mapping(self._as_geopandas_df())

    def to_wkt(self) -> List[str]:
        wkts = [str(g) for g in self._geometries.geometry]
        return wkts

    def get_crs(self) -> pyproj.CRS:
        return self._geometries.crs or pyproj.CRS.from_epsg(4326)

    def write_assets(
            self, directory: Union[str, Path], format: str, options: Optional[dict] = None
    ) -> Dict[str, StacAsset]:
        directory = ensure_dir(directory)
        format_info = IOFORMATS.get(format)
        # TODO: check if format can be used for vector data?
        path = directory / f"vectorcube.{format_info.extension}"

        if format_info.format == "JSON":
            # TODO: eliminate this legacy format?
            log.warning(
                f"Exporting vector cube {self} to legacy, non-standard JSON format"
            )
            return self.to_legacy_save_result().write_assets(directory)

        self._as_geopandas_df().to_file(path, driver=format_info.fiona_driver)

        if not format_info.multi_file:
            # single file format
            return {path.name: {
                "href": path,
                "title": "Vector cube",
                "type": format_info.mimetype,
                "roles": ["data"],
            }}
        else:
            # Multi-file format
            components = list(directory.glob("vectorcube.*"))
            if options.get("zip_multi_file"):
                # TODO: automatically zip shapefile components?
                zip_path = path.with_suffix(f".{format_info.extension}.zip")
                with zipfile.ZipFile(zip_path, "w") as zip_file:
                    for component in components:
                        zip_file.write(component, arcname=component.name)
                return {path.name: {
                    "href": zip_path,
                    "title": "Vector cube",
                    "type": "application/zip",
                    "roles": ["data"],
                }}
            else:
                # TODO: better multi-file support?
                return {p.name: {"href": p} for p in components}

    def to_multipolygon(self) -> shapely.geometry.MultiPolygon:
        return shapely.ops.unary_union(self._geometries.geometry)

    def to_legacy_save_result(self) -> Union["AggregatePolygonResult", "JSONResult"]:
        """
        Export to legacy AggregatePolygonResult/JSONResult objects.
        Provided as temporary adaption layer while migrating to real vector cubes.
        """
        # TODO: eliminate these legacy, non-standard formats?
        from openeo_driver.save_result import AggregatePolygonResult, JSONResult

        cube = self._cube
        # TODO: more flexible temporal/band dimension detection?
        if cube.dims == (self.DIM_GEOMETRIES, "t"):
            # Add single band dimension
            cube = cube.expand_dims({"bands": ["band"]}, axis=-1)
        if cube.dims == (self.DIM_GEOMETRIES, "t", "bands"):
            cube = cube.transpose("t", self.DIM_GEOMETRIES, "bands")
            timeseries = {
                t.item(): t_slice.values.tolist()
                for t, t_slice in zip(cube.coords["t"], cube)
            }
            return AggregatePolygonResult(timeseries=timeseries, regions=self)
        elif cube.dims == (self.DIM_GEOMETRIES, "bands"):
            # This covers the legacy `AggregatePolygonSpatialResult` code path,
            # but as AggregatePolygonSpatialResult's constructor expects a folder of CSV file(s),
            # we keep it simple here with a basic JSONResult result.
            cube = cube.transpose(self.DIM_GEOMETRIES, "bands")
            return JSONResult(data=cube.values.tolist())
        raise ValueError(
            f"Unsupported cube configuration {cube.dims} for _write_legacy_aggregate_polygon_result_json"
        )

    def get_bounding_box(self) -> Tuple[float, float, float, float]:
        # TODO: cache bounding box?
        # TODO #114 #141 Open-EO/openeo-geopyspark-driver#239: option to buffer point geometries (if any)
        #       before calculating bounding box?  Or add minimum width/height constraint for bbox?
        return tuple(self._geometries.total_bounds)

    def get_bounding_box_geometry(self) -> shapely.geometry.Polygon:
        return shapely.geometry.Polygon.from_bounds(*self.get_bounding_box())

    def get_bounding_box_geojson(self) -> dict:
        return shapely.geometry.mapping(self.get_bounding_box_geometry())

    def get_bounding_box_area(self) -> float:
        """Bounding box area in square meters"""
        return area_in_square_meters(
            self.get_bounding_box_geometry(), crs=self.get_crs()
        )

    def get_area(self) -> float:
        """Total geometry area in square meters"""
        return area_in_square_meters(self.to_multipolygon(), self.get_crs())

    def geometry_count(self) -> int:
        """Size of the geometry dimension"""
        return len(self._geometries.index)

    def get_geometries(self) -> Sequence[shapely.geometry.base.BaseGeometry]:
        return self._geometries.geometry

    def get_ids(self) -> Optional[Sequence]:
        return self._geometries.get("id")

    def get_xarray_cube_basics(self) -> Tuple[tuple, dict]:
        """Get initial dims/coords for xarray DataArray construction"""
        dims = (self.DIM_GEOMETRIES,)
        coords = {self.DIM_GEOMETRIES: self._geometries.index.to_list()}
        return dims, coords

    def __eq__(self, other):
        return (isinstance(other, DriverVectorCube)
                and np.array_equal(self._as_geopandas_df().values, other._as_geopandas_df().values))

    def fit_class_random_forest(
        self,
        target: "DriverVectorCube",
        num_trees: int = 100,
        max_variables: Optional[Union[int, str]] = None,
        seed: Optional[int] = None,
    ) -> "DriverMlModel":
        raise NotImplementedError

    def buffer_points(self, distance: float = 10) -> "DriverVectorCube":
        """
        Buffer point geometries

        :param distance: distance in meter
        :return: new DriverVectorCube
        """
        # TODO: also cover MultiPoints?
        # TODO: do we also need buffering of line geometries?
        # TODO: preserve original properties?
        bufferer = GeometryBufferer.from_meter_for_crs(
            distance=distance, crs=self.get_crs()
        )

        return DriverVectorCube.from_geometry(
            [
                bufferer.buffer(g) if isinstance(g, shapely.geometry.Point) else g
                for g in self.get_geometries()
            ]
        )


class DriverMlModel:
    """Base class for driver-side 'ml-model' data structures"""
    METADATA_FILE_NAME = "ml_model_metadata.json"

    def get_model_metadata(self, directory: Union[str, Path]) -> Dict[str, Any]:
        raise NotImplementedError

    def write_assets(self, directory: Union[str, Path]) -> Dict[str, StacAsset]:
        raise NotImplementedError


class SupportsRunUdf(metaclass=abc.ABCMeta):
    """
    Interface for cube/result classes that (partially) support `run_udf`
    """

    @abc.abstractmethod
    def supports_udf(self, udf: str, runtime: str = "Python") -> bool:
        """Check if UDF code is supported."""
        return False

    @abc.abstractmethod
    def run_udf(
        self, udf: str, runtime: str = "Python", context: Optional[dict] = None
    ):
        ...
