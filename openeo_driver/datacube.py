from __future__ import annotations
import abc
import inspect
import io
import logging
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import geopandas as gpd
import numpy
import openeo.udf
import pandas
import pyproj
import requests
import shapely.geometry
import shapely.geometry.base
import shapely.ops
import xarray
from openeo.metadata import CollectionMetadata
from openeo.util import ensure_dir, str_truncate
from pyproj import CRS

from openeo_driver.datastructs import ResolutionMergeArgs, SarBackscatterArgs, StacAsset
from openeo_driver.errors import FeatureUnsupportedException, InternalException, ProcessGraphInvalidException
from openeo_driver.util.geometry import GeometryBufferer, validate_geojson_coordinates
from openeo_driver.util.ioformats import IOFORMATS
from openeo_driver.util.pgparsing import SingleRunUDFProcessGraph
from openeo_driver.util.utm import area_in_square_meters
from openeo_driver.utils import EvalEnv

log = logging.getLogger(__name__)


class SupportsRunUdf(metaclass=abc.ABCMeta):
    """
    Interface/Mixin for cube/result classes that (partially) support `run_udf`
    """

    # TODO: as there is quite some duplication between the current methods of this API:
    #       simplify it by just providing a single method: e.g. `get_udf_runner`,
    #       which returns None if run_udf is not supported, and returns a callable (to run the udf on the data) when it is supported.

    @abc.abstractmethod
    def supports_udf(self, udf: str, *, runtime: str = "Python") -> bool:
        """Check if UDF code is supported."""
        return False

    @abc.abstractmethod
    def run_udf(self, udf: str, *, runtime: str = "Python", context: Optional[dict] = None, env: EvalEnv):
        ...


class DriverDataCube:
    """Base class for "driver" side raster data cubes."""

    def __init__(self, metadata: CollectionMetadata = None):
        self.metadata = (
            metadata if isinstance(metadata, CollectionMetadata) else CollectionMetadata(metadata=metadata or {})
        )

    def __eq__(self, o: object) -> bool:
        if o.__class__ == self.__class__:
            if o.metadata == self.metadata:
                return True
        return False

    def get_dimension_names(self) -> List[str]:
        return self.metadata.dimension_names()

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

    def filter_labels(self, condition: dict,dimensin: str, context: Optional[dict] = None, env: EvalEnv = None ) -> 'DriverDataCube':
        self._not_implemented()

    def apply(self, process: dict, *, context: Optional[dict] = None, env: EvalEnv) -> "DriverDataCube":
        self._not_implemented()

    def apply_kernel(self, kernel: list, factor=1, border=0, replace_invalid=0) -> 'DriverDataCube':
        self._not_implemented()

    def apply_neighborhood(
        self, process: dict, *, size: List[dict], overlap: List[dict], context: Optional[dict] = None, env: EvalEnv
    ) -> "DriverDataCube":
        self._not_implemented()

    def apply_dimension(
        self,
        process: dict,
        *,
        dimension: str,
        target_dimension: Optional[str],
        context: Optional[dict] = None,
        env: EvalEnv,
    ) -> "DriverDataCube":
        self._not_implemented()

    def apply_tiles_spatiotemporal(self, process, *, context: Optional[dict] = None) -> "DriverDataCube":
        self._not_implemented()

    def reduce_dimension(
        self, reducer: dict, *, dimension: str, context: Optional[dict] = None, env: EvalEnv
    ) -> "DriverDataCube":
        self._not_implemented()

    def chunk_polygon(
        self,
        *,
        reducer: dict,
        chunks: Union[shapely.geometry.base.BaseGeometry],
        mask_value: Union[float, None],
        env: EvalEnv,
        context: Optional[dict] = None,
    ) -> "DriverDataCube":
        # TODO #229 drop this deprecated API once unused (replaced by `apply_polygon`) (https://github.com/Open-EO/openeo-processes/pull/298)
        self._not_implemented()

    def apply_polygon(
        self,
        *,
        # TODO #229 better type for `polygons` arg: should be vector cube or feature collection like construct
        polygons: shapely.geometry.base.BaseGeometry,
        process: dict,
        mask_value: Optional[float] = None,
        context: Optional[dict] = None,
        env: EvalEnv,
    ) -> DriverDataCube:
        # TODO #229 remove this temporary adapter to deprecated `chunk_polygon` method.
        return self.chunk_polygon(reducer=process, chunks=polygons, mask_value=mask_value, env=env, context=context)
        # self._not_implemented()

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

    def aggregate_temporal(
        self, intervals: list, reducer, labels: list = None, dimension: str = None, context: Optional[dict] = None
    ) -> "DriverDataCube":
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

    def atmospheric_correction(
        self,
        method: Optional[str] = None,
        elevation_model: Optional[str] = None,
        options: Optional[dict] = None,
    ) -> "DriverDataCube":
        self._not_implemented()

    def sar_backscatter(self, args: SarBackscatterArgs) -> 'DriverDataCube':
        self._not_implemented()

    def resolution_merge(self, args: ResolutionMergeArgs) -> 'DriverDataCube':
        self._not_implemented()

    def resample_spatial(
        self,
        resolution: Union[float, Tuple[float, float]],
        projection: Union[int, str] = None,
        method: str = "near",
        align: str = "upper-left",
    ):
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

    # Note: the geometry _dimension_ is called "geometry" per https://github.com/Open-EO/openeo-api/issues/479,
    #       while some other internal aspects/attributes confusingly might be called "geometries".
    DIM_GEOMETRY = "geometry"
    DIM_BANDS = "bands"
    DIM_PROPERTIES = "properties"
    COLUMN_SELECTION_ALL = "all"
    COLUMN_SELECTION_NUMERICAL = "numerical"

    # Xarray cube attribute to indicate that it is a dummy cube
    CUBE_ATTR_VECTOR_CUBE_DUMMY = "vector_cube_dummy"

    def __init__(
        self,
        geometries: gpd.GeoDataFrame,
        cube: Optional[xarray.DataArray] = None,
    ):
        """

        :param geometries:
        :param cube:
        :param flatten_prefix: prefix for column/field/property names when flattening the cube
        """
        # TODO #114 EP-3981: lazy loading (like DelayedVector)?
        if cube is not None:
            if cube.dims[0] != self.DIM_GEOMETRY:
                log.error(f"First cube dim should be {self.DIM_GEOMETRY!r} but got dims {cube.dims!r}")
                raise VectorCubeError("Cube's first dimension is invalid.")
            if not geometries.index.equals(cube.indexes[cube.dims[0]]):
                log.error(f"Invalid VectorCube components {geometries.index=} != {cube.indexes[cube.dims[0]]=}")
                raise VectorCubeError("Incompatible vector cube components")
        self._geometries: gpd.GeoDataFrame = geometries
        self._cube = cube

    def filter_bands(self, bands) -> "DriverVectorCube":
        return self.with_cube(self._cube.sel({self.DIM_PROPERTIES: bands}))

    def with_cube(self, cube: xarray.DataArray) -> "DriverVectorCube":
        """Create new vector cube with same geometries but new cube"""
        log.info(f"Creating vector cube with new cube {cube.name!r}")
        return type(self)(geometries=self._geometries, cube=cube)

    @classmethod
    def from_geodataframe(
        cls,
        data: gpd.GeoDataFrame,
        *,
        columns_for_cube: Union[List[str], str] = COLUMN_SELECTION_NUMERICAL,
        dimension_name: str = DIM_PROPERTIES,
    ) -> "DriverVectorCube":
        """
        Build a DriverVectorCube from given GeoPandas data frame,
        using the data frame geometries as vector cube geometries
        and other columns (as specified) as cube values along a "bands" dimension

        :param data: geopandas data frame
        :param columns_for_cube: which data frame columns to use as cube values.
            One of:
            - "numerical": automatically pick numerical columns
            - "all": use all columns as cube values
            - list of column names
        :param dimension_name: name of the "bands" dimension
        :return: vector cube
        """
        available_columns = [c for c in data.columns if c != "geometry"]

        if columns_for_cube is None:
            # TODO #114: what should default selection be?
            columns_for_cube = cls.COLUMN_SELECTION_NUMERICAL

        if columns_for_cube == cls.COLUMN_SELECTION_NUMERICAL:
            columns_for_cube = [c for c in available_columns if numpy.issubdtype(data[c].dtype, numpy.number)]
        elif columns_for_cube == cls.COLUMN_SELECTION_ALL:
            columns_for_cube = available_columns
        elif isinstance(columns_for_cube, list):
            columns_for_cube = columns_for_cube
        else:
            raise ValueError(columns_for_cube)
        assert isinstance(columns_for_cube, list)

        if columns_for_cube:
            existing = [c for c in columns_for_cube if c in available_columns]
            to_add = [c for c in columns_for_cube if c not in available_columns]
            if existing:
                cube_df = data[existing]
                if to_add:
                    cube_df.loc[:, to_add] = numpy.nan
            else:
                cube_df = pandas.DataFrame(index=data.index, columns=to_add)

            # TODO: remove `columns_for_cube` from geopandas data frame?
            #   Enabling that triggers failure of som existing tests that use `aggregate_spatial`
            #   to "enrich" a vector cube with pre-existing properties
            #   Also see https://github.com/Open-EO/openeo-api/issues/504
            # geometries_df = data.drop(columns=columns_for_cube)
            geometries_df = data

            # TODO: leverage pandas `to_xarray` and xarray `to_array` instead of this manual building?
            cube: xarray.DataArray = xarray.DataArray(
                data=cube_df.values,
                dims=[cls.DIM_GEOMETRY, dimension_name],
                coords={
                    cls.DIM_GEOMETRY: data.geometry.index.to_list(),
                    dimension_name: cube_df.columns,
                },
            )
            return cls(geometries=geometries_df, cube=cube)

        else:
            # Use 1D dummy cube of NaN values
            cube: xarray.DataArray = xarray.DataArray(
                data=numpy.full(shape=[data.shape[0]], fill_value=numpy.nan),
                dims=[cls.DIM_GEOMETRY],
                coords={cls.DIM_GEOMETRY: data.geometry.index.to_list()},
                attrs={cls.CUBE_ATTR_VECTOR_CUBE_DUMMY: True},
            )
            return cls(geometries=data, cube=cube)

    @classmethod
    def from_fiona_supports(cls, format: str) -> bool:
        """Does `from_fiona` supports given format?"""
        # TODO: also cover input format options?
        return format.lower() in {"geojson", "esri shapefile", "gpkg", "parquet"}

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
        columns_for_cube = (options or {}).get("columns_for_cube", cls.COLUMN_SELECTION_NUMERICAL)
        # TODO #114 EP-3981: lazy loading like/with DelayedVector
        # note for GeoJSON: will consider Feature.id as well as Feature.properties.id
        if driver and "parquet" == driver.lower():
            return cls.from_parquet(paths=paths, columns_for_cube=columns_for_cube)
        else:
            gdf = gpd.read_file(paths[0], driver=driver)
            return cls.from_geodataframe(gdf, columns_for_cube=columns_for_cube)

    @classmethod
    def from_parquet(
        cls,
        paths: List[Union[str, Path]],
        columns_for_cube: Union[List[str], str] = COLUMN_SELECTION_NUMERICAL,
    ):
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
        return cls.from_geodataframe(df, columns_for_cube=columns_for_cube)

    def write_to_parquet(
        self, path: str, flatten_prefix: Optional[str] = None, include_properties=True, only_numeric=True
    ):
        return self._as_geopandas_df(
            flatten_prefix=flatten_prefix, include_properties=include_properties, only_numeric=only_numeric
        ).to_parquet(path)

    @classmethod
    def from_geojson(
        cls,
        geojson: dict,
        columns_for_cube: Union[List[str], str] = COLUMN_SELECTION_NUMERICAL,
    ) -> "DriverVectorCube":
        """Construct vector cube from GeoJson dict structure"""
        validate_geojson_coordinates(geojson)
        # TODO support more geojson types?
        if geojson["type"] in {"Polygon", "MultiPolygon", "Point", "MultiPoint"}:
            features = [{"type": "Feature", "geometry": geojson, "properties": {}}]
        elif geojson["type"] in {"Feature"}:
            features = [geojson]
        elif geojson["type"] in {"GeometryCollection"}:
            # TODO #71 #114 Deprecate/avoid usage of GeometryCollection
            log.warning(
                "Input GeoJSON of deprecated type 'GeometryCollection', please use a FeatureCollection or another type of Multi geometry."
            )
            features = [{"type": "Feature", "geometry": g, "properties": {}} for g in geojson["geometries"]]
        elif geojson["type"] in {"FeatureCollection"}:
            features = geojson
        else:
            raise FeatureUnsupportedException(
                f"Can not construct DriverVectorCube from {geojson.get('type', type(geojson))!r}"
            )
        gdf = gpd.GeoDataFrame.from_features(features)
        return cls.from_geodataframe(gdf, columns_for_cube=columns_for_cube)

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

    def _as_geopandas_df(
        self,
        flatten_prefix: Optional[str] = None,
        flatten_name_joiner: str = "~",
        include_properties: bool = True,
        only_numeric: bool = False,
    ) -> gpd.GeoDataFrame:
        """Join geometries and cube as a geopandas dataframe"""
        # TODO: avoid copy?
        df = self._geometries.copy(deep=True)
        if not include_properties:
            df = df[[df.geometry.name]]
        if self._cube is not None and not self._cube.attrs.get(self.CUBE_ATTR_VECTOR_CUBE_DUMMY):
            assert self._cube.dims[0] == self.DIM_GEOMETRY
            # TODO: better way to combine cube with geometries
            # Flatten multiple (non-geometry) dimensions from cube to new properties in geopandas dataframe
            if self._cube.dims[1:]:
                stacked = self._cube.stack(prop=self._cube.dims[1:])
                log.info(f"Flattened cube component of vector cube to {stacked.shape[1]} properties")
                name_prefix = [flatten_prefix] if flatten_prefix else []
                for p in stacked.indexes["prop"]:
                    if only_numeric and type(stacked.sel(prop=p)[0].item()) not in [int, float]:
                        continue
                    name = flatten_name_joiner.join(str(x) for x in name_prefix + list(p))
                    # TODO: avoid column collisions?
                    df[name] = stacked.sel(prop=p)
            else:
                # TODO: better fallback column/property name in this case?
                df[flatten_prefix or "_vc"] = self._cube

        return df

    def to_geojson(self, flatten_prefix: Optional[str] = None, include_properties: bool = True) -> dict:
        """Export as GeoJSON FeatureCollection."""
        return shapely.geometry.mapping(
            self._as_geopandas_df(flatten_prefix=flatten_prefix, include_properties=include_properties)
        )

    def to_wkt(self) -> List[str]:
        wkts = [str(g) for g in self._geometries.geometry]
        return wkts

    def to_internal_json(self) -> dict:
        """
        Export to an internal JSON-style representation.
        Subject to change any time: not intended for public consumption, just for (unit) test purposes.
        """
        return {
            "geometries": shapely.geometry.mapping(self._geometries),
            "cube": self._cube.to_dict(data="array") if self._cube is not None else None,
        }

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

        gdf = self._as_geopandas_df(flatten_prefix=options.get("flatten_prefix"))
        gdf.to_file(path, driver=format_info.fiona_driver)

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

        if self._cube is None or self._cube.attrs.get(self.CUBE_ATTR_VECTOR_CUBE_DUMMY):
            # No cube: no real data to return (in legacy style), so let's just return a `null` per geometry.
            return JSONResult(data=[None] * self.geometry_count())

        cube = self._cube
        # TODO: more flexible temporal/band dimension detection?
        if cube.dims == (self.DIM_GEOMETRY, "t"):
            # Add single band dimension
            cube = cube.expand_dims({"bands": ["band"]}, axis=-1)
        if cube.dims == (self.DIM_GEOMETRY, "t", "bands"):
            cube = cube.transpose("t", self.DIM_GEOMETRY, "bands")
            timeseries = {
                t.item(): t_slice.values.tolist()
                for t, t_slice in zip(cube.coords["t"], cube)
            }
            return AggregatePolygonResult(timeseries=timeseries, regions=self)
        elif cube.dims == (self.DIM_GEOMETRY, "bands"):
            # This covers the legacy `AggregatePolygonSpatialResult` code path,
            # but as AggregatePolygonSpatialResult's constructor expects a folder of CSV file(s),
            # we keep it simple here with a basic JSONResult result.
            cube = cube.transpose(self.DIM_GEOMETRY, "bands")
            return JSONResult(data=cube.values.tolist())
        raise ValueError(
            f"Unsupported cube configuration {cube.dims} for _write_legacy_aggregate_polygon_result_json"
        )

    def get_dimension_names(self) -> List[str]:
        if self._cube is None:
            return [self.DIM_GEOMETRY]
        else:
            return list(str(d) for d in self._cube.dims)

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

    def get_cube(self) -> Optional[xarray.DataArray]:
        return self._cube

    def get_ids(self) -> Optional[Sequence]:
        return self._geometries.get("id")

    def get_xarray_cube_basics(self) -> Tuple[tuple, dict]:
        """Get initial dims/coords for xarray DataArray construction"""
        dims = (self.DIM_GEOMETRY,)
        coords = {self.DIM_GEOMETRY: self._geometries.index.to_list()}
        return dims, coords

    def __eq__(self, other):
        return isinstance(other, DriverVectorCube) and numpy.array_equal(
            self._as_geopandas_df().values, other._as_geopandas_df().values
        )

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

    def apply_dimension(
        self,
        process: dict,
        *,
        dimension: str,
        target_dimension: Optional[str] = None,
        context: Optional[dict] = None,
        env: EvalEnv,
    ) -> "DriverVectorCube":
        # Is callback a single run_udf node process?
        single_run_udf = SingleRunUDFProcessGraph.parse_or_none(process)

        if single_run_udf:
            # Process with single "run_udf" node
            if single_run_udf.data != {"from_parameter": "data"}:
                raise ProcessGraphInvalidException(
                    message="Vector cube `apply_dimension` process does not reference `data` parameter."
                )
            if (
                dimension == self.DIM_GEOMETRY
                or (dimension in {self.DIM_BANDS, self.DIM_PROPERTIES}.intersection(self.get_dimension_names()))
                and target_dimension is None
            ):
                log.warning(
                    f"Using experimental feature: DriverVectorCube.apply_dimension along dim {dimension} and empty cube"
                )
                # TODO: data chunking (e.g. large feature collections)
                gdf = self._as_geopandas_df()
                feature_collection = openeo.udf.FeatureCollection(id="_", data=gdf)
                # TODO: dedicated UDF signature to indicate to work on vector cube through a feature collection based API
                udf_data = openeo.udf.UdfData(
                    proj={"EPSG": self._geometries.crs.to_epsg()} if self._geometries.crs else None,
                    feature_collection_list=[feature_collection],
                    user_context=context,
                )
                log.info(f"[run_udf] Running UDF {str_truncate(single_run_udf.udf, width=256)!r} on {udf_data!r}")
                result_data = env.backend_implementation.processing.run_udf(udf=single_run_udf.udf, data=udf_data)
                log.info(f"[run_udf] UDF resulted in {result_data!r}")

                if not isinstance(result_data, openeo.udf.UdfData):
                    raise ValueError(f"UDF should return UdfData, but got {type(result_data)}")
                result_features = result_data.get_feature_collection_list()
                if not (result_features and len(result_features) == 1):
                    raise ValueError(
                        f"UDF should return single feature collection but got {result_features and len(result_features)}"
                    )
                return DriverVectorCube.from_geodataframe(result_features[0].data)

        raise FeatureUnsupportedException(
            message=f"DriverVectorCube.apply_dimension with {dimension=} and {bool(single_run_udf)=}"
        )


class DriverMlModel:
    """Base class for driver-side 'ml-model' data structures"""
    METADATA_FILE_NAME = "ml_model_metadata.json"

    def get_model_metadata(self, directory: Union[str, Path]) -> Dict[str, Any]:
        raise NotImplementedError

    def write_assets(self, directory: Union[str, Path]) -> Dict[str, StacAsset]:
        raise NotImplementedError
