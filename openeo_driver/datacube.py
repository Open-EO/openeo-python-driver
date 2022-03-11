import inspect
import logging
import zipfile
from pathlib import Path
from typing import List, Union, Optional, Dict, Any, Tuple, Sequence

import geopandas as gpd
import pandas as pd
import shapely.geometry
import shapely.geometry.base
import shapely.ops
import xarray

from openeo import ImageCollection
from openeo.metadata import CollectionMetadata
from openeo.util import ensure_dir
from openeo_driver.datastructs import SarBackscatterArgs, ResolutionMergeArgs, StacAsset
from openeo_driver.errors import FeatureUnsupportedException, InternalException
from openeo_driver.util.ioformats import IOFORMATS
from openeo_driver.utils import EvalEnv

log = logging.getLogger(__name__)


class DriverDataCube(ImageCollection):
    """Base class for "driver" side raster data cubes."""

    # TODO cut the openeo.ImageCollection chord (https://github.com/Open-EO/openeo-python-client/issues/100)

    def __init__(self, metadata: CollectionMetadata = None):
        self.metadata = CollectionMetadata.get_or_create(metadata)

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

    def reduce(self, reducer: str, dimension: str) -> 'DriverDataCube':
        # TODO #47: remove  this deprecated 0.4-style method
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
        self._not_implemented()

    def zonal_statistics(self, regions, func:str) -> 'DriverDataCube':
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

    def __init__(self, geometries: gpd.GeoDataFrame, cube: Optional[xarray.DataArray] = None):
        # TODO EP-3981: consider other data containers (xarray) and lazy loading?
        if cube is not None:
            if cube.dims[0] != self.DIM_GEOMETRIES:
                log.error(f"First cube dim should be {self.DIM_GEOMETRIES!r} but got dims {cube.dims!r}")
                raise VectorCubeError("Cube's first dimension is invalid.")
            if not geometries.index.equals(cube.indexes[cube.dims[0]]):
                log.error(f"Invalid VectorCube components {geometries.index!r} != {cube.indexes[cube.dims[0]]!r}")
                raise VectorCubeError("Incompatible vector cube components")
        self._geometries = geometries
        self._cube = cube

    def with_cube(self, cube: xarray.DataArray) -> "DriverVectorCube":
        """Create new vector cube with same geometries but new cube"""
        log.info(f"Creating vector cube with new cube {cube.name!r}")
        return DriverVectorCube(geometries=self._geometries, cube=cube)

    @classmethod
    def from_fiona(cls, paths: List[str], driver: str, options: dict):
        """Factory to load vector cube data using fiona/GeoPandas."""
        if len(paths) != 1:
            # TODO EP-3981: support multiple paths
            raise FeatureUnsupportedException(message="Loading a vector cube from multiple files is not supported")
        # TODO EP-3981: lazy loading like/with DelayedVector
        return cls(geometries=gpd.read_file(paths[0], driver=driver))

    def _as_geopandas_df(self) -> gpd.GeoDataFrame:
        """Join geometries and cube as a geopandas dataframe"""
        # TODO: avoid copy?
        df = self._geometries.copy(deep=True)
        if self._cube is not None:
            # TODO: better way to combine cube with geometries
            # Flatten multiple (non-geometry) dimensions from cube to new properties in geopandas dataframe
            stacked = self._cube.stack(prop=self._cube.dims[1:])
            log.info(f"Flattened cube component of vector cube to {stacked.shape[1]} properties")
            for p in stacked.indexes["prop"]:
                name = "~".join(str(x) for x in p)
                # TODO: avoid column collisions?
                df[name] = stacked.sel(prop=p)
        return df

    def to_geojson(self):
        return shapely.geometry.mapping(self._as_geopandas_df())

    def write_assets(self, directory: Union[str, Path], format: str, options: Optional[dict] = None) -> Dict[str, StacAsset]:
        directory = ensure_dir(directory)
        format_info = IOFORMATS.get(format)
        # TODO: check if format can be used for vector data?
        path = directory / f"vectorcube.{format_info.extension}"
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

    def get_bounding_box(self) -> Tuple[float, float, float, float]:
        return self._geometries.total_bounds

    def get_geometries(self) -> Sequence[shapely.geometry.base.BaseGeometry]:
        return self._geometries.geometry

    def get_geometries_index(self) -> pd.Index:
        return self._geometries.index


class DriverMlModel:
    """Base class for driver-side 'ml-model' data structures"""

    def write_assets(self, directory: Union[str, Path]) -> Dict[str, StacAsset]:
        raise NotImplementedError
