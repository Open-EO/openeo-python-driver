import inspect
from typing import List

import geopandas as gpd

from openeo import ImageCollection
from openeo.metadata import CollectionMetadata
from openeo_driver.datastructs import SarBackscatterArgs, ResolutionMergeArgs
from openeo_driver.errors import FeatureUnsupportedException
from openeo_driver.utils import EvalEnv


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

    def reduce_dimension(self, reducer, dimension: str, env: EvalEnv) -> 'DriverDataCube':
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

    def mask_polygon(self, mask, replacement=None, inside: bool = False) -> 'DriverDataCube':
        self._not_implemented()

    def merge_cubes(self, other: 'DriverDataCube', overlap_resolver) -> 'DriverDataCube':
        self._not_implemented()

    def resample_cube_spatial(self, target: 'DriverDataCube', method: str = 'near') -> 'DriverDataCube':
        self._not_implemented()

    def aggregate_temporal(self, intervals: list, reducer, labels: list = None,
                           dimension: str = None, context:dict = None) -> 'DriverDataCube':
        self._not_implemented()

    def aggregate_spatial(self, geometries: dict, reducer, target_dimension: str = "result") -> 'DriverDataCube':
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

    def fit_class_random_forest(self, predictors, target, training, num_trees, mtry):
        self._not_implemented()


class DriverVectorCube:
    """
    Base class for driver-side 'vector cubes'

    Conceptually comparable to GeoJSON FeatureCollections, but possibly more advanced with more dimensions, bands, ...
    """

    def __init__(self, data: gpd.GeoDataFrame):
        # TODO EP-3981: consider other data containers (xarray) and lazy loading?
        self.data = data

    @classmethod
    def from_geojson(cls, paths: List[str], options: dict):
        # TODO EP-3981: provide a more general factory instead of this GeoJSON-specific one?
        if len(paths) != 1:
            # TODO EP-3981: support multiple paths
            raise FeatureUnsupportedException(message="Loading a vector cube from multiple files is not supported")
        # TODO EP-3981: lazy loading like/with DelayedVector
        return cls(data=gpd.read_file(paths[0]))

    def save_result(self, filename: str, format: str, format_options: dict = None) -> str:
        # TODO EP-3981: proper mapping of format to driver
        self.data.to_file(filename, driver=format)
        return filename
