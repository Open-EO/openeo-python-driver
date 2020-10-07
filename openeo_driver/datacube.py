from typing import List

from openeo import ImageCollection
from openeo.metadata import CollectionMetadata


class DriverDataCube(ImageCollection):
    """Base class for "driver" side data cubes."""

    def __init__(self, metadata: CollectionMetadata = None):
        self.metadata = metadata if isinstance(metadata, CollectionMetadata) else CollectionMetadata(metadata or {})

    def filter_temporal(self, start: str, end: str) -> 'DriverDataCube':
        raise NotImplementedError

    def filter_bbox(self, west, south, east, north, crs=None, base=None, height=None) -> 'DriverDataCube':
        raise NotImplementedError

    def filter_bands(self, bands) -> 'DriverDataCube':
        raise NotImplementedError

    def apply(self, process) -> 'DriverDataCube':
        raise NotImplementedError

    def apply_kernel(self, kernel: list, factor=1, border=0, replace_invalid=0) -> 'DriverDataCube':
        raise NotImplementedError

    def apply_neighborhood(self, process, size: List[dict], overlap: List[dict]) -> 'DriverDataCube':
        raise NotImplementedError

    def apply_dimension(self, process, dimension: str, target_dimension: str) -> 'DriverDataCube':
        raise NotImplementedError

    def reduce_dimension(self, reducer, dimension: str) -> 'DriverDataCube':
        raise NotImplementedError

    def add_dimension(self, name: str, label, type: str = "other") -> 'DriverDataCube':
        raise NotImplementedError

    def reduce(self, reducer: str, dimension: str) -> 'DriverDataCube':
        # TODO #47: remove  this deprecated 0.4-style method
        raise NotImplementedError

    def rename_dimension(self, source: str, target: str) -> 'DriverDataCube':
        raise NotImplementedError

    def rename_labels(self, dimension: str, target: list, source: list = None) -> 'DriverDataCube':
        raise NotImplementedError

    def reduce_bands(self, process) -> 'DriverDataCube':
        # TODO #47: remove this non-standard process
        raise NotImplementedError

    def mask(self, mask: 'DriverDataCube', replacement=None) -> 'DriverDataCube':
        raise NotImplementedError

    def mask_polygon(self, mask, replacement=None, inside: bool = False) -> 'DriverDataCube':
        raise NotImplementedError

    def merge_cubes(self, other: 'DriverDataCube', overlap_resolver) -> 'DriverDataCube':
        raise NotImplementedError

    def resample_cube_spatial(self, target: 'DriverDataCube', method: str = 'near') -> 'DriverDataCube':
        raise NotImplementedError

    def aggregate_temporal(self, intervals: list, reducer, labels: list = None,
                           dimension: str = None) -> 'DriverDataCube':
        raise NotImplementedError

    def aggregate_spatial(self, geometries, reducer, target_dimension: str) -> dict:
        raise NotImplementedError

    def timeseries(self, x, y, srs="EPSG:4326") -> dict:
        # TODO #47: remove this non-standard process
        raise NotImplementedError

    def ndvi(self, nir: str = "nir", red: str = "red", target_band: str = None) -> 'DriverDataCube':
        raise NotImplementedError

    def download(self, outputfile: str, **format_options) -> str:
        raise NotImplementedError
