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

    def mask(self, mask: 'DriverDataCube', replacement=None) -> 'DriverDataCube':
        raise NotImplementedError

    def merge(self, other: 'DriverDataCube', overlap_resolver) -> 'DriverDataCube':
        raise NotImplementedError
