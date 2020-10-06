from openeo import ImageCollection
from openeo.metadata import CollectionMetadata


class DriverDataCube(ImageCollection):
    """Base class for "driver" side data cubes."""

    def __init__(self, metadata: CollectionMetadata = None):
        self.metadata = metadata if isinstance(metadata, CollectionMetadata) else CollectionMetadata(metadata or {})
