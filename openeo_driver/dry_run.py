import collections
from typing import List, Union

import shapely.geometry.base

from openeo.metadata import CollectionMetadata
from openeo_driver.datacube import DriverDataCube
from openeo_driver.delayed_vector import DelayedVector
from openeo_driver.save_result import AggregatePolygonResult
from openeo_driver.utils import geojson_to_geometry


class DataJournal:
    """
    Journal of certain operations applied on a spatio-temporal data cube,
    e.g. to keep track of all filter_temporal/filter_bbox/resample operations.

    The journal is immutable in the sense that adding an entry creates a new journal with copied entries first.
    """

    class Creation(tuple):
        """Hashable identifier for how a data cube is created (load_collection, load_disk_data, ...)"""
        pass

    Entry = collections.namedtuple("Entry", ["operation", "arguments"])

    def __init__(self, creation: Creation, journal: List[Entry] = None):
        # How the data cube was created initially (load_collection or something else?)
        self._creation = creation
        # Additional operations on the data cube
        self._journal = journal or []

    @property
    def creation(self):
        return self._creation

    def __repr__(self):
        return '<{c!r}: {r!r} {j!r}>'.format(c=self.__class__, r=self._creation, j=self._journal)

    def add(self, operation: str, arguments: Union[dict, tuple]) -> 'DataJournal':
        """Copy journal with extra operation appended."""
        return DataJournal(creation=self._creation, journal=list(self._journal) + [self.Entry(operation, arguments)])

    def get(self, operation: str) -> List[Union[dict, tuple]]:
        """Get list of arguments for entries of given operation"""
        return [entry.arguments for entry in self._journal if entry.operation == operation]


class DryRunDataCube(DriverDataCube):
    """
    Data cube (mock/spy) to be used for a process graph dry-run,
    to detect data cube constraints (filter_bbox, filter_temporal, ...), resolution, tile layout,
    estimate memory/cpu usage, ...
    """

    def __init__(self, data_journals: List[DataJournal], metadata: CollectionMetadata = None):
        # TODO: can/should we work with real metadata?
        super(DryRunDataCube, self).__init__(metadata=metadata)
        self._journals = data_journals

    @property
    def journals(self) -> List[DataJournal]:
        return self._journals

    @staticmethod
    def load_collection_id(collection_id: str) -> DataJournal.Creation:
        """Hashable identifier for data cube creation with load_collection call"""
        return DataJournal.Creation(("load_collection", collection_id))

    @classmethod
    def load_collection(cls, collection_id: str, arguments: dict, metadata: dict = None) -> 'DryRunDataCube':
        """Create a DryRunDataCube from a `load_collection` process."""
        cube = cls(
            data_journals=[DataJournal(creation=cls.load_collection_id(collection_id=collection_id))],
            metadata=metadata
        )
        if "temporal_extent" in arguments:
            cube = cube.filter_temporal(*arguments["temporal_extent"])
        if "spatial_extent" in arguments:
            cube = cube.filter_bbox(**arguments["spatial_extent"])
        if "bands" in arguments:
            cube = cube.filter_bands(arguments["bands"])
        # TODO: load_collection `properties` argument
        return cube

    @staticmethod
    def load_data_disk_id(glob_pattern: str, format: str, options: dict) -> DataJournal.Creation:
        """Hashable identifier for data cube creation with load_disk_data call."""
        return DataJournal.Creation(("load_disk_data", glob_pattern, format, tuple(sorted(options.items()))))

    @classmethod
    def load_disk_data(cls, glob_pattern: str, format: str, options: dict):
        creation = cls.load_data_disk_id(glob_pattern=glob_pattern, format=format, options=options)
        return cls(data_journals=[DataJournal(creation=creation)])

    def _process(self, operation, arguments) -> 'DryRunDataCube':
        # New data cube with operation added to each journal
        journals = [journal.add(operation, arguments) for journal in self._journals]
        return DryRunDataCube(journals, metadata=self.metadata)

    def filter_temporal(self, start: str, end: str) -> 'DryRunDataCube':
        return self._process("temporal_extent", (start, end))

    def filter_bbox(self, west, south, east, north, crs=None, base=None, height=None) -> 'DryRunDataCube':
        return self._process("spatial_extent", {"west": west, "south": south, "east": east, "north": north, "crs": crs})

    def filter_bands(self, bands) -> 'DryRunDataCube':
        return self._process("bands", bands)

    def mask(self, mask: 'DryRunDataCube', replacement=None) -> 'DryRunDataCube':
        # TODO: if mask cube has no temporal or bbox extent: copy from self?
        # TODO: or add reference to the self journal to the mask journal and vice versa?
        return DryRunDataCube(data_journals=self._journals + mask._journals)

    def merge_cubes(self, other: 'DryRunDataCube', overlap_resolver) -> 'DryRunDataCube':
        # TODO:
        return DryRunDataCube(data_journals=self._journals + other._journals)

    def aggregate_spatial(
            self, geometries: Union[str, dict, DelayedVector, shapely.geometry.base.BaseGeometry],
            reducer, target_dimension: str = "result"
    ) -> AggregatePolygonResult:
        if isinstance(geometries, dict):
            geometries = geojson_to_geometry(geometries)
            bbox = geometries.bounds
        elif isinstance(geometries, str):
            bbox = DelayedVector(geometries).bounds
        elif isinstance(geometries, DelayedVector):
            bbox = geometries.bounds
        elif isinstance(geometries, shapely.geometry.base.BaseGeometry):
            bbox = geometries.bounds
        else:
            raise ValueError(geometries)
        self.filter_bbox(west=bbox[0], south=bbox[1], east=bbox[2], north=bbox[3], crs="EPSG:4326")
        return AggregatePolygonResult(timeseries={}, regions=geometries)

    def zonal_statistics(self, regions, func: str) -> AggregatePolygonResult:
        return self.aggregate_spatial(geometries=regions, reducer=func)

    def resample_cube_spatial(self, target: 'DryRunDataCube', method: str = 'near') -> 'DryRunDataCube':
        # TODO: EP3561 record resampling operation
        return self

    def _nop(self, *args, **kwargs) -> 'DryRunDataCube':
        """No Operation: do nothing"""
        return self

    # TODO: some methods need metadata manipulation?
    apply_kernel = _nop
    apply_neighborhood = _nop
    apply = _nop
    apply_tiles = _nop
    apply_tiles_spatiotemporal = _nop
    apply_dimension = _nop
    reduce = _nop
    reduce_dimension = _nop
    reduce_bands = _nop
    mask_polygon = _nop
    add_dimension = _nop
    aggregate_temporal = _nop
    rename_labels = _nop
    rename_dimension = _nop
    ndvi = _nop


def extract_load_collection_constraints(cube: DryRunDataCube) -> dict:
    collections_constraints = {}
    for journal in cube.journals:
        creation_id = journal.creation
        constraints = {}
        for operation in ["temporal_extent", "spatial_extent", "bands"]:
            ops = journal.get(operation)
            if ops:
                # Take first item (to reproduce original behavior)
                # TODO: take temporal/spatial/categorical intersection instead
                constraints[operation] = ops[0]
        if creation_id not in collections_constraints:
            collections_constraints[creation_id] = constraints
        else:
            raise RuntimeError("TODO: combine journal constraints?")
    return collections_constraints
