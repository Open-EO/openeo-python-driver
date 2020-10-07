import collections
from typing import List, Union

from openeo_driver.datacube import DriverDataCube


class DataJournal:
    """
    Journal of certain operations applied on a spatio-temporal data cube,
    e.g. to keep track of all filter_temporal/filter_bbox/resample operations.

    The journal is immutable in the sense that adding an entry creates a new journal with copied entries first.
    """
    Entry = collections.namedtuple("Entry", ["operation", "arguments"])

    def __init__(self, journal: List[Entry] = None):
        self._journal = journal or []

    def __repr__(self):
        return '<{c!r}: {j!r}>'.format(c=self.__class__, j=self._journal)

    def add(self, operation: str, arguments: Union[dict, tuple]) -> 'DataJournal':
        """Copy journal with extra operation appended."""
        return DataJournal(journal=list(self._journal) + [self.Entry(operation, arguments)])

    def get(self, operation: str) -> List[Union[dict, tuple]]:
        """Get list of arguments for entries of given operation"""
        return [entry.arguments for entry in self._journal if entry.operation == operation]


class DryRunDataCube(DriverDataCube):
    """
    Data cube (mock/spy) to be used for a process graph dry-run,
    to detect data cube constraints (filter_bbox, filter_temporal, ...), resolution, tile layout,
    estimate memory/cpu usage, ...
    """

    def __init__(self, data_journals: List[DataJournal]):
        # TODO: can/should we work with real metadata?
        super(DryRunDataCube, self).__init__(metadata=None)
        self._journals = data_journals

    @property
    def journals(self) -> List[DataJournal]:
        return self._journals

    @classmethod
    def load_collection(cls, collection_id: str, arguments: dict) -> 'DryRunDataCube':
        """Create a DryRunDataCube from a `load_collection` process."""
        cube = cls([DataJournal().add("load_collection", {"collection_id": collection_id})])
        if "temporal_extent" in arguments:
            cube = cube.filter_temporal(*arguments["temporal_extent"])
        if "spatial_extent" in arguments:
            cube = cube.filter_bbox(**arguments["spatial_extent"])
        if "bands" in arguments:
            cube = cube.filter_bands(arguments["bands"])
        # TODO: load_collection `properties` argument
        return cube

    def _process(self, operation, arguments) -> 'DryRunDataCube':
        # New data cube with operation added to each journal
        journals = [journal.add(operation, arguments) for journal in self._journals]
        return DryRunDataCube(journals)

    def filter_temporal(self, start: str, end: str) -> 'DryRunDataCube':
        return self._process("filter_temporal", (start, end))

    def filter_bbox(self, west, south, east, north, crs=None, base=None, height=None) -> 'DryRunDataCube':
        return self._process("filter_bbox", {"west": west, "south": south, "east": east, "north": north, "crs": crs})

    def filter_bands(self, bands) -> 'DryRunDataCube':
        return self._process("filter_bands", bands)

    def mask(self, mask: 'DryRunDataCube', replacement=None) -> 'DryRunDataCube':
        # TODO: if mask cube has no temporal or bbox extent: copy from self?
        # TODO: or add reference to the self journal to the mask journal and vice versa?
        return DryRunDataCube(data_journals=self._journals + mask._journals)

    def merge_cubes(self, other: 'DryRunDataCube', overlap_resolver) -> 'DryRunDataCube':
        # TODO:
        return DryRunDataCube(data_journals=self._journals + other._journals)
