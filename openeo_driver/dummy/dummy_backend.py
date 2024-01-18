import json
import numbers
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Union, Tuple, Optional, Iterable, Any, Sequence
import unittest.mock
from unittest.mock import Mock

import flask
import numpy
import xarray
from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry.base import BaseGeometry, BaseMultipartGeometry
from shapely.geometry.collection import GeometryCollection

from openeo.api.logs import normalize_log_level
from openeo.internal.process_graph_visitor import ProcessGraphVisitor
from openeo.metadata import CollectionMetadata, Band, SpatialDimension, TemporalDimension, BandDimension
import openeo.udf
from openeo_driver.ProcessGraphDeserializer import ConcreteProcessing
from openeo_driver.backend import (
    SecondaryServices,
    OpenEoBackendImplementation,
    CollectionCatalog,
    ServiceMetadata,
    BatchJobs,
    BatchJobMetadata,
    OidcProvider,
    UserDefinedProcesses,
    UserDefinedProcessMetadata,
    LoadParameters,
    Processing,
    BatchJobResultMetadata,
)
from openeo_driver.config import OpenEoBackendConfig
from openeo_driver.constants import STAC_EXTENSION
from openeo_driver.datacube import DriverDataCube, DriverMlModel, DriverVectorCube
from openeo_driver.datastructs import StacAsset
from openeo_driver.delayed_vector import DelayedVector
from openeo_driver.dry_run import SourceConstraint
from openeo_driver.errors import JobNotFoundException, JobNotFinishedException, ProcessGraphNotFoundException, \
    PermissionsInsufficientException
from openeo_driver.jobregistry import JOB_STATUS
from openeo_driver.save_result import AggregatePolygonResult, AggregatePolygonSpatialResult
from openeo_driver.users import User
from openeo_driver.utils import EvalEnv, generate_unique_id, WhiteListEvalEnv

DEFAULT_DATETIME = datetime(2020, 4, 23, 16, 20, 27)

# TODO: eliminate this global state with proper pytest fixture usage!
_collections = {}
_load_collection_calls = {}


def utcnow() -> datetime:
    # To simplify testing, we break time.
    # TODO: just start using `time_machine` module for time mocking
    return DEFAULT_DATETIME


def get_collection(collection_id: str) -> 'DummyDataCube':
    return _collections[collection_id]


def _register_load_collection_call(collection_id: str, load_params: LoadParameters):
    if collection_id not in _load_collection_calls:
        _load_collection_calls[collection_id] = []
    _load_collection_calls[collection_id].append(load_params.copy())


def all_load_collection_calls(collection_id: str) -> List[LoadParameters]:
    return _load_collection_calls[collection_id]


def last_load_collection_call(collection_id: str) -> LoadParameters:
    return _load_collection_calls[collection_id][-1]


def reset(backend=None):
    # TODO: can we eliminate reset now?
    if backend is not None:
        backend.catalog._load_collection_cached.cache_clear()
    global _collections, _load_collection_calls
    _collections = {}
    _load_collection_calls = {}


class DummyVisitor(ProcessGraphVisitor):

    def __init__(self):
        super(DummyVisitor, self).__init__()
        self.processes = []

    def enterProcess(self, process_id: str, arguments: dict, namespace: Union[str, None]):
        self.processes.append((process_id, arguments, namespace))

    def constantArgument(self, argument_id: str, value):
        if isinstance(value, numbers.Real):
            pass
        elif isinstance(value, str):
            pass
        else:
            raise ValueError(
                'Only numeric constants are accepted, but got: ' + str(value) + ' for argument: ' + str(
                    argument_id))
        return self


class DummySecondaryServices(SecondaryServices):
    _registry = [
        ServiceMetadata(
            id="wmts-foo",
            process={"process_graph": {"foo": {"process_id": "foo", "arguments": {}}}},
            url='https://oeo.net/wmts/foo',
            type="WMTS",
            enabled=True,
            configuration={"version": "0.5.8"},
            attributes={},
            title="Test service",
            created=datetime(2020, 4, 9, 15, 5, 8)
        )
    ]

    def _create_service(self, user_id: str, process_graph: dict, service_type: str, api_version: str,
                        configuration: dict) -> str:
        service_id = 'c63d6c27-c4c2-4160-b7bd-9e32f582daec'
        return service_id

    def service_types(self) -> dict:
        return {
            "WMTS": {
                "title": "Web Map Tile Service",
                "configuration": {
                    "version": {
                        "type": "string",
                        "description": "The WMTS version to use.",
                        "default": "1.0.0",
                        "enum": [
                            "1.0.0"
                        ]
                    }
                },
                "process_parameters": [
                    # TODO: we should at least have bbox and time range parameters here
                ],
                "links": [],
            }
        }

    def list_services(self, user_id: str) -> List[ServiceMetadata]:
        return self._registry

    def service_info(self, user_id: str, service_id: str) -> ServiceMetadata:
        return next(s for s in self._registry if s.id == service_id)

    def get_log_entries(self, service_id: str, user_id: str, offset: str) -> List[dict]:
        return [
            {"id": 3, "level": "info", "message": "Loaded data."}
        ]


def mock_side_effect(fun):
    """
    Decorator to flag a DummyDataCube method to be wrapped in Mock(side_effect=...)
    so that it allows to mock-style inspection (call_count, assert_called_once, ...)
    while still providing a real implementation.
    """
    fun._mock_side_effect = True
    return fun


class DummyDataCube(DriverDataCube):

    def __init__(self, metadata: CollectionMetadata = None):
        super(DummyDataCube, self).__init__(metadata=metadata)

        # TODO #47: remove this non-standard process?
        self.timeseries = Mock(name="timeseries", return_value={})

        # TODO can we get rid of these non-standard "apply_tiles" processes?
        self.apply_tiles = Mock(name="apply_tiles", return_value=self)
        self.apply_tiles_spatiotemporal = Mock(name="apply_tiles_spatiotemporal", return_value=self)

        # Create mock methods for remaining DriverDataCube methods that are not yet defined directly by DummyDataCube
        to_keep = set(DummyDataCube.__dict__.keys()).union(self.__dict__.keys())
        to_keep.update(m for m in DriverDataCube.__dict__.keys() if m.startswith("_"))
        to_keep.update(["get_dimension_names"])
        for name, method in DriverDataCube.__dict__.items():
            if not name in to_keep and callable(method):
                setattr(self, name, Mock(name=name, return_value=self))

        for name in [n for n, m in DummyDataCube.__dict__.items() if getattr(m, '_mock_side_effect', False)]:
            setattr(self, name, Mock(side_effect=getattr(self, name)))

    @mock_side_effect
    def reduce_dimension(
        self, reducer, *, dimension: str, context: Optional[dict] = None, env: EvalEnv
    ) -> "DummyDataCube":
        return DummyDataCube(self.metadata.reduce_dimension(dimension_name=dimension))

    @mock_side_effect
    def add_dimension(self, name: str, label, type: str = "other") -> 'DummyDataCube':
        return DummyDataCube(self.metadata.add_dimension(name=name, label=label, type=type))

    @mock_side_effect
    def drop_dimension(self, name: str) -> 'DriverDataCube':
        return DummyDataCube(self.metadata.drop_dimension(name=name))

    @mock_side_effect
    def dimension_labels(self, dimension: str) -> 'DriverDataCube':
        return self.metadata.dimension_names()

    def _metadata_to_dict(self):
        dimensions = {d.name: {"type": d.type} for d in self.metadata._dimensions}
        return {"cube:dimensions": dimensions}

    def save_result(self, filename: str, format: str, format_options: dict = None) -> str:
        # TODO: this method should be deprecated (limited to single asset) in favor of write_assets (supports multiple assets)
        if "JSON" == format.upper():
            import json
            with open(filename, "w") as f:
                json.dump(self._metadata_to_dict(),f,indent=2)
        else:
            with open(filename, "w") as f:
                f.write("{f}:save_result({s!r}".format(f=format, s=self))
        return filename

    def aggregate_spatial(
            self,
            geometries: Union[BaseGeometry, str, DriverVectorCube],
            reducer: dict,
            target_dimension: str = "result",
    ) -> Union[AggregatePolygonResult, AggregatePolygonSpatialResult, DriverVectorCube]:

        # TODO: support more advanced reducers too
        assert isinstance(reducer, dict) and len(reducer) == 1
        reducer = next(iter(reducer.values()))["process_id"]
        assert reducer == 'mean' or reducer == 'avg'

        def assert_polygon_sequence(geometries: Union[Sequence, BaseMultipartGeometry]) -> int:
            n_geometries = len(geometries)

            assert n_geometries > 0
            for g in geometries:
                assert isinstance(g, Polygon) or isinstance(g, MultiPolygon)

            return n_geometries

        # TODO #114 EP-3981 normalize to vector cube and preserve original properties
        if isinstance(geometries, DriverVectorCube):
            # Build dummy aggregation data cube
            dims, coords = geometries.get_xarray_cube_basics()
            if self.metadata.has_temporal_dimension():
                dims += (self.metadata.temporal_dimension.name,)
                coords[self.metadata.temporal_dimension.name] = ["2015-07-06T00:00:00Z", "2015-08-22T00:00:00Z"]
            if self.metadata.has_band_dimension():
                dims += (self.metadata.band_dimension.name,)
                coords[self.metadata.band_dimension.name] = self.metadata.band_names
            shape = [len(coords[d]) for d in dims]
            data = numpy.arange(numpy.prod(shape), dtype="float")
            # Start with some more interesting values (e.g. to test NaN/null/None handling)
            data[0] = 2.345
            data[1] = float("nan")
            cube = xarray.DataArray(
                data=data.reshape(shape),
                dims=dims,
                coords=coords,
                name="aggregate_spatial",
            )
            return geometries.with_cube(cube=cube)
        elif isinstance(geometries, str):
            geometries = [geometry for geometry in DelayedVector(geometries).geometries]
            n_geometries = assert_polygon_sequence(geometries)
        elif isinstance(geometries, GeometryCollection):
            # TODO #71 #114 EP-3981: GeometryCollection is deprecated
            n_geometries = assert_polygon_sequence(geometries)
        elif isinstance(geometries, BaseGeometry):
            n_geometries = assert_polygon_sequence([geometries])
        else:
            n_geometries = assert_polygon_sequence(geometries)

        if self.metadata.has_temporal_dimension():
            return AggregatePolygonResult(timeseries={
                "2015-07-06T00:00:00Z": [[2.345]] * n_geometries,
                "2015-08-22T00:00:00Z": [[float('nan')]] * n_geometries
            }, regions=geometries)
        else:
            return DummyAggregatePolygonSpatialResult(cube=self, geometries=geometries)


class DummyAggregatePolygonSpatialResult(AggregatePolygonSpatialResult):
    # TODO #114 EP-3981 replace with proper VectorCube implementation

    def __init__(self, cube: DummyDataCube, geometries: Iterable[BaseGeometry]):
        super().__init__(csv_dir="/dev/null", regions=geometries)
        bands = len(cube.metadata.bands)
        # Dummy data: #geometries rows x #bands columns
        self.data = [[100 + g + 0.1 * b for b in range(bands)] for g in range(len(self._regions))]

    def prepare_for_json(self):
        return self.data

    def fit_class_random_forest(
            self,
            target: dict,
            num_trees: int = 100,
            max_variables: Optional[Union[int, str]] = None,
            seed: Optional[int] = None
    ) -> DriverMlModel:
        # Fake ML training: just store inputs
        return DummyMlModel(
            process_id="fit_class_random_forest",
            data=self.data, target=target, num_trees=num_trees, max_variables=max_variables, seed=seed,
        )


class DummyVectorCube(DriverVectorCube):
    def fit_class_random_forest(
        self,
        target: DriverVectorCube,
        num_trees: int = 100,
        max_variables: Optional[Union[int, str]] = None,
        seed: Optional[int] = None,
    ) -> "DriverMlModel":
        return DummyMlModel(
            process_id="fit_class_random_forest",
            # TODO: handle `to_geojson` in `DummyMlModel.write_assets` instead of here?
            data=self.to_geojson(),
            target=target,
            num_trees=num_trees,
            max_variables=max_variables,
            seed=seed,
        )


class DummyMlModel(DriverMlModel):

    def __init__(self, **kwargs):
        self.creation_data = kwargs

    def write_assets(self, directory: Union[str, Path], options: Optional[dict] = None) -> Dict[str, StacAsset]:
        path = (Path(directory) / "mlmodel.json")
        with path.open("w") as f:
            json.dump({"type": type(self).__name__, "creation_data": self.creation_data}, f)
        return {path.name: {"href": str(path), "path": str(path)}}


class DummyCatalog(CollectionCatalog):
    _COLLECTIONS = [
        {
            'id': 'S2_FAPAR_CLOUDCOVER',
            'product_id': 'S2_FAPAR_CLOUDCOVER',
            'name': 'S2_FAPAR_CLOUDCOVER',
            'description': 'fraction of the solar radiation absorbed by live leaves for the photosynthesis activity',
            'license': 'free',
            "extent": {
                "spatial": {
                    "bbox": [
                        [-180, -90, 180, 90]
                    ]
                },
                "temporal": {
                    "interval": [
                        ["2019-01-02", "2019-02-03"]
                    ]
                }
            },
            'cube:dimensions': {
                "x": {"type": "spatial", "extent": [-180, 180], "step":10, "reference_system": "AUTO:42001"},
                "y": {"type": "spatial", "extent": [-90, 90], "step":10, "reference_system": "AUTO:42001"},
                "t": {"type": "temporal", "extent": ["2019-01-02", "2019-02-03"]},
            },
            'summaries': {},
            'links': [],
        },
        {
            'id': 'ESA_WORLDCOVER_10M_2020_V1',
            'license': 'free',
            "extent": {
                "spatial": {
                    "bbox": [
                        [2.5, 49.5, 6.2, 51.5]
                    ]
                },
                "temporal": {
                    "interval": [
                        ["2019-01-01", None]
                    ]
                }
            },
            'cube:dimensions': {
                "x": {"type": "spatial", "axis": "x", "extent": [-180, 180], "step": 0.000083333333333,
                      "reference_system": 4326},
                "y": {"type": "spatial", "axis": "y", "extent": [-70, 70], "step": 0.000083333333333,
                      "reference_system": 4326},
                "t": {"type": "temporal", "extent": ["2019-01-01", None]},
                "bands": {"type": "bands", "values": ["MAP"]},
            },
            'summaries': {
                "eo:bands": [
                    {"name": "MAP", "common_name": "blue"}
                ]
            },
            'links': [],
            '_private': {'password': 'dragon'},
            '_vito': {"data_source": {"realign": True}}
        },
        {
            'id': 'SENTINEL1_GRD',
            'license': 'free',
            "extent": {
                "spatial": {
                    "bbox": [
                        [2.5, 49.5, 6.2, 51.5]
                    ]
                },
                "temporal": {
                    "interval": [
                        ["2019-01-01", None]
                    ]
                }
            },
            'cube:dimensions': {
                "x": {"type": "spatial", "extent": [2.5, 6.2], "step": 10, "reference_system": "AUTO:42001"},
                "y": {"type": "spatial", "extent": [49.5, 51.5], "step": 10, "reference_system": "AUTO:42001"},
                "t": {"type": "temporal", "extent": ["2019-01-01", None]},
                "bands": {"type": "bands", "values": ["VV","VH"]},
            },
            'summaries': {
                "eo:bands": [
                    {"name": "VV", "common_name": "blue"},
                    {"name": "VH", "common_name": "blue"}
                ]
            },
            'links': [],
            '_private': {'password': 'dragon'}
        },
        {
            'id': 'SENTINEL2_L2A_SENTINELHUB',
            'license': 'free',
            "extent": {
                "spatial": {
                    "bbox": [
                        [2.5, 49.5, 6.2, 51.5]
                    ]
                },
                "temporal": {
                    "interval": [
                        ["2019-01-01", None]
                    ]
                }
            },
            'cube:dimensions': {
                "x": {"type": "spatial", "extent": [2.5, 6.2], "step": 10, "reference_system": "AUTO:42001"},
                "y": {"type": "spatial", "extent": [49.5, 51.5], "step": 10, "reference_system": "AUTO:42001"},
                "t": {"type": "temporal", "extent": ["2019-01-01", None]},
                "bands": {"type": "bands", "values": ["B02", "B03", "B04", "B08", "CLP", "SCL","sunAzimuthAngles","sunZenithAngles"]},
            },
            'summaries': {
                "eo:bands": [
                    {"name": "B02", "common_name": "blue"},
                    {"name": "B03", "common_name": "green"},
                    {"name": "B04", "common_name": "red"},
                    {"name": "B08", "common_name": "nir"},
                    {"name": "CLP", "common_name": "nir"},
                    {"name": "SCL", "common_name": "nir"},
                    {"name": "sunAzimuthAngles", "common_name": "nir"},
                    {"name": "sunZenithAngles", "common_name": "nir"},
                ]
            },
            'links': [],
            '_private': {'password': 'dragon'}
        },
        {
            'id': 'S2_FOOBAR',
            'license': 'free',
            "extent": {
                "spatial": {
                    "bbox": [
                        [2.5, 49.5, 6.2, 51.5]
                    ]
                },
                "temporal": {
                    "interval": [
                        ["2019-01-01", None]
                    ]
                }
            },
            'cube:dimensions': {
                "x": {"type": "spatial", "extent": [2.5, 6.2], "step":10, "reference_system": "AUTO:42001"},
                "y": {"type": "spatial", "extent": [49.5, 51.5], "step":10, "reference_system": "AUTO:42001"},
                "t": {"type": "temporal", "extent": ["2019-01-01", None]},
                "bands": {"type": "bands", "values": ["B02", "B03", "B04", "B08"]},
            },
            'summaries': {
                "eo:bands": [
                    {"name": "B02", "common_name": "blue"},
                    {"name": "B03", "common_name": "green"},
                    {"name": "B04", "common_name": "red"},
                    {"name": "B08", "common_name": "nir"},
                ]
            },
            'links': [],
            '_private': {'password': 'dragon'}
        },
        {
            'id': 'PROBAV_L3_S10_TOC_NDVI_333M_V2',
            'cube:dimensions': {
                "x": {"type": "spatial", "extent": [-180, 180],"step": 0.002976190476190, "reference_system": 4326},
                "y": {"type": "spatial", "extent": [-56, 83],"step": 0.002976190476190, "reference_system": 4326},
                "t": {"type": "temporal", "extent": ["2014-01-01", None]},
                "bands": {"type": "bands", "values": ["ndvi"]}
            },

        },
        {
            "id": "TERRASCOPE_S2_FAPAR_V2",
            "extent": {
                "spatial": {"bbox": [[-180, -56, 180, 83]]},
                "temporal": {"interval": [["2015-07-06", None]]}
            },
            "cube:dimensions": {
                "x": {"type": "spatial", "axis": "x"},
                "y": {"type": "spatial", "axis": "y"},
                "t": {"type": "temporal"},
                "bands": {
                    "type": "bands",
                    "values": ["FAPAR_10M", "SCENECLASSIFICATION_20M"]
                }
            },
            "summaries": {
                "eo:bands": [
                    {"name": "FAPAR_10M"},
                    {"name": "SCENECLASSIFICATION_20M"}
                ]
            },
            "_vito": {
                "properties": {
                    "resolution": {
                        "process_graph": {
                            "res": {
                                "process_id": "eq",
                                "arguments": {
                                    "x": {
                                        "from_parameter": "value"
                                    },
                                    "y": "10"
                                },
                                "result": True
                            }
                        }
                    }
                }
            }
        },
        {
            "id": "S2_NO_SYNC_PROCESSING",
        },
        {
            "id": "S2_FAIL_VERIFY_FOR_SYNC_PROCESSING",
        },
    ]

    def __init__(self):
        super().__init__(all_metadata=self._COLLECTIONS)

    def load_collection(self, collection_id: str, load_params: LoadParameters, env: EvalEnv) -> DummyDataCube:
        return self._load_collection_cached(collection_id,load_params,WhiteListEvalEnv(env,[]))

    @lru_cache(maxsize=20)
    def _load_collection_cached(self, collection_id: str, load_params: LoadParameters, env:EvalEnv) -> DummyDataCube:
        _register_load_collection_call(collection_id, load_params)

        metadata = CollectionMetadata(metadata=self.get_collection_metadata(collection_id))
        if load_params.bands:
            metadata = metadata.filter_bands(load_params.bands)
        image_collection = DummyDataCube(metadata=metadata)

        _collections[collection_id] = image_collection
        return image_collection

    def patch_collections(self, extra_collections: dict):
        """Context manager to (temporarily) add a collection with metadata to the catalog."""
        return unittest.mock.patch.dict(self._catalog, extra_collections)


class DummyProcessing(ConcreteProcessing):
    def extra_validation(
            self, process_graph: dict, env: EvalEnv, result, source_constraints: List[SourceConstraint]
    ) -> Iterable[dict]:
        # Fake missing tiles
        for source_id, constraints in source_constraints:
            if source_id[0] == "load_collection" and source_id[1][0] == "S2_FOOBAR":
                dates = constraints.get("temporal_extent")
                bbox = constraints.get("spatial_extent")
                if dates and dates[0] <= "2021-02-10" and bbox and bbox["west"] <= 1.4:
                    yield {"code": "MissingProduct", "message": "Tile 4322 not available"}

    def run_udf(self, udf: str, data: openeo.udf.UdfData) -> openeo.udf.UdfData:
        return openeo.udf.run_udf_code(code=udf, data=data)

    def verify_for_synchronous_processing(self, process_graph: dict, env: EvalEnv = None) -> Iterable[str]:
        collections = [p["arguments"]["id"] for p in process_graph.values() if p["process_id"] == "load_collection"]
        for cid in collections:
            if "NO_SYNC_PROCESSING" in cid:
                yield f"Collection {cid!r} is not available for synchronous processing."
            if "FAIL_VERIFY_FOR_SYNC_PROCESSING" in cid:
                # For testing that things keep working when verifying goes wrong
                raise RuntimeError("Nope, catch this")


class DummyBatchJobs(BatchJobs):
    _job_registry: Dict[Tuple[str, str], BatchJobMetadata] = {}
    _custom_job_logs = {}
    _job_result_registry: Dict[Tuple[str, str], BatchJobResultMetadata] = {}

    def generate_job_id(self):
        return generate_unique_id(prefix="j")

    def create_job(
        self,
        user_id: str,
        user: User,
        process: dict,
        api_version: str,
        metadata: dict,
        job_options: Optional[dict] = None,
    ) -> BatchJobMetadata:
        job_id = self.generate_job_id()
        job_info = BatchJobMetadata(
            id=job_id,
            status=JOB_STATUS.CREATED,
            process=process,
            created=utcnow(),
            job_options=job_options,
            title=metadata.get("title"),
            description=metadata.get("description"),
        )
        self._job_registry[(user_id, job_id)] = job_info
        return job_info

    def get_job_info(self, job_id: str, user_id: str) -> BatchJobMetadata:
        return self._get_job_info(job_id=job_id, user_id=user_id)

    def _get_job_info(self, job_id: str, user_id: str) -> BatchJobMetadata:
        try:
            return self._job_registry[(user_id, job_id)]
        except KeyError:
            raise JobNotFoundException(job_id)

    def get_user_jobs(self, user_id: str) -> List[BatchJobMetadata]:
        return [v for (k, v) in self._job_registry.items() if k[0] == user_id]

    @classmethod
    def _update_status(cls, job_id: str, user_id: str, status: str):
        try:
            cls._job_registry[(user_id, job_id)] = cls._job_registry[(user_id, job_id)]._replace(status=status)
        except KeyError:
            raise JobNotFoundException(job_id)

    @classmethod
    def set_result_metadata(
        cls, job_id: str, user_id: str, metadata: BatchJobResultMetadata
    ):
        cls._job_result_registry[(job_id, user_id)] = metadata

    def start_job(self, job_id: str, user: User):
        self._update_status(
            job_id=job_id, user_id=user.user_id, status=JOB_STATUS.RUNNING
        )

    def _output_root(self) -> str:
        return "/data/jobs"

    def get_result_metadata(self, job_id: str, user_id: str) -> BatchJobResultMetadata:
        if (job_id, user_id) in self._job_result_registry:
            return self._job_result_registry[(job_id, user_id)]
        else:
            result_md = super().get_result_metadata(job_id=job_id, user_id=user_id)
            result_md = BatchJobResultMetadata(
                assets=result_md.assets,
                links=result_md.links,
                providers=self._get_providers(job_id=job_id, user_id=user_id),
            )
            return result_md

    def get_result_assets(self, job_id: str, user_id: str) -> Dict[str, dict]:
        if (
            self._get_job_info(job_id=job_id, user_id=user_id).status
            != JOB_STATUS.FINISHED
        ):
            raise JobNotFinishedException
        return {
            "output.tiff": {
                "asset": True,
                "output_dir": f"{self._output_root()}/{job_id}",
                "href": f"{self._output_root()}/{job_id}/output.tiff",
                "type": "image/tiff; application=geotiff",
                "roles": ["data"],
                "bands": [Band(name="NDVI", common_name="NDVI", wavelength_um=1.23)],
                "nodata": 123,
                "instruments": "MSI"
            },
            "randomforest.model": {
                "asset": True,
                "href": str(Path(job_id) / "randomforest.model"),
            },
            DriverMlModel.METADATA_FILE_NAME: {
                "ml_model_metadata": True,
                "asset": False,
                "stac_version": "1.0.0",
                "stac_extensions": [
                    STAC_EXTENSION.MLMODEL,
                ],
                "type": "Feature",
                "id": generate_unique_id(prefix="ml"),
                "collection": "collection-id",
                "bbox": [
                    -179.999,
                    -89.999,
                    179.999,
                    89.999
                ],
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [
                                -179.999,
                                -89.999
                            ],
                            [
                                179.999,
                                -89.999
                            ],
                            [
                                179.999,
                                89.999
                            ],
                            [
                                -179.999,
                                89.999
                            ],
                            [
                                -179.999,
                                -89.999
                            ]
                        ]
                    ]
                },
                'properties': {
                    "datetime": None,
                    "start_datetime": "1970-01-01T00:00:00Z",
                    "end_datetime": "9999-12-31T23:59:59Z",
                    "ml-model:type": "ml-model",
                    "ml-model:learning_approach": "supervised",
                    "ml-model:prediction_type": "classification",
                    "ml-model:architecture": "random-forest",
                    "ml-model:training-processor-type": "cpu",
                    "ml-model:training-os": "linux",
                },
                'links': [],
                'assets': {
                    'model': {
                        "href": str(Path(job_id) / "randomforest.model"),
                        "type": "application/octet-stream",
                        "title": "org.apache.spark.mllib.tree.model.RandomForestModel",
                        "roles": ["ml-model:checkpoint"]
                    }
                }
            }
        }

    def get_log_entries(
        self,
        job_id: str,
        user_id: str,
        offset: Optional[str] = None,
        level: Optional[str] = None,
    ) -> Iterable[dict]:
        self._get_job_info(job_id=job_id, user_id=user_id)
        default_logs = [{"id": "1", "level": "info", "message": "hello world"}]
        requested_level = normalize_log_level(level)
        for log in self._custom_job_logs.get(job_id, default_logs):
            if isinstance(log, dict):
                actual_level = normalize_log_level(log.get("log_level"))
                if actual_level < requested_level:
                    continue
                yield log
            elif isinstance(log, Exception):
                raise log
            else:
                raise ValueError(log)

    def cancel_job(self, job_id: str, user_id: str):
        self._get_job_info(job_id=job_id, user_id=user_id)

    def delete_job(self, job_id: str, user_id: str):
        self.cancel_job(job_id, user_id)


class DummyUserDefinedProcesses(UserDefinedProcesses):
    def __init__(self):
        super().__init__()
        self._processes: Dict[Tuple[str, str], UserDefinedProcessMetadata] = {}

    def reset(self, db: Dict[Tuple[str, str], UserDefinedProcessMetadata]):
        self._processes = db

    def get(self, user_id: str, process_id: str) -> Union[UserDefinedProcessMetadata, None]:
        return self._processes.get((user_id, process_id))

    def get_for_user(self, user_id: str) -> List[UserDefinedProcessMetadata]:
        return [udp for key, udp in self._processes.items() if key[0] == user_id]

    def save(self, user_id: str, process_id: str, spec: dict) -> None:
        self._processes[user_id, process_id] = UserDefinedProcessMetadata.from_dict(spec)

    def delete(self, user_id: str, process_id: str) -> None:
        try:
            self._processes.pop((user_id, process_id))
        except KeyError:
            raise ProcessGraphNotFoundException(process_id)


class DummyBackendImplementation(OpenEoBackendImplementation):

    vector_cube_cls = DummyVectorCube

    def __init__(
        self,
        processing: Optional[Processing] = None,
        config: Optional[OpenEoBackendConfig] = None,
    ):
        super(DummyBackendImplementation, self).__init__(
            secondary_services=DummySecondaryServices(),
            catalog=DummyCatalog(),
            batch_jobs=DummyBatchJobs(),
            user_defined_processes=DummyUserDefinedProcesses(),
            processing=processing or DummyProcessing(),
            config=config,
        )

    def file_formats(self) -> dict:
        return {
            "input": {
                "GeoJSON": {
                    "gis_data_types": ["vector"],
                    "parameters": {},
                },
                "ESRI Shapefile": {
                    "gis_data_types": ["vector"],
                    "parameters": {},
                },
                "GPKG": {
                    "title": "OGC GeoPackage",
                    "gis_data_types": ["raster", "vector"],
                    "parameters": {},
                },
            },
            "output": {
                "GTiff": {
                    "title": "GeoTiff",
                    "gis_data_types": ["raster"],
                    "parameters": {},
                },
            },
        }

    def load_disk_data(
            self, format: str, glob_pattern: str, options: dict, load_params: LoadParameters, env: EvalEnv
    ) -> DummyDataCube:
        _register_load_collection_call(glob_pattern, load_params)
        metadata = CollectionMetadata(
            {},
            dimensions=[
                SpatialDimension(name="x", extent=[]),
                SpatialDimension(name="y", extent=[]),
                TemporalDimension(name="t", extent=[]),
                BandDimension(name="bands", bands=[Band("unknown")]),
            ],
        )
        return DummyDataCube(metadata=metadata)

    def load_result(self, job_id: str, user_id: Optional[str], load_params: LoadParameters,
                    env: EvalEnv) -> DummyDataCube:
        _register_load_collection_call(job_id, load_params)
        return DummyDataCube()

    def visit_process_graph(self, process_graph: dict) -> ProcessGraphVisitor:
        return DummyVisitor().accept_process_graph(process_graph)

    def user_access_validation(self, user: User, request: flask.Request) -> User:
        if "mark" in user.user_id.lower():
            raise PermissionsInsufficientException(message="No access for Mark.")
        if user.user_id == "Alice":
            user.set_default_plan("alice-plan")
        if user.user_id == "Carol":
            user.add_roles(["admin", "devops"])
        return user
