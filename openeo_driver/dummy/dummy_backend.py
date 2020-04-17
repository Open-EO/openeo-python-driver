from datetime import datetime
import numbers
import os
from pathlib import Path
from typing import List, Dict
from unittest.mock import Mock

from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry.collection import GeometryCollection

from openeo import ImageCollection
from openeo_driver.delayed_vector import DelayedVector
from openeo_driver.backend import SecondaryServices, OpenEoBackendImplementation, CollectionCatalog, ServiceMetadata, BatchJobs, BatchJobMetadata
from openeo_driver.errors import JobNotFoundException
from openeo.internal.process_graph_visitor import ProcessGraphVisitor
from openeo_driver.testing import TEST_USER

TEST_BATCH_JOB_ID = '07024ee9-7847-4b8a-b260-6c879a2b3cdc'

collections = {}


class DummyVisitor(ProcessGraphVisitor):

    def __init__(self):
        super(DummyVisitor, self).__init__()
        self.processes = []

    def enterProcess(self, process_id: str, arguments: dict):
        self.processes.append((process_id, arguments))

    def constantArgument(self, argument_id: str, value):
        if isinstance(value, numbers.Real):
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
            attributes={},
            title="Test service",
            created=datetime(2020, 4, 9, 15, 5, 8)
        )
    ]

    def service_types(self) -> dict:
        return {
            "WMTS": {
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
                "process_parameters": [],
                "links": [],
            }
        }

    def list_services(self) -> List[ServiceMetadata]:
        return self._registry

    def service_info(self, service_id: str) -> ServiceMetadata:
        return next(s for s in self._registry if s.id == service_id)


class DummyCatalog(CollectionCatalog):
    _COLLECTIONS = [{
        'product_id': 'DUMMY_S2_FAPAR_CLOUDCOVER',
        'name': 'DUMMY_S2_FAPAR_CLOUDCOVER',
        'id': 'DUMMY_S2_FAPAR_CLOUDCOVER',
        'description': 'fraction of the solar radiation absorbed by live leaves for the photosynthesis activity',
        'license': 'free',
        'extent': {
            'spatial': [-180, -90, 180, 90],
            'temporal': ["2019-01-02", "2019-02-03"],
        },
        'links': [],
        'stac_version': '0.1.2',
        'properties': {'cube:dimensions': {}},
        'other_properties': {},
    }]

    def __init__(self):
        super().__init__(all_metadata=self._COLLECTIONS)

    def load_collection(self, collection_id: str, viewing_parameters: dict) -> ImageCollection:
        if collection_id in collections:
            return collections[collection_id]

        # TODO simplify all this mocki/return_value stuff?
        image_collection = ImageCollection()
        image_collection.viewingParameters = viewing_parameters

        image_collection.mask = Mock(name="mask")
        image_collection.mask.return_value = image_collection

        image_collection.mask_polygon = Mock(name="mask_polygon")
        image_collection.mask_polygon.return_value = image_collection

        image_collection.bbox_filter = Mock(name="bbox_filter")
        image_collection.bbox_filter.return_value = image_collection

        image_collection.tiled_viewing_service = Mock(name="tiled_viewing_service")
        image_collection.tiled_viewing_service.return_value = ServiceMetadata(
            id='c63d6c27-c4c2-4160-b7bd-9e32f582daec',
            process={"process_graph": {"foo": {"process_id": "foo", "arguments": {}}}},
            url="http://openeo.vgt.vito.be/openeo/services/c63d6c27-c4c2-4160-b7bd-9e32f582daec/service/wmts",
            type="WMTS",
            enabled=True,
            attributes={},
        )

        download = Mock(name='download')
        # TODO: download something more real to allow higher quality testing
        download.return_value = os.path.realpath(__file__)

        image_collection.download = download

        timeseries = Mock(name='timeseries')
        timeseries.return_value = {
            "viewingParameters": image_collection.viewingParameters
        }

        image_collection.timeseries = timeseries

        def is_one_or_more_polygons(return_value, regions, func):
            assert func == 'mean' or func == 'avg'

            def assert_polygon_or_multipolygon(geometry):
                assert isinstance(geometry, Polygon) or isinstance(geometry, MultiPolygon)

            if isinstance(regions, str):
                geometries = [geometry for geometry in DelayedVector(regions).geometries]

                assert len(geometries) > 0
                for geometry in geometries:
                    assert_polygon_or_multipolygon(geometry)
            elif isinstance(regions, GeometryCollection):
                assert len(regions) > 0
                for geometry in regions:
                    assert_polygon_or_multipolygon(geometry)
            else:
                assert_polygon_or_multipolygon(regions)

            return return_value

        zonal_statistics = Mock(name='zonal_statistics')
        zonal_statistics.side_effect = lambda regions, func: is_one_or_more_polygons({
            "2015-07-06T00:00:00": [2.9829132080078127],
            "2015-08-22T00:00:00": [float('nan')]
        }, regions, func)

        image_collection.zonal_statistics = zonal_statistics

        image_collection.apply_pixel = Mock(name="apply_pixel")
        image_collection.apply_pixel.return_value = image_collection

        image_collection.apply_tiles_spatiotemporal = Mock(name="apply_tiles_spatiotemporal")
        image_collection.apply_tiles_spatiotemporal.return_value = image_collection

        image_collection.apply_dimension = Mock(name="apply_dimension")
        image_collection.apply_dimension.return_value = image_collection

        image_collection.apply_tiles = Mock(name="apply_tiles")
        image_collection.apply_tiles.return_value = image_collection

        image_collection.apply = Mock(name="apply")
        image_collection.apply.return_value = image_collection

        image_collection.reduce = Mock(name="reduce")
        image_collection.reduce.return_value = image_collection

        image_collection.reduce_bands = Mock(name="reduce_bands")
        image_collection.reduce_bands.return_value = image_collection

        image_collection.aggregate_temporal = Mock(name="aggregate_temporal")
        image_collection.aggregate_temporal.return_value = image_collection

        image_collection.max_time = Mock(name="max_time")
        image_collection.max_time.return_value = image_collection

        image_collection.apply_kernel = Mock(name="apply_kernel")
        image_collection.apply_kernel.return_value = image_collection

        image_collection.merge = Mock(name="merge")
        image_collection.merge.return_value = image_collection

        collections[collection_id] = image_collection
        return image_collection


class DummyBatchJobs(BatchJobs):

    def create_job(self, user_id: str, job_specification: dict, api_version: str) -> BatchJobMetadata:
        # TODO: actually "create" a new job instead of reusing an existing one?
        return self.get_job_info(job_id=TEST_BATCH_JOB_ID, user_id=user_id)

    def get_job_info(self, job_id: str, user_id: str) -> BatchJobMetadata:
        if (job_id, user_id) == (TEST_BATCH_JOB_ID, TEST_USER):
            return BatchJobMetadata(
                id=job_id,
                status='running',
                process={'process_graph': {'foo': {'process_id': 'foo', 'arguments': {}}}},
                created=datetime(2017, 1, 1, 9, 32, 12),
            )
        raise JobNotFoundException(job_id=job_id)

    def get_user_jobs(self, user_id: str) -> List[BatchJobMetadata]:
        if user_id == TEST_USER:
            return [self.get_job_info(TEST_BATCH_JOB_ID, TEST_USER)]
        else:
            return []

    def start_job(self, job_id: str, user_id: str):
        self.get_job_info(job_id=job_id, user_id=user_id)

    def _output_root(self) -> Path:
        return Path("/data/jobs")

    def get_results(self, job_id: str, user_id: str) -> Dict[str, str]:
        self.get_job_info(job_id=job_id, user_id=user_id)
        return {
            "output.tiff": str(self._output_root() / job_id / "out")
        }

    def get_log_entries(self, job_id: str, user_id: str, offset: str) -> List[dict]:
        self.get_job_info(job_id=job_id, user_id=user_id)
        return [
            {"id": "1", "level": "info", "message": "hello world", "path": []}
        ]

    def cancel_job(self, job_id: str, user_id: str):
        self.get_job_info(job_id=job_id, user_id=user_id)


class DummyBackendImplementation(OpenEoBackendImplementation):
    def __init__(self):
        super(DummyBackendImplementation, self).__init__(
            secondary_services=DummySecondaryServices(),
            catalog=DummyCatalog(),
            batch_jobs=DummyBatchJobs(),
        )

    def file_formats(self) -> dict:
        return {
            "input": {
                "GeoJSON": {
                    "gis_data_type": ["vector"]
                }
            },
            "output": {
                "GTiff": {
                    "title": "GeoTiff",
                    "gis_data_types": ["raster"]
                },
            },
        }

    def load_disk_data(self, format: str, glob_pattern: str, options: dict, viewing_parameters: dict) -> object:
        return {}

    def visit_process_graph(self, process_graph: dict) -> ProcessGraphVisitor:
        return DummyVisitor().accept_process_graph(process_graph)


def get_openeo_backend_implementation() -> OpenEoBackendImplementation:
    return DummyBackendImplementation()
