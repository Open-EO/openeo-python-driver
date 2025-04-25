import datetime
import urllib.parse
from unittest import mock

import pytest
from openeo.utils.version import ComparableVersion

from openeo_driver.backend import (
    BatchJobMetadata,
    CollectionCatalog,
    JobListing,
    LoadParameters,
    OpenEoBackendImplementation,
    ServiceMetadata,
    UserDefinedProcessMetadata,
    is_not_implemented,
    not_implemented,
)
from openeo_driver.errors import CollectionNotFoundException
from openeo_driver.users import User


def test_collection_catalog_basic():
    catalog = CollectionCatalog([{"id": "Sentinel2", "flavor": "salty"}, {"id": "NDVI", "flavor": "smurf"}])
    all_metadata = catalog.get_all_metadata()
    assert len(all_metadata) == 2
    assert set(c["id"] for c in all_metadata) == {"Sentinel2", "NDVI"}
    assert catalog.get_collection_metadata("Sentinel2") == {"id": "Sentinel2", "flavor": "salty"}
    assert catalog.get_collection_metadata("NDVI") == {"id": "NDVI", "flavor": "smurf"}


def test_collection_catalog_invalid_id(caplog):
    catalog = CollectionCatalog([{"id": "Sentinel2", "flavor": "salty"}, {"id": "NDVI", "flavor": "smurf"}])
    with pytest.raises(CollectionNotFoundException):
        catalog.get_collection_metadata("nope")


def test_load_parameters():
    params = LoadParameters(temporal_extent=("2021-01-01", None))
    assert params.temporal_extent == ("2021-01-01", None)
    assert params.spatial_extent == {}
    assert params.bands is None
    assert params.sar_backscatter is None

    params_copy = params.copy()
    assert isinstance(params_copy, LoadParameters)
    assert params_copy.temporal_extent == ("2021-01-01", None)

    params.bands = ["red", "green"]
    assert params.bands == ["red", "green"]
    assert params_copy.bands is None


class TestUserDefinedProcessMetadata:
    def test_basic(self):
        udp = UserDefinedProcessMetadata(id="enhance", process_graph={"foo": {"process_id": "foo"}})
        assert udp.prepare_for_json() == {
            "id": "enhance",
            "process_graph": {"foo": {"process_id": "foo"}},
            "parameters": None,
            "returns": None,
            "summary": None,
            "description": None,
            "links": None,
            "public": False,
        }

    def test_from_dict_minimal(self):
        udp = UserDefinedProcessMetadata.from_dict({"id": "enhance", "process_graph": {"foo": {"process_id": "foo"}}})
        assert udp.id == "enhance"
        assert udp.process_graph == {"foo": {"process_id": "foo"}}
        assert udp.parameters is None


    def test_from_dict_no_id(self):
        with pytest.raises(KeyError):
            _ = UserDefinedProcessMetadata.from_dict({"process_graph": {"foo": {"process_id": "foo"}}})


    def test_from_dict_extra(self):
        udp = UserDefinedProcessMetadata.from_dict(
            {
                "id": "enhance",
                "process_graph": {"foo": {"process_id": "foo"}},
                "parameters": [],
                "returns": {"schema": {"type": "number"}},
                "summary": "Enhance it!",
                "description": "Enhance the image with the foo process.",
            }
        )
        assert udp.id == "enhance"
        assert udp.process_graph == {"foo": {"process_id": "foo"}}
        assert udp.parameters == []
        assert udp.returns == {"schema": {"type": "number"}}
        assert udp.summary == "Enhance it!"
        assert udp.description == "Enhance the image with the foo process."

    @pytest.mark.parametrize(
        ["initial_links", "expected_links"],
        [
            (None, [{"rel": "doc", "href": "https://doc.test/enhance"}]),
            ([], [{"rel": "doc", "href": "https://doc.test/enhance"}]),
            (
                [{"rel": "self", "href": "https://oeo.test/udp/enhance"}],
                [
                    {"rel": "self", "href": "https://oeo.test/udp/enhance"},
                    {"rel": "doc", "href": "https://doc.test/enhance"},
                ],
            ),
        ],
    )
    def test_add_link(self, initial_links, expected_links):
        udp1 = UserDefinedProcessMetadata(
            id="enhance", process_graph={"foo": {"process_id": "foo"}}, links=initial_links
        )

        udp2 = udp1.add_link(rel="doc", href="https://doc.test/enhance")
        assert udp2 is not udp1

        assert udp1.links == initial_links
        assert udp2.links == expected_links


def test_service_metadata_from_dict_essentials():
    service = ServiceMetadata.from_dict(
        {"id": "badcafe", "url": "https://oeo.test/srv/f00b67", "type": "WMTS"}
    )
    assert service.id == "badcafe"
    assert service.url == "https://oeo.test/srv/f00b67"
    assert service.type == "WMTS"
    assert service.enabled is True
    assert service.process is None
    assert service.attributes is None
    assert service.configuration is None

def test_service_metadata_from_dict_basic():
    service = ServiceMetadata.from_dict({
        "id": "badcafe", "process": {"id": "ndvi", "process_graph": {}},
        "url": "https://oeo.test/srv/f00b67",
        "type": "WMTS", "enabled": True,
        "configuration": {}, "attributes": {},
        "flavor": "strawberry",
    })
    assert service.id == "badcafe"
    assert service.process == {"id": "ndvi", "process_graph": {}}
    assert service.url == "https://oeo.test/srv/f00b67"
    assert service.type == "WMTS"
    assert service.enabled is True


def test_service_metadata_from_dict_created_date():
    service = ServiceMetadata.from_dict({
        "id": "badcafe", "process": {"id": "ndvi", "process_graph": {}},
        "url": "https://oeo.test/srv/f00b67",
        "type": "WMTS", "enabled": True,
        "configuration": {}, "attributes": {},
        "created": "2020-05-18T12:34:56Z",
    })
    assert service.created == datetime.datetime(2020, 5, 18, 12, 34, 56)


def test_batch_job_metadata_from_api_dict_emtpy():
    with pytest.raises(KeyError, match="Missing BatchJobMetadata fields: created, id, status"):
        _ = BatchJobMetadata.from_api_dict({})


def test_batch_job_metadata_from_api_dict_basic():
    job = BatchJobMetadata.from_api_dict({
        "id": "ba7c470b", "created": "2021-06-18T12:34:56Z", "status": "running",
    })
    assert job.id == "ba7c470b"
    assert job.created == datetime.datetime(2021, 6, 18, 12, 34, 56)
    assert job.status == "running"

    # Full round trip check
    assert job == BatchJobMetadata.from_api_dict(job.to_api_dict())


def test_batch_job_metadata_from_api_dict_auto_conversions():
    job = BatchJobMetadata.from_api_dict({
        "id": "ba7c470b",
        "status": "running",
        "created": "2021-06-18T12:34:56Z",
        "updated": "2021-06-20T20:20:20Z",
        "usage": {"memory": {"value": 2000, "unit": "mb-seconds"}},
    })
    assert job.created == datetime.datetime(2021, 6, 18, 12, 34, 56)
    assert job.updated == datetime.datetime(2021, 6, 20, 20, 20, 20)
    assert job.memory_time_megabyte == datetime.timedelta(seconds=2000)  # not actually auto-converted

    # Full round trip check
    assert job == BatchJobMetadata.from_api_dict(job.to_api_dict())


def test_batch_job_metadata_from_api_dict_usage():
    job = BatchJobMetadata.from_api_dict({
        "id": "ba7c470b", "created": "2021-06-18T12:34:56Z", "status": "running",
        "usage": {
            "cpu": {"value": 1000, "unit": "cpu-seconds"},
            "memory": {"value": 2000, "unit": "mb-seconds"},
            "duration": {"value": 3000, "unit": "seconds"},
        }
    })
    assert job.id == "ba7c470b"
    assert job.created == datetime.datetime(2021, 6, 18, 12, 34, 56)
    assert job.status == "running"
    assert job.cpu_time == datetime.timedelta(seconds=1000)
    assert job.memory_time_megabyte == datetime.timedelta(seconds=2000)
    assert job.duration == datetime.timedelta(seconds=3000)
    assert job.duration_ == datetime.timedelta(seconds=3000)

    # Full round trip check
    assert job == BatchJobMetadata.from_api_dict(job.to_api_dict())


def test_batch_job_metadata_to_api_dict():
    api_version = ComparableVersion("1.0.0")
    job = BatchJobMetadata(
        id="123", status="running", created=datetime.datetime(2022, 1, 18, 16, 42, 0),
        process={"add": {"process_id": "add", "arguments": {"x": 3, "y": 5}, "result": True}},
        title="Untitled01", description="Lorem ipsum.",
        progress=0.3,
        cpu_time=datetime.timedelta(seconds=1000),
        memory_time_megabyte=datetime.timedelta(seconds=2000),
        started=datetime.datetime(2022, 1, 18, 17, 0, 0),
        finished=datetime.datetime(2022, 1, 18, 17, 20, 0),
        epsg=4326,
        links=[{}],
    )

    assert job.to_api_dict(full=False, api_version=api_version) == {
        "id": "123",
        "title": "Untitled01", "description": "Lorem ipsum.",
        "status": "running",
        "progress": 0.3,
        "created": "2022-01-18T16:42:00Z",
    }
    assert job.to_api_dict(full=True, api_version=api_version) == {
        "id": "123",
        "title": "Untitled01", "description": "Lorem ipsum.",
        "process": {"add": {"process_id": "add", "arguments": {"x": 3, "y": 5}, "result": True}},
        "status": "running",
        "progress": 0.3,
        "created": "2022-01-18T16:42:00Z",
        "usage": {
            "cpu": {"value": 1000, "unit": "cpu-seconds"},
            "memory": {"value": 2000, "unit": "mb-seconds"},
            "duration": {"value": 1200, "unit": "seconds"},
        }
    }


def test_not_implemented():
    def foo(x):
        return x

    @not_implemented
    def bar(x):
        ...

    meh = None

    assert is_not_implemented(foo) is False
    assert is_not_implemented(bar) is True
    assert is_not_implemented(meh) is True




def test_request_costs_user():
    backend = OpenEoBackendImplementation()
    assert backend.request_costs(user=User(user_id="someuser"), request_id="r-abc123", success=True) is None


class TestJobListing:
    def _build_url(self, params: dict):
        return "https://oeo.test/jobs?" + urllib.parse.urlencode(query=params)

    def test_basic(self):
        listing = JobListing(
            jobs=[
                BatchJobMetadata(id="j-123", status="created", created=datetime.datetime(2024, 11, 22)),
                BatchJobMetadata(id="j-456", status="running", created=datetime.datetime(2024, 12, 6)),
            ]
        )
        assert len(listing) == 2
        assert listing.to_response_dict(build_url=self._build_url) == {
            "jobs": [
                {"id": "j-123", "progress": 0, "status": "created", "created": "2024-11-22T00:00:00Z"},
                {"id": "j-456", "status": "running", "created": "2024-12-06T00:00:00Z"},
            ],
            "links": [],
        }

    def test_next_parameters(self):
        listing = JobListing(
            jobs=[
                BatchJobMetadata(id="j-123", status="created", created=datetime.datetime(2024, 11, 22)),
                BatchJobMetadata(id="j-456", status="running", created=datetime.datetime(2024, 12, 6)),
            ],
            next_parameters={"offset": 1234, "state": "foo"},
        )
        assert len(listing) == 2
        assert listing.to_response_dict(build_url=self._build_url) == {
            "jobs": [
                {"id": "j-123", "progress": 0, "status": "created", "created": "2024-11-22T00:00:00Z"},
                {"id": "j-456", "status": "running", "created": "2024-12-06T00:00:00Z"},
            ],
            "links": [{"href": "https://oeo.test/jobs?offset=1234&state=foo", "rel": "next"}],
        }
