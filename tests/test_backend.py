import datetime

import pytest

from openeo_driver.backend import CollectionCatalog, LoadParameters, UserDefinedProcessMetadata, ServiceMetadata, \
    BatchJobMetadata, OidcProvider
from openeo_driver.errors import CollectionNotFoundException


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


def test_user_defined_process_metadata():
    udp = UserDefinedProcessMetadata(id="enhance", process_graph={"foo": {"process_id": "foo"}})
    assert udp.prepare_for_json() == {
        "id": "enhance",
        "process_graph": {"foo": {"process_id": "foo"}},
        "parameters": None,
        "returns": None,
        "summary": None,
        "description": None,
        "public": False
    }


def test_user_defined_process_metadata_from_dict_basic():
    udp = UserDefinedProcessMetadata.from_dict({"id": "enhance", "process_graph": {"foo": {"process_id": "foo"}}})
    assert udp.id == "enhance"
    assert udp.process_graph == {"foo": {"process_id": "foo"}}
    assert udp.parameters is None


def test_user_defined_process_metadata_from_dict_extra():
    udp = UserDefinedProcessMetadata.from_dict({
        "id": "enhance",
        "process_graph": {"foo": {"process_id": "foo"}},
        "parameters": [],
        "returns": {"schema": {"type": "number"}},
        "summary": "Enhance it!",
        "description": "Enhance the image with the foo process."
    })
    assert udp.id == "enhance"
    assert udp.process_graph == {"foo": {"process_id": "foo"}}
    assert udp.parameters == []
    assert udp.returns == {"schema": {"type": "number"}}
    assert udp.summary == "Enhance it!"
    assert udp.description == "Enhance the image with the foo process."


def test_user_defined_process_metadata_from_dict_no_id():
    udp = UserDefinedProcessMetadata.from_dict({"process_graph": {"foo": {"process_id": "foo"}}})
    assert udp.id is None
    assert udp.process_graph == {"foo": {"process_id": "foo"}}
    assert udp.parameters is None


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


def test_batch_job_metadata_from_dict_emtpy():
    with pytest.raises(KeyError, match="Missing BatchJobMetadata fields: created, id, process, status"):
        _ = BatchJobMetadata.from_dict({})


def test_batch_job_metadata_from_dict_basic():
    job = BatchJobMetadata.from_dict({
        "id": "ba7c470b", "created": "2021-06-18T12:34:56Z",
        "process": {"id": "ndvi", "process_graph": {}}, "status": "running"
    })
    assert job.id == "ba7c470b"
    assert job.created == datetime.datetime(2021, 6, 18, 12, 34, 56)
    assert job.process == {"id": "ndvi", "process_graph": {}}
    assert job.status == "running"


def test_oidc_provider_from_dict_empty():
    with pytest.raises(KeyError, match="Missing OidcProvider fields: id, issuer, title"):
        _ = OidcProvider.from_dict({})


def test_oidc_provider_from_dict_basic():
    p = OidcProvider.from_dict({"id": "foo", "issuer": "https://oidc.foo.test/", "title": "Foo ID"})
    assert p.id == "foo"
    assert p.issuer == "https://oidc.foo.test/"
    assert p.title == "Foo ID"
    assert p.scopes == ["openid"]
    assert p.description is None

    assert p.prepare_for_json() == {
        "id": "foo", "issuer": "https://oidc.foo.test/", "title": "Foo ID", "scopes": ["openid"]
    }


def test_oidc_provider_from_dict_more():
    p = OidcProvider.from_dict({
        "id": "foo", "issuer": "https://oidc.foo.test/", "title": "Foo ID",
        "scopes": ["openid", "email"],
        "default_clients": {"id": "dcf0e6384", "grant_types": ["refresh_token"]},
    })
    assert p.id == "foo"
    assert p.issuer == "https://oidc.foo.test/"
    assert p.title == "Foo ID"
    assert p.scopes == ["openid", "email"]
    assert p.description is None
    assert p.default_clients == {"id": "dcf0e6384", "grant_types": ["refresh_token"]}

    assert p.prepare_for_json() == {
        "id": "foo", "issuer": "https://oidc.foo.test/", "title": "Foo ID",
        "scopes": ["openid", "email"],
        "default_clients": {"id": "dcf0e6384", "grant_types": ["refresh_token"]},
    }
