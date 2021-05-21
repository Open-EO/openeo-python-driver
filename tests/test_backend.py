import pytest

from openeo_driver.backend import CollectionCatalog, LoadParameters, UserDefinedProcessMetadata
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
