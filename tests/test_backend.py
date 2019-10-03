import pytest

from openeo_driver.backend import CollectionCatalog, CollectionIncompleteMetadataWarning
from openeo_driver.errors import CollectionNotFoundException


def test_collection_catalog_basic():
    catalog = CollectionCatalog([{"id": "Sentinel2", "flavor": "salty"}, {"id": "NDVI", "flavor": "smurf"}])
    with pytest.warns(CollectionIncompleteMetadataWarning):
        all_metadata = catalog.get_all_metadata()
    assert len(all_metadata) == 2
    assert set(c["id"] for c in all_metadata) == {"Sentinel2", "NDVI"}
    with pytest.warns(CollectionIncompleteMetadataWarning):
        assert catalog.get_collection_metadata("Sentinel2")["flavor"] == "salty"
        assert catalog.get_collection_metadata("NDVI")["flavor"] == "smurf"
    with pytest.raises(CollectionNotFoundException):
        catalog.get_collection_metadata("nope")


def test_collection_catalog_normalize_metadata():
    catalog = CollectionCatalog([
        {"id": "Sentinel2", "description": "Sentinel 2 data", "license": "free"},
        {"id": "NDVI"}
    ])

    with pytest.warns(CollectionIncompleteMetadataWarning):
        s2 = catalog.get_collection_metadata("Sentinel2")
    assert "stac_version" in s2
    assert s2["links"] == []
    assert s2["other_properties"] == {}
    assert s2["description"] == "Sentinel 2 data"
    assert s2["license"] == "free"
    assert "extent" in s2
    assert "properties" in s2

    with pytest.warns(UserWarning):
        ndvi = catalog.get_collection_metadata("NDVI")
    assert "license" in ndvi
    assert "description" in ndvi


def test_collection_catalog_normalize_dont_override():
    catalog = CollectionCatalog([{"id": "SENTINEL2", "license": "free", "properties": {"eo:bands": [{"name": "red"}]}}])
    with pytest.warns(CollectionIncompleteMetadataWarning):
        s2 = catalog.get_collection_metadata("SENTINEL2")
    assert s2["license"] == "free"
    assert s2["properties"] == {"eo:bands": [{"name": "red"}]}
    assert s2["links"] == []
