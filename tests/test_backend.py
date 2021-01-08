import pytest

from openeo_driver.backend import CollectionCatalog, LoadParameters
from openeo_driver.datastructs import SarBackscatterArgs
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
    assert isinstance(params.sar_backscatter, SarBackscatterArgs)
    assert params.sar_backscatter == SarBackscatterArgs()

    params_copy = params.copy()
    assert isinstance(params_copy, LoadParameters)
    assert params_copy.temporal_extent == ("2021-01-01", None)

    params.bands = ["red", "green"]
    assert params.bands == ["red", "green"]
    assert params_copy.bands is None
