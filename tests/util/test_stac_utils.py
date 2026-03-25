import shutil
from pathlib import Path

from pystac import Collection

from openeo_driver.util.stac_utils import (
    get_files_from_stac_catalog,
    get_assets_from_stac_catalog,
    get_items_from_stac_catalog,
    find_stac_root,
)

repository_root = Path(__file__).parent.parent.parent
print(repository_root)


def test_get_files_from_stac_catalog_path():
    stac_root = repository_root / "tests/data/example_stac_catalog/collection.json"
    ret = get_files_from_stac_catalog(stac_root)
    print(ret)
    assert len(ret) == 3


def test_get_files_from_stac_catalog_path_include_metadata():
    stac_root = repository_root / "tests/data/example_stac_catalog/collection.json"
    ret = get_files_from_stac_catalog(stac_root, include_metadata=True)
    print(ret)
    assert len(ret) == 7


def test_get_files_from_stac_catalog_url():
    stac_root = "https://raw.githubusercontent.com/Open-EO/openeo-geopyspark-driver/refs/heads/master/docker/local_batch_job/example_stac_catalog/collection.json"
    ret = get_files_from_stac_catalog(stac_root)

    print(ret)
    assert len(ret) == 3


def test_get_files_from_stac_catalog_url_include_metadata():
    stac_root = "https://raw.githubusercontent.com/Open-EO/openeo-geopyspark-driver/refs/heads/master/docker/local_batch_job/example_stac_catalog/collection.json"
    ret = get_files_from_stac_catalog(stac_root, include_metadata=True)

    print(ret)
    assert len(ret) == 7


def test_get_assets_from_stac_catalog():
    stac_root = repository_root / "tests/data/example_stac_catalog/collection.json"
    ret = get_assets_from_stac_catalog(stac_root)
    print(ret)
    assert len(ret.values()) == 3


def test_get_items_from_stac_catalog():
    stac_root = repository_root / "tests/data/example_stac_catalog/collection.json"
    ret = get_items_from_stac_catalog(stac_root)
    print(ret)
    assert len(ret) == 3


def test_get_items_from_stac_catalog_recursive():
    stac_root = str(repository_root / "tests/data/recursive-stac-example/collection.json")
    ret = get_items_from_stac_catalog(stac_root, make_hrefs_absolute=True)
    print(ret)
    assert len(ret) == 3


def test_find_stac_root_dictionary():
    listing_directory = [
        "r-2512020921004d489e-cal-cwl-e706a8a7/o1kip6_h/collection.json",
        "r-2512020921004d489e-cal-cwl-e706a8a7/o1kip6_h/openEO_2023-06-01Z.tif",
        "r-2512020921004d489e-cal-cwl-e706a8a7/o1kip6_h/openEO_2023-06-01Z.tif.json",
        "r-2512020921004d489e-cal-cwl-e706a8a7/o1kip6_h/openEO_2023-06-04Z.tif",
        "r-2512020921004d489e-cal-cwl-e706a8a7/o1kip6_h/openEO_2023-06-04Z.tif.json",
        "r-2512020921004d489e-cal-cwl-e706a8a7/o1kip6_h/openEO_2023-06-06Z.tif",
        "r-2512020921004d489e-cal-cwl-e706a8a7/o1kip6_h/openEO_2023-06-06Z.tif.json",
    ]
    result = find_stac_root(listing_directory)
    assert result
    assert isinstance(result, str)
    assert result == "r-2512020921004d489e-cal-cwl-e706a8a7/o1kip6_h/collection.json"


def test_find_stac_root_file_array_01():
    listing_directory = [
        "aaa/collection.json",
        "bbb/collection-custom.json",
    ]
    result = find_stac_root(listing_directory, "collection-custom.json")
    assert result
    assert isinstance(result, str)
    assert result == "bbb/collection-custom.json"


def test_find_stac_root_file_array_02():
    listing_directory = [
        "aaa/collection.json",
        "bbb/collection-custom.json",
    ]
    result = find_stac_root(listing_directory)
    assert result
    assert isinstance(result, str)
    assert result == "aaa/collection.json"


def test_find_stac_root_file_array_03():
    listing_directory = [
        "aaa/collection.json",
        "bbb/collection-custom.json",
        "ccc/catalog.json",
        "ddd/catalogue.json",
    ]
    result = find_stac_root(listing_directory)
    assert result
    assert isinstance(result, str)
    assert result == "ccc/catalog.json"


def test_find_stac_root_file_array_04():
    listing_directory = [
        "aaa/collection.json",
        "bbb/collection-custom.json",
        "ddd/catalogue.json",
    ]
    result = find_stac_root(listing_directory)
    assert result
    assert isinstance(result, str)
    assert result == "ddd/catalogue.json"
