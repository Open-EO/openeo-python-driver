import json

import pystac
import pystac.extensions.datacube
import pytest
from openeo.testing.stac import StacDummyBuilder

from openeo_driver.stac.datacube import get_spatial_dimensions


def test_get_spatial_dimensions_basic(tmp_path):
    path = tmp_path / "collection.json"
    path.write_text(
        json.dumps(
            StacDummyBuilder.collection(
                cube_dimensions={
                    "x": {"type": "spatial", "axis": "x", "extent": [-180, 180], "reference_system": 4326},
                    "y": {"type": "spatial", "axis": "y", "extent": [-56, 83], "reference_system": 4326},
                    "time": {
                        "type": "temporal",
                        "extent": ["2015-06-23T00:00:00Z", "2019-07-10T13:44:56Z"],
                        "step": "P5D",
                    },
                    "spectral": {"type": "bands", "values": ["B1", "B2"]},
                },
            )
        )
    )

    stac_object = pystac.read_file(path)
    spatial_dimensions = get_spatial_dimensions(stac_object)
    assert len(spatial_dimensions) == 2
    assert sorted(spatial_dimensions.keys()) == ["x", "y"]
    assert set(type(dim) for dim in spatial_dimensions.values()) == {
        pystac.extensions.datacube.HorizontalSpatialDimension
    }
    x = spatial_dimensions["x"]
    assert (x.dim_type, x.axis, x.extent, x.reference_system, x.step) == ("spatial", "x", [-180, 180], 4326, None)
    y = spatial_dimensions["y"]
    assert (y.dim_type, y.axis, y.extent, y.reference_system, y.step) == ("spatial", "y", [-56, 83], 4326, None)


@pytest.mark.parametrize(
    ["datacube_extension"],
    [
        ("https://stac-extensions.github.io/datacube/v2.0.0/schema.json",),
        ("https://stac-extensions.github.io/datacube/v2.2.0/schema.json",),
    ],
)
def test_get_spatial_dimensions_version_compatibility(tmp_path, datacube_extension):
    path = tmp_path / "collection.json"
    path.write_text(
        json.dumps(
            {
                "type": "Collection",
                "stac_version": "1.0.0",
                "stac_extensions": [datacube_extension],
                "id": "collection123",
                "description": "Collection 123",
                "license": "proprietary",
                "extent": {
                    "spatial": {"bbox": [[3, 4, 5, 6]]},
                    "temporal": {"interval": [["2024-01-01", "2024-05-05"]]},
                },
                "cube:dimensions": {
                    "x": {"type": "spatial", "axis": "x", "extent": [-180, 180], "reference_system": 4326},
                    "y": {"type": "spatial", "axis": "y", "extent": [-56, 83], "reference_system": 4326},
                    "time": {
                        "type": "temporal",
                        "extent": ["2015-06-23T00:00:00Z", "2019-07-10T13:44:56Z"],
                        "step": "P5D",
                    },
                    "spectral": {"type": "bands", "values": ["B1", "B2"]},
                },
                "links": [],
            }
        )
    )

    stac_object = pystac.read_file(path)
    spatial_dimensions = get_spatial_dimensions(stac_object)
    assert len(spatial_dimensions) == 2
    assert sorted(spatial_dimensions.keys()) == ["x", "y"]
    assert set(type(dim) for dim in spatial_dimensions.values()) == {
        pystac.extensions.datacube.HorizontalSpatialDimension
    }
