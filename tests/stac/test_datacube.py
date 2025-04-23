import json

import pytest
from openeo.testing.stac import StacDummyBuilder

from openeo_driver.stac.datacube import stac_to_cube_metadata


def test_stac_to_cube_metadata_basic(tmp_path):
    path = tmp_path / "collection.json"
    path.write_text(
        json.dumps(
            StacDummyBuilder.collection(
                cube_dimensions={
                    "x": {"type": "spatial", "axis": "x", "extent": [-180, 180], "reference_system": 4326},
                    "y": {"type": "spatial", "axis": "y", "extent": [-56, 83], "reference_system": 4326},
                    "t": {
                        "type": "temporal",
                        "extent": ["2015-06-23T00:00:00Z", "2019-07-10T13:44:56Z"],
                        "step": "P5D",
                    },
                    "bands": {"type": "bands", "values": ["B1", "B2"]},
                },
            )
        )
    )

    metadata = stac_to_cube_metadata(stac_ref=path)

    [dim_x, dim_y] = metadata.spatial_dimensions
    assert (dim_x.name, dim_x.extent, dim_x.crs, dim_x.step) == ("x", [-180, 180], 4326, None)
    assert (dim_y.name, dim_y.extent, dim_y.crs, dim_y.step) == ("y", [-56, 83], 4326, None)

    dim_t = metadata.temporal_dimension
    assert (dim_t.name, dim_t.extent) == ("t", ["2015-06-23T00:00:00Z", "2019-07-10T13:44:56Z"])

    dim_bands = metadata.band_dimension
    assert (dim_bands.name, dim_bands.band_names) == ("bands", ["B1", "B2"])


@pytest.mark.parametrize(
    ["datacube_extension"],
    [
        ("https://stac-extensions.github.io/datacube/v2.0.0/schema.json",),
        ("https://stac-extensions.github.io/datacube/v2.2.0/schema.json",),
    ],
)
def test_stac_to_cube_metadata_version_compatibility(tmp_path, datacube_extension):
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
                    "t": {
                        "type": "temporal",
                        "extent": ["2015-06-23T00:00:00Z", "2019-07-10T13:44:56Z"],
                        "step": "P5D",
                    },
                    "bands": {"type": "bands", "values": ["B1", "B2"]},
                },
                "links": [],
            }
        )
    )

    metadata = stac_to_cube_metadata(stac_ref=path)

    [dim_x, dim_y] = metadata.spatial_dimensions
    assert (dim_x.name, dim_x.extent, dim_x.crs, dim_x.step) == ("x", [-180, 180], 4326, None)
    assert (dim_y.name, dim_y.extent, dim_y.crs, dim_y.step) == ("y", [-56, 83], 4326, None)

    dim_t = metadata.temporal_dimension
    assert (dim_t.name, dim_t.extent) == ("t", ["2015-06-23T00:00:00Z", "2019-07-10T13:44:56Z"])

    dim_bands = metadata.band_dimension
    assert (dim_bands.name, dim_bands.band_names) == ("bands", ["B1", "B2"])
