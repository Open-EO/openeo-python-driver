import openeo.metadata
from openeo_driver.datacube import DriverDataCube
import dirty_equals


class TestDriverDatacCube:
    def test_repr_default(self):
        cube = DriverDataCube()
        assert repr(cube) == "DriverDataCube(metadata=CubeMetadata(dimensions=None))"

    def test_repr_simple(self):
        metadata = openeo.metadata.CubeMetadata(
            dimensions=[
                openeo.metadata.SpatialDimension(name="x", extent=[12, 34]),
                openeo.metadata.BandDimension(
                    name="bands", bands=[openeo.metadata.Band("B02"), openeo.metadata.Band("B03")]
                ),
            ]
        )
        cube = DriverDataCube(metadata=metadata)
        expected = "DriverDataCube(metadata=CubeMetadata(dimension_names=['x', 'bands'], band_names=['B02', 'B03']))"
        assert repr(cube) == expected
