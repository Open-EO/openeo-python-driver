from openeo_driver.delayed_vector import DelayedVector
from .data import get_path


def test_feature_collection_bounds():
    dv = DelayedVector(str(get_path("FeatureCollection.geojson")))
    assert dv.bounds == (4.45, 51.1, 4.52, 51.2)


def test_geometry_collection_bounds():
    dv = DelayedVector(str(get_path("GeometryCollection.geojson")))
    assert dv.bounds == (5.05, 51.21, 5.15, 51.3)
