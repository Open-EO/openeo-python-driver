from openeo_driver.delayed_vector import DelayedVector
from .data import get_path


def test_feature_collection_bounds():
    dv = DelayedVector(str(get_path("FeatureCollection.geojson")))
    assert dv.bounds == (4.461822509765625, 51.14704810491208, 4.5208740234375, 51.19354556162766)


def test_geometry_collection_bounds():
    dv = DelayedVector(str(get_path("GeometryCollection.geojson")))
    assert dv.bounds == (5.05, 51.21, 5.15, 51.3)
