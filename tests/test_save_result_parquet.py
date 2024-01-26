from openeo_driver.datacube import DriverVectorCube
from openeo_driver.save_result import AggregatePolygonSpatialResult
from .data import get_path

import geopandas as gpd
from shapely.geometry import Polygon


def test_save_aggregate_polygon_spatial_result(tmp_path):
    csv_dir = get_path("aggregate_spatial_spatial_cube")

    vector_cube = DriverVectorCube(gpd.read_file(str(get_path("geojson/FeatureCollection02.json"))))

    output_file = tmp_path / "test.parquet"

    spatial_result = AggregatePolygonSpatialResult(csv_dir, regions=vector_cube, format="Parquet")
    spatial_result.to_geoparquet(destination=str(output_file))

    assert gpd.read_parquet(output_file).to_dict('list') == {
        'geometry': [Polygon([(1, 1), (3, 1), (2, 3), (1, 1)]), Polygon([(4, 2), (5, 4), (3, 4), (4, 2)])],
        'id': ['first', 'second'],
        'pop': [1234, 5678],
        'avg_band_0': [4646.262612301313, 4645.719597475695],
        'avg_band_1': [4865.926572218383, 4865.467252259935],
        'avg_band_2': [5178.517363510712, 5177.803342998465],
    }
