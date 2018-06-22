from unittest.mock import Mock
from openeo import ImageCollection
import os
from shapely.geometry import Polygon, MultiPolygon


def getImageCollection(product_id, viewingParameters):
    image_collection = ImageCollection()

    if product_id == 'S2_FAPAR_CLOUDCOVER':
        download = Mock(name='download')
        download.return_value = os.path.realpath(__file__)

        image_collection.download = download
    else:
        timeseries = Mock(name='timeseries')
        timeseries.return_value = {"hello": "world"}

        image_collection.timeseries = timeseries

        def is_polygon_or_multipolygon(return_value, *args):
            assert len(args) == 1
            assert isinstance(args[0], Polygon) or isinstance(args[0], MultiPolygon)
            return return_value

        polygonal_mean_timeseries = Mock(name='polygonal_mean_timeseries')
        polygonal_mean_timeseries.side_effect = lambda args: is_polygon_or_multipolygon({'hello': 'world'}, args)

        image_collection.polygonal_mean_timeseries = polygonal_mean_timeseries

    return image_collection


fapar_layer = {'product_id': 'S2_FAPAR_CLOUDCOVER'}
def get_layers():
    return [fapar_layer]

def get_layer(product_id):
    if product_id == 'S2_FAPAR_CLOUDCOVER':
        return fapar_layer
    else:
        raise ValueError("Unknown collection: " + product_id)


def health_check():
    return "OK"
