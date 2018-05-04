from unittest.mock import Mock
from openeo import ImageCollection
import os


def getImageCollection(product_id, viewingParameters):
    image_collection = ImageCollection()

    if product_id == 'S2_FAPAR_CLOUDCOVER':
        download = Mock(name='download')
        download.return_value = os.path.realpath(__file__)

        image_collection.download = download

        return image_collection
    else:
        timeseries = Mock(name='timeseries')
        timeseries.return_value = {"hello": "world"}

        image_collection.timeseries = timeseries

    return image_collection


def get_layers():
    pass


def health_check():
    return "OK"
