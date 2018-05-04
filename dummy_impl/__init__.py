from unittest.mock import Mock
from openeo import ImageCollection
import os


def getImageCollection(product_id, viewingParameters):
    download = Mock(name='download')
    download.return_value = os.path.realpath(__file__)

    image_collection = ImageCollection()
    image_collection.download = download

    return image_collection


def get_layers():
    pass


def health_check():
    return "OK"
