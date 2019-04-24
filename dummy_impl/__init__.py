from unittest.mock import Mock
from openeo import ImageCollection
import os
from shapely.geometry import Polygon, MultiPolygon

collections = {}

def getImageCollection(product_id, viewingParameters):
    if product_id in collections:
        return collections[product_id]
    image_collection = ImageCollection()
    image_collection.viewingParameters = viewingParameters

    image_collection.mask = Mock(name="mask")
    image_collection.mask.return_value = image_collection

    image_collection.mask_polygon = Mock(name="mask_polygon")
    image_collection.mask_polygon.return_value = image_collection

    image_collection.bbox_filter = Mock(name="bbox_filter")
    image_collection.bbox_filter.return_value = image_collection

    image_collection.tiled_viewing_service = Mock(name="tiled_viewing_service")
    image_collection.tiled_viewing_service.return_value = {
        'type': 'WMTS',
        'url': "http://openeo.vgt.vito.be/openeo/services/c63d6c27-c4c2-4160-b7bd-9e32f582daec/service/wmts"
    }

    download = Mock(name='download')
    download.return_value = os.path.realpath(__file__)

    image_collection.download = download

    timeseries = Mock(name='timeseries')
    timeseries.return_value = {
        "viewingParameters" : image_collection.viewingParameters
    }

    image_collection.timeseries = timeseries

    def is_polygon_or_multipolygon(return_value, regions, func):
        assert func == 'mean' or func == 'avg'
        assert isinstance(regions, Polygon) or isinstance(regions, MultiPolygon)
        return return_value

    zonal_statistics = Mock(name='zonal_statistics')
    zonal_statistics.side_effect = lambda regions, func: is_polygon_or_multipolygon({'hello': 'world'}, regions, func)

    image_collection.zonal_statistics = zonal_statistics

    image_collection.apply_pixel = Mock(name = "apply_pixel")
    image_collection.apply_pixel.return_value = image_collection

    image_collection.apply_tiles_spatiotemporal = Mock(name="apply_tiles_spatiotemporal")
    image_collection.apply_tiles_spatiotemporal.return_value = image_collection

    image_collection.apply_tiles = Mock(name="apply_tiles")
    image_collection.apply_tiles.return_value = image_collection

    image_collection.apply = Mock(name="apply")
    image_collection.apply.return_value = image_collection

    image_collection.reduce= Mock(name="reduce")
    image_collection.reduce.return_value = image_collection

    image_collection.reduce_bands= Mock(name="reduce_bands")
    image_collection.reduce_bands.return_value = image_collection

    image_collection.aggregate_temporal= Mock(name="aggregate_temporal")
    image_collection.aggregate_temporal.return_value = image_collection

    image_collection.max_time = Mock(name="max_time")
    image_collection.max_time.return_value = image_collection

    collections[product_id] = image_collection
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


def create_batch_job(*_):
    return '07024ee9-7847-4b8a-b260-6c879a2b3cdc'


def run_batch_job(*_):
    return


def get_batch_job_info(job_id):
    return {
        'job_id': job_id,
        'status': 'running'
    }


def get_batch_job_result_filenames(job_id):
    pass


def get_batch_job_result_output_dir(job_id):
    return "/path/to/%s" % job_id


def cancel_batch_job(job_id):
    pass


from openeo.internal.process_graph_visitor import ProcessGraphVisitor
class DummyVisitor(ProcessGraphVisitor):

    def __init__(self):
        super(DummyVisitor, self).__init__()

def create_process_visitor():
    return DummyVisitor()


def get_secondary_services_info():
    pass


def get_secondary_service_info(service_id):
    pass
