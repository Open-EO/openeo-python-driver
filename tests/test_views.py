from unittest import TestCase, skip
from openeo_driver import app
import json


class Test(TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()

    def test_health(self):
        resp = self.client.get('/openeo/health')

        assert resp.status_code == 200
        assert "OK" in resp.get_data(as_text=True)

    def test_data(self):
        resp = self.client.get('/openeo/data')

        assert resp.status_code == 200
        collections = json.loads(resp.get_data().decode('utf-8'))
        assert collections
        assert 'S2_FAPAR_CLOUDCOVER' in [collection['product_id'] for collection in collections]

    def test_data_detail(self):
        resp = self.client.get('/openeo/data/S2_FAPAR_CLOUDCOVER')

        assert resp.status_code == 200
        collection = json.loads(resp.get_data().decode('utf-8'))
        assert collection['product_id'] == 'S2_FAPAR_CLOUDCOVER'

    def test_execute_image_collection(self):
        resp = self.client.post('/openeo/execute', content_type='application/json', data=json.dumps({
            'process_graph': {
                'collection_id': 'S2_FAPAR_CLOUDCOVER'
            },
            'output': {}
        }))

        assert resp.status_code == 200
        assert resp.content_length > 0

    def test_point(self):
        resp = self.client.post('/openeo/timeseries/point?x=1&y=2', content_type='application/json', data=json.dumps({
            'collection_id': 'Sentinel2-L1C'
        }))

        assert resp.status_code == 200
        assert json.loads(resp.get_data(as_text=True)) == {"hello": "world"}

    def test_execute_zonal_statistics(self):
        resp = self.client.post('/openeo/execute', content_type='application/json', data=json.dumps({
            "process_graph": {
                "process_id": "zonal_statistics",
                "args": {
                    "imagery": {
                        "product_id": "Sentinel2-L1C"
                    },
                    "regions": {
                        "type": "Polygon",
                        "coordinates": [
                            [[7.022705078125007, 51.75432477678571], [7.659912109375007, 51.74333844866071],
                             [7.659912109375007, 51.29289899553571], [7.044677734375007, 51.31487165178571],
                             [7.022705078125007, 51.75432477678571]]
                        ]
                    }
                }
            }
        }))

        assert resp.status_code == 200
        assert json.loads(resp.get_data(as_text=True)) == {"hello": "world"}

    def test_execute_filter_daterange(self):
        resp = self.client.post('/openeo/execute', content_type='application/json', data=json.dumps({
            'process_graph': {
                'process_id': 'filter_daterange',
                'args': {
                    'imagery': {
                        'collection_id': 'S2_FAPAR_CLOUDCOVER'
                    },
                    'from': "2018-01-01",
                    'to': "2018-12-31"
                }
            },
            'output': {}
        }))

        assert resp.status_code == 200
        assert resp.content_length > 0

    def test_execute_max_time(self):
        graph = {"process_graph": {"process_id": "max_time", "args": {"imagery": {"process_id": "filter_bbox",
                                                                          "args": {"srs": "EPSG:4326", "right": 5.7,
                                                                                   "top": 50.28, "left": 4.3,
                                                                                   "imagery": {
                                                                                       "process_id": "filter_daterange",
                                                                                       "args": {"imagery": {
                                                                                           "collection_id": "S2_FAPAR"},
                                                                                                "to": "2017-08-10",
                                                                                                "from": "2017-06-01"}},
                                                                                   "bottom": 50.55
                                                                                   }
                                                                                  }
                                                                      }
                                   }, "output": {}}
        resp = self.client.post('/openeo/execute', content_type='application/json', data=json.dumps(graph))

        assert resp.status_code == 200
        assert resp.content_length > 0

    def test_execute_apply_tiles(self):
        graph = {
                    "process_graph": {"process_id": "apply_tiles", "args": {"imagery": {"process_id": "filter_bbox",
                                                                                     "args": {"imagery": {
                                                                                         "process_id": "filter_daterange",
                                                                                         "args": {"imagery": {
                                                                                             "collection_id": "CGS_SENTINEL2_RADIOMETRY_V101"},
                                                                                                  "from": "2017-10-01",
                                                                                                  "to": "2017-10-11"}},
                                                                                              "srs": "EPSG:3857",
                                                                                              "right": 763281,
                                                                                              "top": 6544655,
                                                                                              "bottom": 6543830,
                                                                                              "left": 761104}},
                                                                         "code": {"language": "python",
                                                                                  "source": "# -*- coding: utf-8 -*-\n# Uncomment the import only for coding support\n#import numpy\n#import pandas\n#import torch\n#import torchvision\n#import tensorflow\n#import tensorboard\n#from openeo_udf.api.base import SpatialExtent, RasterCollectionTile, FeatureCollectionTile, UdfData\n\n__license__ = \"Apache License, Version 2.0\"\n__author__ = \"Soeren Gebbert\"\n__copyright__ = \"Copyright 2018, Soeren Gebbert\"\n__maintainer__ = \"Soeren Gebbert\"\n__email__ = \"soerengebbert@googlemail.com\"\n\n\ndef rct_ndvi(udf_data):\n    \"\"\"Compute the NDVI based on RED and NIR tiles\n\n    Tiles with ids \"red\" and \"nir\" are required. The NDVI computation will be applied\n    to all time stamped 2D raster tiles that have equal time stamps.\n\n    Args:\n        udf_data (UdfData): The UDF data object that contains raster and vector tiles\n\n    Returns:\n        This function will not return anything, the UdfData object \"udf_data\" must be used to store the resulting\n        data.\n\n    \"\"\"\n    red = None\n    nir = None\n\n    # Iterate over each tile\n    for tile in udf_data.raster_collection_tiles:\n        if \"red\" in tile.id.lower():\n            red = tile\n        if \"nir\" in tile.id.lower():\n            nir = tile\n    if red is None:\n        raise Exception(\"Red raster collection tile is missing in input\")\n    if nir is None:\n        raise Exception(\"Nir raster collection tile is missing in input\")\n    if red.start_times is None or red.start_times.tolist() == nir.start_times.tolist():\n        # Compute the NDVI\n        ndvi = (nir.data - red.data) / (nir.data + red.data)\n        # Create the new raster collection tile\n        rct = RasterCollectionTile(id=\"ndvi\", extent=red.extent, data=ndvi,\n                                   start_times=red.start_times, end_times=red.end_times)\n        # Insert the new tiles as list of raster collection tiles in the input object. The new tiles will\n        # replace the original input tiles.\n        udf_data.set_raster_collection_tiles([rct,])\n    else:\n        raise Exception(\"Time stamps are not equal\")\n\n\n# This function call is the entry point for the UDF.\n# The caller will provide all required data in the **data** object.\nrct_ndvi(data)\n"
                                                                                  }
                                                                         }
                                   },
                 "output": {}
        }
        resp = self.client.post('/openeo/execute', content_type='application/json', data=json.dumps(graph))
        assert resp.status_code == 200
        assert resp.content_length > 0

    def test_create_job(self):
        resp = self.client.post('/openeo/jobs', content_type='application/json', data=json.dumps({
            'process_graph': {},
            'output': {}
        }))

        assert resp.status_code == 201
        assert resp.content_length == 0
        assert resp.headers['Location'].endswith('/openeo/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc')

    def test_get_job_info(self):
        resp = self.client.get('/openeo/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc')

        self.assertEqual(200, resp.status_code)

        info = resp.get_json()

        self.assertEqual(info['job_id'], '07024ee9-7847-4b8a-b260-6c879a2b3cdc')
        self.assertEqual(info['status'], 'running')


    def test_api_propagates_http_status_codes(self):
        resp = self.client.get('/openeo/jobs/unknown_job_id/results/some_file')

        assert resp.status_code == 404
