from unittest import TestCase, skip
from multiprocessing import Pool
def setup_local_spark():
    '''
    For local debugging, we can run with the openeogeotrellis backend, which requires a local spark to be running.
    :return:
    '''
    from pyspark import find_spark_home
    import os, sys
    from glob import glob

    spark_python = os.path.join(find_spark_home._find_spark_home(), 'python')
    py4j = glob(os.path.join(spark_python, 'lib', 'py4j-*.zip'))[0]
    sys.path[:0] = [spark_python, py4j]
    if 'TRAVIS' in os.environ:
        master_str = "local[2]"
    else:
        master_str = "local[*]"

    from geopyspark import geopyspark_conf, Pyramid, TiledRasterLayer

    conf = geopyspark_conf(master=master_str, appName="test")
    conf.set('spark.kryoserializer.buffer.max', value='1G')
    conf.set('spark.ui.enabled', True)

    if 'TRAVIS' in os.environ:
        conf.set(key='spark.driver.memory', value='2G')
        conf.set(key='spark.executor.memory', value='2G')

    from pyspark import SparkContext
    pysc = SparkContext.getOrCreate(conf)
#setup_local_spark()

from openeo_driver import app
import json
import os
os.environ["DRIVER_IMPLEMENTATION_PACKAGE"] = "dummy_impl"

client = app.test_client()

class Test(TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()

    def test_health(self):
        resp = self.client.get('/openeo/health')

        assert resp.status_code == 200
        assert "OK" in resp.get_data(as_text=True)

    def test_data(self):
        resp = self.client.get('/openeo/collections')

        assert resp.status_code == 200
        collections = json.loads(resp.get_data().decode('utf-8'))
        assert collections
        assert 'S2_FAPAR_CLOUDCOVER' in [collection['product_id'] for collection in collections['collections']]

    def test_data_detail(self):
        resp = self.client.get('/openeo/data/S2_FAPAR_CLOUDCOVER')

        assert resp.status_code == 200
        collection = json.loads(resp.get_data().decode('utf-8'))
        assert collection['product_id'] == 'S2_FAPAR_CLOUDCOVER'

    def test_processes(self):
        resp = self.client.get('/openeo/processes')

        assert resp.status_code == 200
        process_dict = {item['name']: item for item in resp.json['processes']}
        print(process_dict)
        assert 'apply_pixel' in process_dict

    def test_process_details(self):
        resp = self.client.get('/openeo/processes/apply_pixel')

        assert resp.status_code == 200
        print(resp.json)

    def test_execute_image_collection(self):
        resp = self.client.post('/openeo/execute', content_type='application/json', data=json.dumps({
            'process_graph': {
                'collection_id': 'S2_FAPAR_CLOUDCOVER'
            },
            'output': {}
        }))

        assert resp.status_code == 200
        assert resp.content_length > 0



    def test_point_with_bbox(self):
        bbox = { 'left': 3, 'right': 6, 'top': 52, 'bottom': 50, 'srs': 'EPSG:4326','version':'0.3.1'}
        process_graph = {'process_graph': {'process_id': 'filter_bbox',
                                            'args': {'imagery': {'collection_id': 'SENTINEL2_FAPAR'}, 'left': 3,
                                                     'right': 6, 'top': 52, 'bottom': 50, 'srs': 'EPSG:4326'}}}

        resp = self.client.post('/openeo/timeseries/point?x=1&y=2', content_type='application/json', data=json.dumps(process_graph))

        assert resp.status_code == 200
        result = json.loads(resp.get_data(as_text=True))
        print(result)
        assert result == {"viewingParameters" : bbox}

    def test_execute_mask(self):
        resp = self.client.post('/openeo/execute', content_type='application/json', data=json.dumps({
            "process_graph": {
                "process_id": "mask_polygon",
                "args": {
                    "imagery": {
                        "product_id": "S2_FAPAR_CLOUDCOVER"
                    },
                    "mask_shape": {
                        "type": "Polygon",
                        "crs": {
                            'type': 'name',
                            'properties': {
                                'name': "EPSG:4326"
                            }
                        },
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
        assert resp.content_length > 0

        import dummy_impl
        print(dummy_impl.collections["S2_FAPAR_CLOUDCOVER"])
        assert dummy_impl.collections["S2_FAPAR_CLOUDCOVER"].mask_polygon.call_count == 1


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

    def test_execute_simple_download(self):
        import dummy_impl
        dummy_impl.collections["S2_FAPAR_CLOUDCOVER"].download.reset_mock()
        download_expected_graph = {'process_id': 'filter_bbox', 'args': {'imagery': {'process_id': 'filter_daterange',
                                                                                     'args': {'imagery': {'collection_id': 'S2_FAPAR_CLOUDCOVER'},
                                                                                              'from': '2018-08-06T00:00:00Z',
                                                                                              'to': '2018-08-06T00:00:00Z'}},
                                                                         'left': 5.027, 'right': 5.0438, 'top': 51.2213,
                                                                         'bottom': 51.1974, 'srs': 'EPSG:4326'}}
        resp = self.client.post('/openeo/execute', content_type='application/json', data=json.dumps({"process_graph":download_expected_graph}))

        assert resp.status_code == 200
        assert resp.content_length > 0
        #assert resp.headers['Content-Type'] == "application/octet-stream"


        print(dummy_impl.collections["S2_FAPAR_CLOUDCOVER"])
        self.assertEquals(1,dummy_impl.collections["S2_FAPAR_CLOUDCOVER"].download.call_count)




    @classmethod
    def _post_download(cls,index):
        download_expected_graph = {'process_id': 'filter_bbox', 'args': {'imagery': {'process_id': 'filter_daterange',
                                                                                     'args': {'imagery': {
                                                                                         'collection_id': 'S2_FAPAR_SCENECLASSIFICATION_V102_PYRAMID'},
                                                                                              'from': '2018-08-06T00:00:00Z',
                                                                                              'to': '2018-08-06T00:00:00Z'}},
                                                                         'left': 5.027, 'right': 5.0438, 'top': 51.2213,
                                                                         'bottom': 51.1974, 'srs': 'EPSG:4326'}}

        post_request = json.dumps({"process_graph": download_expected_graph})
        resp = client.post('/openeo/execute', content_type='application/json',data=post_request)
        assert resp.status_code == 200
        assert resp.content_length > 0
        return index

    @skip
    def test_execute_download_parallel(self):
        """
        Tests downloading in parallel, see EP-2743
        Spark related issues are only exposed/tested when not using the dummy backend
        :return:
        """
        Test._post_download(1)

        with Pool(2) as pool:
            result = pool.map(Test._post_download,range(1,3))

        print(result)



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

    def test_execute_apply_pixels(self):
        graph = {
                "process_graph":
                     {
                         "process_id": "max_time",
                         "args":
                             {
                                 "imagery":
                                     {
                                         "process_id": "apply_pixel",
                                         "args": {
                                             "bands": [],
                                             "function": "gASVTgEAAAAAAACMF2Nsb3VkcGlja2xlLmNsb3VkcGlja2xllIwOX2ZpbGxfZnVuY3Rpb26Uk5QoaACMD19tYWtlX3NrZWxfZnVuY5STlGgAjA1fYnVpbHRpbl90eXBllJOUjAhDb2RlVHlwZZSFlFKUKEsCSwBLAksES0NDIHwAAGQBABl8AABkAgAZGHwAAGQBABl8AABkAgAZFxtTlE5LA0sCh5QpjAVjZWxsc5SMBm5vZGF0YZSGlIwfPGlweXRob24taW5wdXQtMTEtNzQ5OGQ0YWM4Y2RiPpSMCDxsYW1iZGE+lEsBQwCUKSl0lFKUSv////99lIeUUpR9lCiMBGRpY3SUfZSMCGRlZmF1bHRzlE6MCHF1YWxuYW1llGgQjAZtb2R1bGWUjAhfX21haW5fX5SMB2dsb2JhbHOUfZSMDmNsb3N1cmVfdmFsdWVzlE51dFIu",
                                             "imagery": {
                                                 "process_id": "filter_bbox",
                                                 "args": {
                                                     "bottom": 6543830, "srs": "EPSG:3857", "right": 763281, "top": 6544655, "left": 761104,
                                                     "imagery": {
                                                         "process_id": "filter_daterange",
                                                         "args": {"to": "2017-10-17", "from": "2017-10-14", "imagery": {"collection_id": "CGS_SENTINEL2_RADIOMETRY_V101"}}}}}}}}}, "output": {}}
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

    def test_queue_job(self):
        resp = self.client.post('/openeo/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc/results')

        self.assertEqual(202, resp.status_code)

    def test_get_job_info(self):
        resp = self.client.get('/openeo/jobs/07024ee9-7847-4b8a-b260-6c879a2b3cdc')

        self.assertEqual(200, resp.status_code)

        info = resp.get_json()

        self.assertEqual(info['job_id'], '07024ee9-7847-4b8a-b260-6c879a2b3cdc')
        self.assertEqual(info['status'], 'running')

    def test_api_propagates_http_status_codes(self):
        resp = self.client.get('/openeo/jobs/unknown_job_id/results/some_file')

        assert resp.status_code == 404

    def test_create_wmts(self):
        resp = self.client.post('/openeo/services', content_type='application/json', json={
            "process_graph": {'product_id': 'S2'},
            "type": 'WMTS',
            "parameters": {'version': "1.3.0"},
            "title": "My Service",
            "description": "Service description"
        })

        self.assertEqual(201, resp.status_code)
        self.assertTrue(resp.location.endswith('/services/c63d6c27-c4c2-4160-b7bd-9e32f582daec/service/wmts'))

        import dummy_impl
        print(dummy_impl.collections["S2"])
        self.assertEqual(1, dummy_impl.collections["S2"].tiled_viewing_service.call_count )
        dummy_impl.collections["S2"].tiled_viewing_service.assert_called_with(custom_param=45,
                                                                              description='Service description', process_graph={'product_id': 'S2'}, title='My Service', type='WMTS')