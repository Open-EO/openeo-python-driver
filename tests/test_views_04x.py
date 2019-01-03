from unittest import TestCase, skip

from openeo_driver import app
import json
import os
os.environ["DRIVER_IMPLEMENTATION_PACKAGE"] = "dummy_impl"

client = app.test_client()

class Test(TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()


    def test_execute_filter_temporal(self):
        resp = self.client.post('/openeo/0.4.0/execute', content_type='application/json', data=json.dumps({
            'filter_temp': {
                'process_id': 'filter_temporal',
                'arguments': {
                    'data': {
                        'from_node': 'collection'
                    },
                    'from': "2018-01-01",
                    'to': "2018-12-31"
                },
                'result':'true'
            },
            'collection': {
                'process_id': 'get_collection',
                'arguments':{
                    'name': 'S2_FAPAR_CLOUDCOVER'
                }
            }
        }))

        assert resp.status_code == 200
        assert resp.content_length > 0
