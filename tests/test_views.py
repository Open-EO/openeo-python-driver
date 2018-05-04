from unittest import TestCase, skip
from openeo_driver import app
import json


class Test(TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()

    def test_health(self):
        resp = self.client.get('/openeo/health')

        print("resp.data is %s" % resp.data)

        assert resp.status_code == 200
        assert b"OK" in resp.data

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
