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
                    "geometry": {
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
