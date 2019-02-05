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
        resp = self.client.post('/openeo/0.4.0/preview', content_type='application/json', data=json.dumps({'process_graph':{
            'filter_temp': {
                'process_id': 'filter_temporal',
                'arguments': {
                    'data': {
                        'from_node': 'collection'
                    },
                    'from': "2018-01-01",
                    'to': "2018-12-31"
                },
                'result':True
            },
            'collection': {
                'process_id': 'get_collection',
                'arguments':{
                    'name': 'S2_FAPAR_CLOUDCOVER'
                }
            }
        }

        }))

        assert resp.status_code == 200
        assert resp.content_length > 0

    def test_execute_apply_unary(self):
        resp = self.client.post('/openeo/0.4.0/preview', content_type='application/json', data=json.dumps({'process_graph':{
            'apply': {
                'process_id': 'apply',
                'arguments': {
                    'data': {
                        'from_node': 'collection'
                    },

                    'process':{
                        'callback':{
                            "abs":{
                                "arguments":{
                                    "data": {
                                        "from_argument": "dimension_data"
                                    }
                                },
                                "process_id":"abs"
                            },
                            "cos": {
                                "arguments":{
                                    "data": {
                                        "from_node": "abs"
                                    }
                                },
                                "process_id": "cos",
                                "result": True
                            }
                        }
                    }
                },
                'result':True
            },
            'collection': {
                'process_id': 'get_collection',
                'arguments':{
                    'name': 'S2_FAPAR_CLOUDCOVER'
                }
            }
        }

        }))

        assert resp.status_code == 200
        assert resp.content_length > 0

    def test_execute_reduce_max(self):
        resp = self.client.post('/openeo/0.4.0/preview', content_type='application/json', data=json.dumps({'process_graph':{
            'apply': {
                'process_id': 'reduce',
                'arguments': {
                    'data': {
                        'from_node': 'collection'
                    },
                    'dimension': 'temporal',
                    'reducer':{
                        'callback':{
                            "max":{
                                "arguments":{
                                    "data": {
                                        "from_argument": "dimension_data"
                                    }
                                },
                                "process_id":"max",
                                "result":True
                            }
                        }
                    }
                },
                'result':True
            },
            'collection': {
                'process_id': 'get_collection',
                'arguments':{
                    'name': 'S2_FAPAR_CLOUDCOVER'
                }
            }
        }

        }))

        assert resp.status_code == 200
        assert resp.content_length > 0

    def test_execute_reduce_bands(self):
        resp = self.client.post('/openeo/0.4.0/preview', content_type='application/json',
                                data=json.dumps({'process_graph': {
                                    'apply': {
                                        'process_id': 'reduce',
                                        'arguments': {
                                            'data': {
                                                'from_node': 'collection'
                                            },
                                            'dimension': 'spectral_bands',
                                            'reducer': {
                                                'callback': {
                                                    "sum": {
                                                        "arguments": {
                                                            "data": {
                                                                "from_argument": "dimension_data"
                                                            }
                                                        },
                                                        "process_id": "sum"
                                                    },
                                                    "subtract": {
                                                        "arguments": {
                                                            "data": {
                                                                "from_argument": "dimension_data"
                                                            }
                                                        },
                                                        "process_id": "subtract"
                                                    },
                                                    "divide": {
                                                        "arguments": {
                                                            "y": {
                                                                "from_node": "subtract"
                                                            },
                                                            "x": {
                                                                "from_node": "sum"
                                                            }
                                                        },
                                                        "process_id": "divide",
                                                        "result": True
                                                    }
                                                }
                                            }
                                        },
                                        'result': True
                                    },
                                    'collection': {
                                        'process_id': 'get_collection',
                                        'arguments': {
                                            'name': 'S2_FAPAR_CLOUDCOVER'
                                        }
                                    }
                                }

                                }))

        assert resp.status_code == 200
        assert resp.content_length > 0

    def test_execute_mask(self):
        resp = self.client.post('/openeo/0.4.0/preview', content_type='application/json', data=json.dumps({'process_graph':{
            'apply': {
                'process_id': 'mask',
                'arguments': {
                    'data': {
                        'from_node': 'collection'
                    },
                    'mask': {
                        'from_node': 'mask_collection'
                    },
                    'replacement':'10'
                },
                'result':True
            },
            'collection': {
                'process_id': 'get_collection',
                'arguments':{
                    'name': 'S2_FAPAR_CLOUDCOVER'
                }
            },
            'mask_collection': {
                'process_id': 'get_collection',
                'arguments': {
                    'name': 'S2_FAPAR_CLOUDCOVER'
                }
            }
        }

        }))

        assert resp.status_code == 200
        assert resp.content_length > 0
        import dummy_impl
        print(dummy_impl.collections["S2_FAPAR_CLOUDCOVER"])
        assert dummy_impl.collections["S2_FAPAR_CLOUDCOVER"].mask.call_count == 1

    def test_preview_aggregate_temporal_max(self):
        resp = self.client.post('/openeo/0.4.0/preview', content_type='application/json',
                                data=json.dumps({'process_graph': {
                                    'apply': {
                                        'process_id': 'aggregate_temporal',
                                        'arguments': {
                                            'data': {
                                                'from_node': 'collection'
                                            },
                                            'dimension': 'temporal',
                                            'intervals': [],
                                            'labels':[],
                                            'reducer': {
                                                'callback': {
                                                    "max": {
                                                        "arguments": {
                                                            "data": {
                                                                "from_argument": "dimension_data"
                                                            }
                                                        },
                                                        "process_id": "max",
                                                        "result": True
                                                    }
                                                }
                                            }
                                        },
                                        'result': True
                                    },
                                    'collection': {
                                        'process_id': 'get_collection',
                                        'arguments': {
                                            'name': 'S2_FAPAR_CLOUDCOVER'
                                        }
                                    }
                                }

                                }))

        assert resp.status_code == 200
        assert resp.content_length > 0
