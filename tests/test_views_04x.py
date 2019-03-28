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

    def test_udf_runtimes(self):
        runtimes = self.client.get('/openeo/0.4.0/udf_runtimes').json
        print(runtimes)
        self.assertIn("Python",runtimes)



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

    def test_execute_apply_run_udf(self):
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
                            "udf": {
                                "arguments":{
                                    "data": {
                                        "from_node": "abs"
                                    },
                                    "runtime":"Python",
                                    "version":"3.5.1",
                                    "udf":"my python code"

                                },
                                "process_id": "run_udf",
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
        import dummy_impl
        print(dummy_impl.collections["S2_FAPAR_CLOUDCOVER"])
        assert dummy_impl.collections["S2_FAPAR_CLOUDCOVER"].apply_tiles.call_count == 1

    def test_execute_reduce_temporal_run_udf(self):
        resp = self.client.post('/openeo/0.4.0/preview', content_type='application/json', data=json.dumps({'process_graph':{
            'apply': {
                'process_id': 'reduce',
                'arguments': {
                    'data': {
                        'from_node': 'collection'
                    },
                    'dimension':'temporal',
                    #'binary': False,
                    'reducer':{
                        'callback':{
                            "udf": {
                                "arguments":{
                                    "data": {
                                        "from_argument": "dimension_data"
                                    },
                                    "runtime":"Python",
                                    "version":"3.5.1",
                                    "udf":"my python code"

                                },
                                "process_id": "run_udf",
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
        import dummy_impl
        print(dummy_impl.collections["S2_FAPAR_CLOUDCOVER"])
        assert dummy_impl.collections["S2_FAPAR_CLOUDCOVER"].apply_tiles_spatiotemporal.call_count == 1

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


    def test_create_wmts(self):
        process_graph = {
            'collection': {
                'arguments': {
                    'name': 'S2'
                },
                'process_id': 'get_collection'
            },
            'filter_temp': {
                'arguments': {
                    'data': {
                        'from_node': 'collection'
                    },
                    'from': "2018-01-01",
                    'to': "2018-12-31"
                },
                'process_id': 'filter_temporal',
                'result': True
            }
        }
        from openeo.internal.process_graph_visitor import ProcessGraphVisitor
        resp = self.client.post('/openeo/0.4.0/services',content_type='application/json', json={
            "custom_param":45,
            "process_graph": process_graph,
            "type":'WMTS',
            "title":"My Service",
            "description":"Service description"
        })

        self.assertEqual(201,resp.status_code )
        self.assertDictEqual(resp.json,{
            "type": "WMTS",
            "url": "http://openeo.vgt.vito.be/service/wmts"
        })
        #TODO fix and check on Location Header
        #self.assertEqual("http://.../openeo/services/myservice", resp.location)

        import dummy_impl
        print(dummy_impl.collections["S2"])
        self.assertEqual(1, dummy_impl.collections["S2"].tiled_viewing_service.call_count )
        result_node = ProcessGraphVisitor._list_to_graph(process_graph)
        dummy_impl.collections["S2"].tiled_viewing_service.assert_called_with(custom_param=45,
                                                                              description='Service description', process_graph=process_graph, title='My Service', type='WMTS')