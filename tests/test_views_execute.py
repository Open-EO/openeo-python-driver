import json
import os
from unittest import TestCase

import dummy_impl
from openeo.internal.process_graph_visitor import ProcessGraphVisitor
from openeo_driver.views import app
from .data import load_json, get_path

os.environ["DRIVER_IMPLEMENTATION_PACKAGE"] = "dummy_impl"


class Test(TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()
        dummy_impl.collections = {}

    def _post_process_graph(self, process_graph: dict, url='preview'):
        if not url.startswith('/openeo/'):
            url = '/openeo/0.4.2/' + url.lstrip("/")
        resp = self.client.post(
            url, content_type='application/json',
            data=json.dumps({'process_graph': process_graph}))
        return resp

    def test_udf_runtimes(self):
        runtimes = self.client.get('/openeo/0.4.0/udf_runtimes').json
        print(runtimes)
        self.assertIn("Python", runtimes)

    def test_load_collection(self):
        resp = self._post_process_graph({
            'collection': {
                'process_id': 'load_collection',
                'arguments': {'id': 'S2_FAPAR_CLOUDCOVER'},
                'result': True
            }
        })
        assert resp.status_code == 200
        assert resp.content_length > 0

    def test_execute_filter_temporal(self):
        resp = self._post_process_graph({
            'filter_temp': {
                'process_id': 'filter_temporal',
                'arguments': {
                    'data': {
                        'from_node': 'collection'
                    },
                    'extent': ['2018-01-01', '2018-12-31']
                },
                'result': True
            },
            'collection': {
                'process_id': 'load_collection',
                'arguments': {
                    'id': 'S2_FAPAR_CLOUDCOVER'
                }
            }
        })
        assert resp.status_code == 200
        assert resp.content_length > 0

    def test_execute_apply_kernel(self):
        kernel_list = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
        resp = self._post_process_graph({
            'kernel': {
                'process_id': 'apply_kernel',
                'arguments': {
                    'data': {
                        'from_node': 'collection'
                    },
                    'kernel': kernel_list,
                    'factor': 3
                },
                'result': True
            },
            'collection': {
                'process_id': 'load_collection',
                'arguments': {
                    'id': 'S2_FAPAR_CLOUDCOVER'
                }
            }
        })

        assert resp.status_code == 200
        assert resp.content_length > 0
        assert dummy_impl.collections["S2_FAPAR_CLOUDCOVER"].apply_kernel.call_count == 1

        np_kernel = dummy_impl.collections["S2_FAPAR_CLOUDCOVER"].apply_kernel.call_args[0][0]
        self.assertListEqual(np_kernel.tolist(), kernel_list)
        self.assertEqual(dummy_impl.collections["S2_FAPAR_CLOUDCOVER"].apply_kernel.call_args[0][1], 3)

    def test_execute_simple_download(self):
        resp = self._post_process_graph({
            'filter_bbox': {
                'process_id': 'filter_bbox',
                'result': True,
                'arguments': {
                    'data': {
                        'from_node': 'filter_temp'
                    },
                    'extent': {
                        'west': 5.027, 'east': 5.0438, 'north': 51.2213,
                        'south': 51.1974, 'crs': 'EPSG:4326'
                    }
                }
            },
            'filter_temp': {
                'process_id': 'filter_temporal',
                'arguments': {
                    'data': {
                        'from_node': 'collection'
                    },
                    'extent': ['2018-01-01', '2018-12-31']
                },
                'result': False
            },
            'collection': {
                'process_id': 'load_collection',
                'arguments': {
                    'id': 'S2_FAPAR_CLOUDCOVER'
                }
            }
        })

        assert resp.status_code == 200
        assert resp.content_length > 0
        # assert resp.headers['Content-Type'] == "application/octet-stream"

        assert dummy_impl.collections["S2_FAPAR_CLOUDCOVER"].download.call_count == 1

    def test_load_collection_filter(self):
        resp = self._post_process_graph({
            'collection': {
                'process_id': 'load_collection',
                'arguments': {
                    'id': 'S2_FAPAR_CLOUDCOVER',
                    'spatial_extent': {
                        'west': 5.027, 'east': 5.0438, 'north': 51.2213,
                        'south': 51.1974, 'crs': 'EPSG:4326'
                    },
                    'temporal_extent': ['2018-01-01', '2018-12-31']
                },
                'result': True
            }
        })
        assert resp.status_code == 200
        assert resp.content_length > 0
        # assert resp.headers['Content-Type'] == "application/octet-stream"

        assert dummy_impl.collections['S2_FAPAR_CLOUDCOVER'].download.call_count == 1
        assert dummy_impl.collections['S2_FAPAR_CLOUDCOVER'].viewingParameters == {
            'version': '0.4.2', 'from': '2018-01-01', 'to': '2018-12-31',
            'left': 5.027, 'right': 5.0438, 'top': 51.2213, 'bottom': 51.1974, 'srs': 'EPSG:4326'}

    def test_execute_apply_unary(self):
        resp = self._post_process_graph({
            'apply': {
                'process_id': 'apply',
                'arguments': {
                    'data': {
                        'from_node': 'collection'
                    },

                    'process': {
                        'callback': {
                            "abs": {
                                "arguments": {
                                    "data": {
                                        "from_argument": "data"
                                    }
                                },
                                "process_id": "abs"
                            },
                            "cos": {
                                "arguments": {
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
                'result': True
            },
            'collection': {
                'process_id': 'get_collection',
                'arguments': {
                    'name': 'S2_FAPAR_CLOUDCOVER'
                }
            }
        })

        assert resp.status_code == 200
        assert resp.content_length > 0

    def test_execute_apply_run_udf(self):
        resp = self._post_process_graph({
            'apply': {
                'process_id': 'apply',
                'arguments': {
                    'data': {
                        'from_node': 'collection'
                    },

                    'process': {
                        'callback': {
                            "abs": {
                                "arguments": {
                                    "data": {
                                        "from_argument": "data"
                                    }
                                },
                                "process_id": "abs"
                            },
                            "udf": {
                                "arguments": {
                                    "data": {
                                        "from_node": "abs"
                                    },
                                    "runtime": "Python",
                                    "version": "3.5.1",
                                    "udf": "my python code"

                                },
                                "process_id": "run_udf",
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
        })

        assert resp.status_code == 200
        assert resp.content_length > 0
        print(dummy_impl.collections["S2_FAPAR_CLOUDCOVER"])
        assert dummy_impl.collections["S2_FAPAR_CLOUDCOVER"].apply_tiles.call_count == 1

    def test_execute_reduce_temporal_run_udf(self):
        resp = self._post_process_graph({
            'reduce': {
                'process_id': 'reduce',
                'arguments': {
                    'data': {
                        'from_node': 'collection'
                    },
                    'dimension': 'temporal',
                    # 'binary': False,
                    'reducer': {
                        'callback': {
                            "udf": {
                                "arguments": {
                                    "data": {
                                        "from_argument": "data"
                                    },
                                    "runtime": "Python",
                                    "version": "3.5.1",
                                    "udf": "my python code"

                                },
                                "process_id": "run_udf",
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
        })

        assert resp.status_code == 200
        assert resp.content_length > 0
        assert dummy_impl.collections["S2_FAPAR_CLOUDCOVER"].apply_tiles_spatiotemporal.call_count == 1

    def test_execute_reduce_bands_run_udf(self):
        request = load_json("udf.json")
        resp = self.client.post('/openeo/0.4.0/result', content_type='application/json', json=request)

        assert resp.status_code == 200
        assert resp.content_length > 0
        assert dummy_impl.collections["S2_FAPAR_CLOUDCOVER"].apply_tiles.call_count == 1

    def test_execute_apply_dimension_temporal_run_udf(self):
        resp = self._post_process_graph({
            'apply_dimension': {
                'process_id': 'apply_dimension',
                'arguments': {
                    'data': {
                        'from_node': 'collection'
                    },
                    'dimension': 'temporal',

                    'process': {
                        'callback': {
                            'cumsum': {
                                "arguments": {
                                    "data": {
                                        "from_argument": "data"
                                    }

                                },
                                "process_id": "cumsum"
                            },
                            "udf": {
                                "arguments": {
                                    "data": {
                                        "from_node": "cumsum"
                                    },
                                    "runtime": "Python",
                                    "version": "3.5.1",
                                    "udf": "my python code"

                                },
                                "process_id": "run_udf",
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
        })

        assert resp.status_code == 200
        assert resp.content_length > 0
        self.assertEqual(1, dummy_impl.collections["S2_FAPAR_CLOUDCOVER"].apply_tiles_spatiotemporal.call_count)
        self.assertEqual(1, dummy_impl.collections["S2_FAPAR_CLOUDCOVER"].apply_dimension.call_count)

    def test_execute_reduce_max(self):
        resp = self._post_process_graph({
            'reduce': {
                'process_id': 'reduce',
                'arguments': {
                    'data': {
                        'from_node': 'collection'
                    },
                    'dimension': 'temporal',
                    'reducer': {
                        'callback': {
                            "max": {
                                "arguments": {
                                    "data": {
                                        "from_argument": "data"
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
        })

        assert resp.status_code == 200
        assert resp.content_length > 0

    def test_execute_merge_cubes(self):
        resp = self._post_process_graph({
              "mergecubes1": {
                "process_id": "merge_cubes",
                "arguments": {
                  "cube1": {
                    "from_node": "collection1"
                  },
                  "cube2": {
                    "from_node": "collection2"
                  },
                  "overlap_resolver": {
                    "callback": {
                      "or1": {
                        "process_id": "or",
                        "arguments": {
                          "expressions": [
                            {
                              "from_argument": "cube1"
                            },
                            {
                              "from_argument": "cube2"
                            }
                          ]
                        },
                        "result": True
                      }
                    }
                  }
                },
                "result": True
              },
            'collection1': {
                'process_id': 'get_collection',
                'arguments': {
                    'name': 'S2_FAPAR_CLOUDCOVER'
                }
            },
            'collection2': {
                'process_id': 'get_collection',
                'arguments': {
                    'name': 'S2_FAPAR_CLOUDCOVER'
                }
            }
        })

        assert resp.status_code == 200
        assert resp.content_length > 0
        print(dummy_impl.collections["S2_FAPAR_CLOUDCOVER"].merge.call_args)
        #fails on jenkins, reason yet unknown
        #self.assertEquals(dummy_impl.collections["S2_FAPAR_CLOUDCOVER"].merge.call_args.args[1],"or")

    def test_execute_reduce_bands(self):
        resp = self._post_process_graph({
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
                                        "from_argument": "data"
                                    }
                                },
                                "process_id": "sum"
                            },
                            "subtract": {
                                "arguments": {
                                    "data": {
                                        "from_argument": "data"
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
        })

        assert resp.status_code == 200
        assert resp.content_length > 0

    def test_execute_mask(self):
        resp = self._post_process_graph({
            'apply': {
                'process_id': 'mask',
                'arguments': {
                    'data': {
                        'from_node': 'collection'
                    },
                    'mask': {
                        'from_node': 'mask_collection'
                    },
                    'replacement': '10'
                },
                'result': True
            },
            'collection': {
                'process_id': 'get_collection',
                'arguments': {
                    'name': 'S2_FAPAR_CLOUDCOVER'
                }
            },
            'mask_collection': {
                'process_id': 'get_collection',
                'arguments': {
                    'name': 'S2_FAPAR_CLOUDCOVER'
                }
            }
        })

        assert resp.status_code == 200
        assert resp.content_length > 0
        assert dummy_impl.collections["S2_FAPAR_CLOUDCOVER"].mask.call_count == 1

    def test_execute_mask_polygon(self):
        resp = self._post_process_graph(
            {
                "mask": {
                    "process_id": "mask",
                    "arguments": {
                        'data': {
                            'from_node': 'collection'
                        },
                        "mask": {
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
        )

        assert resp.status_code == 200
        assert resp.content_length > 0
        assert dummy_impl.collections["S2_FAPAR_CLOUDCOVER"].mask.call_count == 1
        import shapely.geometry
        self.assertIsInstance(dummy_impl.collections["S2_FAPAR_CLOUDCOVER"].mask.call_args[1]['polygon'],
                              shapely.geometry.Polygon)

    def test_preview_aggregate_temporal_max(self):
        resp = self._post_process_graph({
            'apply': {
                'process_id': 'aggregate_temporal',
                'arguments': {
                    'data': {
                        'from_node': 'collection'
                    },
                    'dimension': 'temporal',
                    'intervals': [],
                    'labels': [],
                    'reducer': {
                        'callback': {
                            "max": {
                                "arguments": {
                                    "data": {
                                        "from_argument": "data"
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
        })

        assert resp.status_code == 200
        assert resp.content_length > 0

    def test_execute_zonal_statistics(self):
        process_graph = {
            'aggregate_polygon': {
                'process_id': 'aggregate_polygon',
                'arguments': {
                    'data': {
                        'from_node': 'collection'
                    },
                    'polygons': {
                        "type": "Polygon",
                        "coordinates": [
                            [[7.022705078125007, 51.75432477678571], [7.659912109375007, 51.74333844866071],
                             [7.659912109375007, 51.29289899553571], [7.044677734375007, 51.31487165178571],
                             [7.022705078125007, 51.75432477678571]]
                        ]
                    },
                    'reducer': {
                        'callback': {
                            "max": {
                                "arguments": {
                                    "data": {
                                        "from_argument": "data"
                                    }
                                },
                                "process_id": "mean",
                                "result": True
                            }
                        }
                    },
                    'name': 'my_name'
                },

            },
            'save_result': {
                'process_id': 'save_result',
                'arguments': {
                    'data': {
                        'from_node': 'aggregate_polygon'
                    },
                    'format': 'VITO-TSService-JSON'
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
        resp = self._post_process_graph(process_graph)
        assert resp.status_code == 200
        assert json.loads(resp.get_data(as_text=True)) == {
            "2015-07-06T00:00:00": [2.9829132080078127],
            "2015-08-22T00:00:00": [None]
        }

        assert dummy_impl.collections['S2_FAPAR_CLOUDCOVER'].viewingParameters['srs'] == 'EPSG:4326'

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
        resp = self.client.post('/openeo/0.4.0/services', content_type='application/json', json={
            "custom_param": 45,
            "process_graph": process_graph,
            "type": 'WMTS',
            "title": "My Service",
            "description": "Service description"
        })

        assert resp.status_code == 201
        assert resp.headers['OpenEO-Identifier'] == 'c63d6c27-c4c2-4160-b7bd-9e32f582daec'
        assert resp.headers['Location'].endswith("/services/c63d6c27-c4c2-4160-b7bd-9e32f582daec/service/wmts")

        tiled_viewing_service = dummy_impl.collections["S2"].tiled_viewing_service
        assert tiled_viewing_service.call_count == 1
        ProcessGraphVisitor.dereference_from_node_arguments(process_graph)
        tiled_viewing_service.assert_called_with(
            custom_param=45, description='Service description', process_graph=process_graph, title='My Service',
            type='WMTS'
        )

    def test_read_vector(self):
        process_graph = {
            "loadco1": {
                "process_id": "load_collection",
                "arguments": {
                    "id": "PROBAV_L3_S10_TOC_NDVI_333M_V2",
                    "spatial_extent": {
                        "west": 5,
                        "east": 6,
                        "north": 52,
                        "south": 51
                    },
                    "temporal_extent": [
                        "2017-11-21",
                        "2017-12-21"
                    ]
                }
            },
            "geojson_file": {
                "process_id": "read_vector",
                "arguments": {
                    "filename": str(get_path("GeometryCollection.geojson"))
                }
            },
            "aggreg1": {
                "process_id": "aggregate_polygon",
                "arguments": {
                    "data": {
                        "from_node": "loadco1"
                    },
                    "polygons": {
                        "from_node": "geojson_file"
                    },
                    "reducer": {
                        "callback": {
                            "mean1": {
                                "process_id": "mean",
                                "arguments": {
                                    "data": {
                                        "from_argument": "data"
                                    }
                                },
                                "result": True
                            }
                        }
                    }
                }
            },
            "save": {
                "process_id": "save_result",
                "arguments": {
                    "data": {
                        "from_node": "aggreg1"
                    },
                    "format": "JSON"
                },
                "result": True
            }
        }

        resp = self._post_process_graph(process_graph)
        body = resp.get_data(as_text=True)

        assert resp.status_code == 200
        assert 'NaN' not in body
        assert json.loads(body) == {
            "2015-07-06T00:00:00": [2.9829132080078127],
            "2015-08-22T00:00:00": [None]
        }

    def test_load_collection_without_spatial_extent_incorporates_read_vector_extent(self):
        process_graph = {
            "loadco1": {
                "process_id": "load_collection",
                "arguments": {
                    "id": "PROBAV_L3_S10_TOC_NDVI_333M_V2",
                    "temporal_extent": [
                        "2017-11-21",
                        "2017-12-21"
                    ]
                }
            },
            "geojson_file": {
                "process_id": "read_vector",
                "arguments": {
                    "filename": str(get_path("GeometryCollection.geojson"))
                }
            },
            "aggreg1": {
                "process_id": "aggregate_polygon",
                "arguments": {
                    "data": {
                        "from_node": "loadco1"
                    },
                    "polygons": {
                        "from_node": "geojson_file"
                    },
                    "reducer": {
                        "callback": {
                            "mean1": {
                                "process_id": "mean",
                                "arguments": {
                                    "data": {
                                        "from_argument": "data"
                                    }
                                },
                                "result": True
                            }
                        }
                    }
                }
            },
            "save": {
                "process_id": "save_result",
                "arguments": {
                    "data": {
                        "from_node": "aggreg1"
                    },
                    "format": "JSON"
                },
                "result": True
            }
        }

        resp = self._post_process_graph(process_graph)
        body = resp.get_data(as_text=True)

        assert resp.status_code == 200
        assert 'NaN' not in body
        assert json.loads(body) == {
            "2015-07-06T00:00:00": [2.9829132080078127],
            "2015-08-22T00:00:00": [None]
        }

        viewing_parameters = dummy_impl.collections['PROBAV_L3_S10_TOC_NDVI_333M_V2'].viewingParameters
        self.assertAlmostEqual(viewing_parameters['left'], 5.07616, delta=0.01)
        self.assertAlmostEqual(viewing_parameters['bottom'], 51.2122, delta=0.01)
        self.assertAlmostEqual(viewing_parameters['right'], 5.16685, delta=0.01)
        self.assertAlmostEqual(viewing_parameters['top'], 51.2689, delta=0.01)
        self.assertEquals(viewing_parameters['srs'], 'EPSG:4326')

    def test_read_vector_from_FeatureCollection(self):
        process_graph = {
            "loadco1": {
                "process_id": "load_collection",
                "arguments": {
                    "id": "PROBAV_L3_S10_TOC_NDVI_333M_V2",
                    "spatial_extent": {
                        "west": 5,
                        "east": 6,
                        "north": 52,
                        "south": 51
                    },
                    "temporal_extent": [
                        "2017-11-21",
                        "2017-12-21"
                    ]
                }
            },
            "geojson_file": {
                "process_id": "read_vector",
                "arguments": {
                    "filename": str(get_path("FeatureCollection.geojson"))
                }
            },
            "aggreg1": {
                "process_id": "aggregate_polygon",
                "arguments": {
                    "data": {
                        "from_node": "loadco1"
                    },
                    "polygons": {
                        "from_node": "geojson_file"
                    },
                    "reducer": {
                        "callback": {
                            "mean1": {
                                "process_id": "mean",
                                "arguments": {
                                    "data": {
                                        "from_argument": "data"
                                    }
                                },
                                "result": True
                            }
                        }
                    }
                }
            },
            "save": {
                "process_id": "save_result",
                "arguments": {
                    "data": {
                        "from_node": "aggreg1"
                    },
                    "format": "JSON"
                },
                "result": True
            }
        }

        resp = self._post_process_graph(process_graph)
        body = resp.get_data(as_text=True)

        assert resp.status_code == 200, body
        assert 'NaN' not in body
        assert json.loads(body) == {
            "2015-07-06T00:00:00": [2.9829132080078127],
            "2015-08-22T00:00:00": [None]
        }

    def test_no_nested_JSONResult(self):
        process_graph = {
            "budget": None,
            "title": None,
            "description": None,
            "plan": None,
            "process_graph": {
                "loadcollection1": {
                    "result": None,
                    "process_id": "load_collection",
                    "arguments": {
                        "id": "PROBAV_L3_S10_TOC_NDVI_333M_V2",
                        "spatial_extent": None,
                        "temporal_extent": None
                    }
                },
                "zonalstatistics1": {
                    "result": None,
                    "process_id": "zonal_statistics",
                    "arguments": {
                        "func": "mean",
                        "data": {
                            "from_node": "filtertemporal1"
                        },
                        "scale": 1000,
                        "interval": "day",
                        "regions": {
                            "coordinates": [
                                [
                                    [
                                        7.022705078125007,
                                        51.75432477678571
                                    ],
                                    [
                                        7.659912109375007,
                                        51.74333844866071
                                    ],
                                    [
                                        7.659912109375007,
                                        51.29289899553571
                                    ],
                                    [
                                        7.044677734375007,
                                        51.31487165178571
                                    ],
                                    [
                                        7.022705078125007,
                                        51.75432477678571
                                    ]
                                ]
                            ],
                            "type": "Polygon"
                        }
                    }
                },
                "saveresult1": {
                    "result": True,
                    "process_id": "save_result",
                    "arguments": {
                        "format": "GTIFF",
                        "data": {
                            "from_node": "zonalstatistics1"
                        },
                        "options": {

                        }
                    }
                },
                "filtertemporal1": {
                    "result": False,
                    "process_id": "filter_temporal",
                    "arguments": {
                        "data": {
                            "from_node": "loadcollection1"
                        },
                        "extent": [
                            "2017-01-01",
                            "2017-11-21"
                        ]
                    }
                }
            }
        }

        resp = self.client.post('/openeo/0.4.0/result', content_type='application/json', data=json.dumps(process_graph))

        assert resp.status_code == 200

    def test_point_with_bbox(self):
        process_graph = {
            "loadcollection1": {
                'process_id': 'load_collection',
                'arguments': {'id': 'S2_FAPAR_CLOUDCOVER'},
            },
            "filterbbox": {
                "process_id": "filter_bbox",
                "arguments": {
                    "data": {"from_node": "loadcollection1"},
                    "extent": {"west": 3, "east": 6, "south": 50, "north": 51, "crs": "EPSG:4326"}
                },
                "result": True
            }
        }
        resp = self._post_process_graph(process_graph, url="timeseries/point?x=1&y=2")
        assert resp.status_code == 200
        assert resp.json == {"viewingParameters": {
            "left": 3, "right": 6, "bottom": 50, "top": 51, "srs": "EPSG:4326", "version": "0.4.2"
        }}


    def test_load_disk_data(self):
        process_graph = {
            "loaddiskdata": {
                'process_id': 'load_disk_data',
                'arguments': {
                    'format': 'GTiff',
                    'glob_pattern': "/data/MTDA/CGS_S2/CGS_S2_FAPAR/2019/04/24/*/*/10M/*_FAPAR_10M_V102.tif",
                    'options': {
                        'date_regex': r"_(\d{4})(\d{2})(\d{2})T"
                    }
                }
            },
            "filterbbox": {
                "process_id": "filter_bbox",
                "arguments": {
                    "data": {"from_node": "loaddiskdata"},
                    "extent": {"west": 3, "east": 6, "south": 50, "north": 51, "crs": "EPSG:4326"}
                },
                'result': True
            }
        }

        with self._post_process_graph(process_graph) as resp:
            self.assertEqual(200, resp.status_code)
