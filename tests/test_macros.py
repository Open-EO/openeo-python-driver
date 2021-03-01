from openeo_driver.macros import expand_macros


def test_ard_normalized_radar_backscatter():
    process_graph = {
        "loadcollection1": {
            "process_id": "load_collection",
            "arguments": {
                "id": "SENTINEL1_GAMMA0_SENTINELHUB"
            }
        },
        "ardnormalizedradarbackscatter1": {
            "process_id": "ard_normalized_radar_backscatter",
            "arguments": {
                "data": {"from_node": "loadcollection1"},
                "elevation_model": "MAPZEN",
                "ellipsoid_incidence_angle": True,
                "noise_removal": True
            },
            "result": True
        }
    }

    assert expand_macros(process_graph) == {
        "loadcollection1": {
            "process_id": "load_collection",
            "arguments": {
                "id": "SENTINEL1_GAMMA0_SENTINELHUB"
            }
        },
        "ardnormalizedradarbackscatter1": {
            "process_id": "sar_backscatter",
            "arguments": {
                "data": {"from_node": "loadcollection1"},
                "orthorectify": True,
                "rtc": True,
                "elevation_model": "MAPZEN",
                "mask": True,
                "contributing_area": True,
                "local_incidence_angle": True,
                "ellipsoid_incidence_angle": True,
                "noise_removal": True
            },
            "result": True
        }
    }


def test_ard_normalized_radar_backscatter_without_optional_arguments():
    process_graph = {
        "loadcollection1": {
            "process_id": "load_collection",
            "arguments": {
                "id": "SENTINEL1_GAMMA0_SENTINELHUB"
            }
        },
        "ardnormalizedradarbackscatter1": {
            "process_id": "ard_normalized_radar_backscatter",
            "arguments": {
                "data": {"from_node": "loadcollection1"}
            },
            "result": True
        }
    }

    assert expand_macros(process_graph) == {
        "loadcollection1": {
            "process_id": "load_collection",
            "arguments": {
                "id": "SENTINEL1_GAMMA0_SENTINELHUB"
            }
        },
        "ardnormalizedradarbackscatter1": {
            "process_id": "sar_backscatter",
            "arguments": {
                "data": {"from_node": "loadcollection1"},
                "orthorectify": True,
                "rtc": True,
                "elevation_model": None,
                "mask": True,
                "contributing_area": True,
                "local_incidence_angle": True,
                "ellipsoid_incidence_angle": False,
                "noise_removal": True
            },
            "result": True
        }
    }
