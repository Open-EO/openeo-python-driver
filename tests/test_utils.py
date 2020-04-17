from datetime import datetime

from openeo_driver.utils import smart_bool, parse_rfc3339


def test_smart_bool():
    for value in [1, 100, [1], (1,), "x", "1", "on", "ON", "yes", "YES", "true", "True", "TRUE", ]:
        assert smart_bool(value) == True
    for value in [0, [], (), {}, False, "0", "off", "OFF", "no", "No", "NO", "False", "false", "FALSE"]:
        assert smart_bool(value) == False


def test_parse_rfc3339():
    assert parse_rfc3339("2017-02-01T19:32:12Z") == datetime(2017, 2, 1, 19, 32, 12)
