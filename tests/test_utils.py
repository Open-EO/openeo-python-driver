from openeo_driver.utils import smart_bool


def test_smart_bool():
    for value in [1, 100, [1], (1,), "x", "1", "on", "ON", "yes", "YES", "true", "True", "TRUE", ]:
        assert smart_bool(value) == True
    for value in [0, [], (), {}, False, "0", "off", "OFF", "no", "No", "NO", "False", "false", "FALSE"]:
        assert smart_bool(value) == False
