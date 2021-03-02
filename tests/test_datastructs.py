from openeo_driver.datastructs import SarBackscatterArgs


def test_sar_backscatter_defaults():
    s = SarBackscatterArgs()
    assert s.coefficient == "gamma0-terrain"
    assert s.elevation_model is None
    assert s.mask is False
    assert s.options == {}


def test_sar_backscatter_coefficient_none():
    s = SarBackscatterArgs(coefficient=None)
    assert s.coefficient is None
