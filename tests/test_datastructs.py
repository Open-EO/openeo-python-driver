import typing

from openeo_driver.datastructs import SarBackscatterArgs, secretive_repr


def test_sar_backscatter_defaults():
    s = SarBackscatterArgs()
    assert s.coefficient == "gamma0-terrain"
    assert s.elevation_model is None
    assert s.mask is False
    assert s.options == {}


def test_sar_backscatter_coefficient_none():
    s = SarBackscatterArgs(coefficient=None)
    assert s.coefficient is None


def test_secretive_repr():
    class Credentials(typing.NamedTuple):
        klient_id: str
        da_secret: str
        size: int
        __repr__ = __str__ = secretive_repr()

    credentials = Credentials(klient_id="lookatme", da_secret="forbidden", size=4)

    expected = "Credentials(klient_id='lookatme', da_secret='***', size=4)"
    assert repr(credentials) == expected
    assert str(credentials) == expected
    assert credentials.da_secret == "forbidden"
