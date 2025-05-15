from openeo_driver.util.stac import sniff_stac_extension_prefix


import pytest


@pytest.mark.parametrize(
    ["data", "prefix", "expected"],
    [
        ({}, "eo:", False),
        ([], "eo:", False),
        (123, "eo:", False),
        ("foobar", "eo:", False),
        ("eo:foobar", "eo:", False),
        ([123, "eo:foobar"], "eo:", False),
        ({123: "eo:foobar"}, "eo:", False),
        ({"bands": [{"name": "red"}]}, "eo:", False),
        ({"eo:bands": [{"name": "red"}]}, "eo:", True),
        ({"bands": [{"name": "red", "eo:center_wavelength": 123}]}, "eo:", True),
        ([{"eo:bands": [{"name": "red"}]}, {"eo:bands": [{"name": "green"}]}], "eo:", True),
        (({"eo:bands": [{"name": "red"}]}, {"eo:bands": [{"name": "green"}]}), "eo:", True),
        (
            {1: {"eo:bands": [{"name": "R"}]}, 2: {"eo:bands": [{"name": "G"}]}},
            "eo:",
            True,
        ),
        (
            # Support passing a `dict.values()` view too
            {1: {"eo:bands": [{"name": "R"}]}, 2: {"eo:bands": [{"name": "G"}]}}.values(),
            "eo:",
            True,
        ),
    ],
)
def test_sniff_extension_prefix(data, prefix, expected):
    assert sniff_stac_extension_prefix(data, prefix=prefix) == expected
