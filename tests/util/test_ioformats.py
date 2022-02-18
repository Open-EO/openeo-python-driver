import pytest

from openeo_driver.util.ioformats import IOFORMATS


@pytest.mark.parametrize("format", ["gtiff", "GTIFF", "GTiff"])
def test_ioformats_gtiff(format):
    format_info = IOFORMATS.get(format)
    assert format_info.format == "GTiff"
    assert format_info.mimetype == "image/tiff; application=geotiff"
    assert format_info.extension == "geotiff"
    assert format_info.fiona_driver == "GTiff"


@pytest.mark.parametrize("format", ["netcdf", "NETCDF", "NetCDF"])
def test_ioformats_netcdf(format):
    format_info = IOFORMATS.get(format)
    assert format_info.format == "NetCDF"
    assert format_info.mimetype == "application/x-netcdf"
    assert format_info.extension == "nc"
    assert format_info.fiona_driver == "NetCDF"


def test_ioformats_get_mimetype():
    assert IOFORMATS.get_mimetype("PNG") == "image/png"
    assert IOFORMATS.get_mimetype("obscure format") == "application/octet-stream"
