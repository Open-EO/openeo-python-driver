"""
Mapping file format  (e.g. GDAL Raster Formats and OGR Vector Formats) to
file extension, mimetype, fiona driver
"""
from typing import Optional, Iterable, Dict


class FormatInfo:
    """Simple container of input/output format information: format code, mimetype, ..."""
    __slots__ = ["format", "mimetype", "extension", "fiona_driver", "multi_file"]

    def __init__(
            self, format: str, mimetype: str,
            extension: Optional[str] = None, fiona_driver: Optional[str] = None, multi_file: bool = False
    ):
        self.format = format
        self.mimetype = mimetype
        self.extension = extension or f"{format.lower()}"
        self.fiona_driver = fiona_driver or format
        self.multi_file = multi_file


class _FormatDb:
    """Case-insensitive `FormatInfo` lookup"""

    def __init__(self, formats: Iterable[FormatInfo]):
        # TODO: also support aliases for lookup?
        self.db: Dict[str, FormatInfo] = {f.format.lower(): f for f in formats}

    def get(self, format: str) -> FormatInfo:
        """Case-insensitive format lookup"""
        return self.db[format.lower()]

    def __contains__(self, format: str) -> bool:
        return format.lower() in self.db

    def get_mimetype(self, format: str, default="application/octet-stream") -> str:
        if format in self:
            return self.get(format).mimetype
        else:
            return default


IOFORMATS = _FormatDb([
    FormatInfo("GTiff", "image/tiff; application=geotiff", extension="geotiff"),
    FormatInfo("COG", "image/tiff; application=geotiff; profile=cloud-optimized"),
    FormatInfo("NetCDF", "application/x-netcdf", extension="nc"),
    FormatInfo("PNG", "image/png"),
    FormatInfo("JSON", "application/json"),
    FormatInfo("GeoJSON", "application/geo+json"),
    FormatInfo("CovJSON", "application/json"),
    FormatInfo("CSV", "text/csv"),
    FormatInfo("ESRI Shapefile", "x-gis/x-shapefile", extension="shp", multi_file=True),
    FormatInfo("GPKG", "application/geopackage+sqlite3"),
    # TODO: support more formats
])
