import logging
from typing import Tuple, Union, Any

import math
import pyproj
import shapely.ops
from pyproj import Geod, CRS
from shapely.geometry.base import BaseGeometry

_log = logging.getLogger(__name__)


def auto_utm_epsg(lon: float, lat: float) -> int:
    """
    Get EPSG code of UTM zone containing given long-lat coordinates
    """
    # Use longitude to determine zone 'band'
    zone = (math.floor((lon + 180.0) / 6.0) % 60) + 1

    # Use latitude to determine north/south
    if lat >= 0.0:
        epsg = 32600 + zone
    else:
        epsg = 32700 + zone

    return epsg


def is_auto_utm_crs(crs: Any) -> bool:
    """
    Is given CRS identifier the special Auto-UTM CRS ("AUTO:42001")?
    """
    return crs == "AUTO:42001" or "Auto42001" in str(crs)


def _is_utm_epsg(epsg: int) -> bool:
    return (32601 <= epsg <= 32660) or (32701 <= epsg <= 32760)


def is_utm_crs(crs: Any) -> int:
    """
    Is given CRS identifier a UTM CRS?

    Returns corresponding EPSG code (as int) if it is a UTM CRS.
    Returns False (which is technically also an int) otherwise.
    """
    try:
        epsg = pyproj.CRS.from_user_input(crs).to_epsg() or 0
    except pyproj.exceptions.CRSError:
        epsg = 0
    return epsg if _is_utm_epsg(epsg) else False


def utm_zone_from_epsg(epsg: int) -> Tuple[int, bool]:
    """Get `(utm_zone, is_northern_hemisphere)` from given EPSG code."""
    if not (32601 <= epsg <= 32660 or 32701 <= epsg <= 32760):
        raise ValueError("Can not convert EPSG {e} to UTM zone".format(e=epsg))
    return epsg % 100, epsg < 32700


def auto_utm_epsg_for_geometry(geometry: BaseGeometry, crs: str = "EPSG:4326") -> int:
    """
    Get EPSG code of best UTM zone for given geometry.
    """
    # Pick a geometry coordinate
    p = geometry.representative_point()
    x = p.x
    y = p.y

    # If needed, convert it to lon/lat (WGS84)
    if CRS.from_user_input(crs).to_epsg() != 4326:
        transformer = pyproj.Transformer.from_crs(crs_from=crs, crs_to="EPSG:4326", always_xy=True)
        x, y = transformer.transform(x, y)

    # And derive the EPSG code
    return auto_utm_epsg(x, y)


def auto_utm_crs_for_geometry(geometry: BaseGeometry, crs: str) -> str:
    epsg = auto_utm_epsg_for_geometry(geometry, crs)
    return 'epsg:' + str(epsg)


def geometry_to_crs(geometry: BaseGeometry, crs_from, crs_to):
    # TODO: This reprojection differs from qgis, especially at the poles.
    # Skip if CRS definitions are exactly the same
    if crs_from == crs_to:
        return geometry

    # Construct a function for projecting coordinates
    proj_from = pyproj.Proj(crs_from)
    proj_to = pyproj.Proj(crs_to)

    def project(x, y, z=0):
        transformer = pyproj.Transformer.from_crs(crs_from, crs_to, always_xy=True)
        return transformer.transform(x, y)

    # And apply to all coordinates in the geometry
    return shapely.ops.transform(project, geometry)


def area_in_square_meters(geometry: BaseGeometry, crs: Union[str, pyproj.CRS]):
    """
    Calculate the area of a geometry in square meters, the area is calculated using a curved surface (WGS84 ellipsoid).

    :param geometry: The geometry to calculate the area for.
    :param crs: The CRS of the geometry.

    :return: The area in square meters.
    """
    if isinstance(crs, str):
        crs = pyproj.crs.CRS.from_user_input(crs)

    geometry_lat_lon = geometry_to_crs(geometry, crs, pyproj.crs.CRS.from_epsg(4326))
    geod = Geod(ellps="WGS84")
    area = abs(geod.geometry_area_perimeter(geometry_lat_lon)[0])
    return area
