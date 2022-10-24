import json
import logging
from pathlib import Path
from typing import Union, Tuple, Optional, Sequence, List

import pyproj
import shapely.geometry
import shapely.ops
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry

from openeo_driver.util.utm import auto_utm_epsg

_log = logging.getLogger(__name__)


def geojson_to_geometry(geojson: dict) -> BaseGeometry:
    """Convert GeoJSON object to shapely geometry object"""
    # TODO #71 #114 EP-3981 standardize on using (FeatureCollection like) vector cubes  instead of GeometryCollection?
    if geojson["type"] == "FeatureCollection":
        geojson = {
            "type": "GeometryCollection",
            "geometries": [feature["geometry"] for feature in geojson["features"]],
        }
    elif geojson["type"] == "Feature":
        geojson = geojson["geometry"]
    try:
        return shapely.geometry.shape(geojson)
    except Exception as e:
        _log.error(e, exc_info=True)
        _log.error(f"Invalid geojson: {json.dumps(geojson)}")
        raise ValueError(
            f"Invalid geojson object, the shapely library generated this error: {str(e)}. When trying to parse your geojson."
        )


def geojson_to_multipolygon(
    geojson: dict,
) -> Union[MultiPolygon, Polygon]:
    """
    Convert GeoJSON object (dict) to shapely MultiPolygon (or Polygon where possible/allowed; in particular, this
    means dissolving overlapping polygons into one).
    """
    # TODO: option to also force conversion of Polygon to MultiPolygon?
    # TODO: #71 #114 migrate/centralize all this kind of logic to vector cubes
    if geojson["type"] == "Feature":
        geojson = geojson["geometry"]

    if geojson["type"] in ("MultiPolygon", "Polygon"):
        geometry = shapely.geometry.shape(geojson)
    elif geojson["type"] == "GeometryCollection":
        geometry = shapely.ops.unary_union(shapely.geometry.shape(geojson).geoms)
    elif geojson["type"] == "FeatureCollection":
        geometry = shapely.ops.unary_union(
            [shapely.geometry.shape(f["geometry"]) for f in geojson["features"]]
        )
    else:
        raise ValueError(
            f"Invalid GeoJSON type for MultiPolygon conversion: {geojson['type']}"
        )

    if not isinstance(geometry, (MultiPolygon, Polygon)):
        raise ValueError(f"Failed to convert to MultiPolygon ({geojson['type']})")

    return geometry


def reproject_bounding_box(bbox: dict, from_crs: str, to_crs: str) -> dict:
    """
    Reproject given bounding box dictionary

    :param bbox: bbox dict with fields "west", "south", "east", "north"
    :param from_crs: source CRS. Specify `None` to use the "crs" field of input bbox dict
    :param to_crs: target CRS
    :return: bbox dict (fields "west", "south", "east", "north", "crs")
    """
    box = shapely.geometry.box(bbox["west"], bbox["south"], bbox["east"], bbox["north"])
    if from_crs is None:
        from_crs = bbox["crs"]
    tranformer = pyproj.Transformer.from_crs(
        crs_from=from_crs, crs_to=to_crs, always_xy=True
    )
    reprojected = shapely.ops.transform(tranformer.transform, box)
    return dict(zip(["west", "south", "east", "north"], reprojected.bounds), crs=to_crs)


def spatial_extent_union(*bboxes: dict, default_crs="EPSG:4326") -> dict:
    """
    Calculate spatial bbox covering all given bounding boxes

    :return: bbox dict (fields "west", "south", "east", "north", "crs")
    """
    assert len(bboxes) >= 1
    crss = set(b.get("crs", default_crs) for b in bboxes)
    if len(crss) > 1:
        # Re-project to CRS of first bbox
        def reproject(bbox, to_crs):
            from_crs = bbox.get("crs", default_crs)
            if from_crs != to_crs:
                return reproject_bounding_box(bbox, from_crs=from_crs, to_crs=to_crs)
            return bbox

        to_crs = bboxes[0].get("crs", default_crs)
        bboxes = [reproject(b, to_crs=to_crs) for b in bboxes]
        crs = to_crs
    else:
        crs = crss.pop()
    bbox = {
        "west": min(b["west"] for b in bboxes),
        "south": min(b["south"] for b in bboxes),
        "east": max(b["east"] for b in bboxes),
        "north": max(b["north"] for b in bboxes),
        "crs": crs,
    }
    return bbox


class GeometryBufferer:
    """
    Geometry buffering helper.

    For example, for buffering 100m when working with lonlat geometries:
    >>> bufferer = GeometryBufferer.from_meter_for_crs(distance=100, crs="EPSG:4326")
    >>> buffered_geometry = bufferer.buffer(geometry)

    """

    __slots__ = ["distance", "resolution"]

    def __init__(self, distance: float, resolution=2):
        self.distance = distance
        self.resolution = resolution

    def buffer(self, geometry: BaseGeometry):
        return geometry.buffer(distance=self.distance, resolution=self.resolution)

    @classmethod
    def from_meter_for_crs(
        cls,
        distance: float = 1.0,
        crs: Union[str, pyproj.CRS] = "EPSG:4326",
        loi: Tuple[float, float] = (0, 0),
        loi_crs: Union[str, pyproj.CRS, None] = None,
        resolution=2,
    ) -> "GeometryBufferer":
        """
        Build bufferer for buffering given distance in meter of geometries in given CRS.

        :param crs: the target CRS to express the distance in (as easting)
        :param distance: distance in meter
        :param loi: location of interest where the distance will be used for buffering
            (e.g. lon-lat coordinates of center of AOI).
            Latitude is for example important to give a better approximation of the transformed distance.
        :param loi_crs: CRS used for the `loi` coordinates (target `crs` will be used by default)
        """
        distance = cls.transform_meter_to_crs(
            crs=crs, distance=distance, loi=loi, loi_crs=loi_crs
        )
        return cls(distance=distance, resolution=resolution)

    @staticmethod
    def transform_meter_to_crs(
        distance: float = 1.0,
        crs: Union[str, pyproj.CRS] = "EPSG:4326",
        loi: Tuple[float, float] = (0, 0),
        loi_crs: Union[str, pyproj.CRS, None] = None,
    ) -> float:
        """
        (Approximate) reproject a distance in meter to a different CRS.

        :param crs: the target CRS to express the distance in (as easting)
        :param distance: distance in meter
        :param loi: location of interest where the distance will be used for buffering
            (e.g. lon-lat coordinates of center of AOI).
            Latitude is for example important to give a better approximation of the transformed distance.
        :param loi_crs: CRS used for the `loi` coordinates (target `crs` will be used by default)

        :return: distance in target CRS (easting)
        """
        proj_target = pyproj.Proj(crs)
        proj_operation = pyproj.Proj(loi_crs or crs)

        # Determine UTM zone and projection for location of operation
        operation_loc_lonlat = pyproj.Transformer.from_proj(
            proj_from=proj_operation, proj_to=pyproj.Proj("EPSG:4326"), always_xy=True
        ).transform(*loi)
        utm_zone = auto_utm_epsg(*operation_loc_lonlat)
        proj_utm = pyproj.Proj(f"EPSG:{utm_zone}")

        # Transform location of operation to UTM
        x_utm, y_utm = pyproj.Transformer.from_proj(
            proj_from=proj_operation, proj_to=proj_utm, always_xy=True
        ).transform(*loi)
        # Transform distance from UTM to target CRS
        utm2crs = pyproj.Transformer.from_proj(proj_utm, proj_target, always_xy=True)
        lon0, lat0 = utm2crs.transform(x_utm, y_utm)
        lon1, lat1 = utm2crs.transform(x_utm + distance, y_utm)
        # TODO: check/incorporate lat0 and lat1?
        # TODO: take average, or max of longitude delta and latitude delta?
        if abs(lat1 - lat0) > 1e-3 * abs(lon1 - lon0):
            _log.warning(
                f"transform_meters_to_crs: large latitude delta: ({lon0},{lat0})-({lon1},{lat1})"
            )
        return abs(lon1 - lon0)


def as_geojson_feature(
    geometry: Union[dict, BaseGeometry, Path],
    properties: Union[dict, None] = None,
) -> dict:
    """
    Helper to construct a GeoJSON-style Feature dictionary from a shapely geometry,
    a GeoJSON-style geometry/feature dictionary, a path to a GeoJSON file, ...

    :param geometry: a shapely geometry, a geometry as GeoJSON-style dict, a feature as GeoJSON-style dict, a path to a GeoJSON file
    :param properties: properties to set on the feature
    :return: GeoJSON-style Feature dict
    """
    # TODO: support loading WKT string?
    if isinstance(geometry, Path):
        # Assume GeoJSON file
        geometry = json.loads(geometry.read_text(encoding="utf-8"))
    elif not isinstance(geometry, dict):
        geometry = shapely.geometry.mapping(geometry)
    assert "type" in geometry
    if geometry["type"] == "Feature":
        properties = properties or geometry["properties"]
        geometry = geometry["geometry"]
    assert isinstance(geometry, dict)
    assert geometry["type"] in [
        "Point",
        "MultiPoint",
        "LineString",
        "MultiLineString",
        "Polygon",
        "MultiPolygon",
        "GeometryCollection",
    ]
    return {"type": "Feature", "geometry": geometry, "properties": properties}


def as_geojson_feature_collection(
    *features: Union[dict, BaseGeometry, Path, List[Union[dict, BaseGeometry, Path]]]
) -> dict:
    """
    Helper to construct a GeoJSON-style FeatureCollection from feature like objects
    :param features: zero, one or more of a shapely geometry, a geometry as GeoJSON-style dict, a feature as GeoJSON-style dict, a path to a GeoJSON file

    :return: GeoJSON-style FeatureCollection dict
    """
    # TODO: some way of setting properties for all/each feature?
    # TODO: support a single arg with `"type": "FeatureCollection"`?
    if len(features) == 1 and isinstance(features[0], list):
        # Support providing multiple features as a single list (instead of *args)
        features = features[0]
    return {
        "type": "FeatureCollection",
        "features": [as_geojson_feature(f) for f in features],
    }
