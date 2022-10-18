import json
import logging
from typing import Union

import pyproj
import shapely.geometry
import shapely.ops
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.geometry.base import CAP_STYLE, BaseGeometry

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


def buffer_point_approx(
    point: Point, point_crs: str, buffer_distance_in_meters=10.0
) -> Polygon:
    # TODO: default buffer distance of 10m assumes certain resolution (e.g. sentinel2 pixels)
    # TODO way to set buffer distance directly from collection resolution metadata?
    # TODO: often, a lot of points have to be buffered, with this per-point implementation a lot of
    #       the exact same preparation is done repetitively
    src_proj = pyproj.Proj("EPSG:3857")
    dst_proj = pyproj.Proj(point_crs)

    def reproject_point(x, y):
        return pyproj.transform(src_proj, dst_proj, x, y, always_xy=True)

    # this is too approximate in a lot of cases, better to use utm zones?
    left, _ = reproject_point(0.0, 0.0)
    right, _ = reproject_point(buffer_distance_in_meters, 0.0)

    buffer_distance = right - left

    return point.buffer(buffer_distance, resolution=2, cap_style=CAP_STYLE.square)
