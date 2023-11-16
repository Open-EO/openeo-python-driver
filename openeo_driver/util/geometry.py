import dataclasses
import json
import logging
import re
from pathlib import Path
from typing import Any, Collection, List, Mapping, Optional, Sequence, Tuple, Union

import pyproj
import shapely.geometry
import shapely.ops
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

from openeo.util import repr_truncate
from openeo_driver.errors import OpenEOApiException
from openeo_driver.util.utm import auto_utm_epsg, auto_utm_epsg_for_geometry

_log = logging.getLogger(__name__)


GEOJSON_GEOMETRY_TYPES_BASIC = frozenset(
    {"Point", "MultiPoint", "LineString", "MultiLineString", "Polygon", "MultiPolygon"}
)
GEOJSON_GEOMETRY_TYPES_EXTENDED = GEOJSON_GEOMETRY_TYPES_BASIC | {"GeometryCollection"}


def validate_geojson_basic(
    value: Any,
    *,
    allowed_types: Optional[Collection[str]] = None,
    raise_exception: bool = True,
    recurse: bool = True,
) -> List[str]:
    """
    Validate if given value looks like a valid GeoJSON construct.

    Note: this is just for basic inspection to catch simple/obvious structural issues.
    It is not intended for a full-blown, deep GeoJSON validation and coordinate inspection.

    :param value: the value to inspect
    :param allowed_types: optional collection of GeoJSON types to accept
    :param raise_exception: whether to raise an exception when issues are found (default),
        or just return list of issues
    :param recurse: whether to recursively validate Feature's geometry and FeatureCollection's features
    :returns: list of issues found (when `raise_exception` is off)
    """
    try:
        if not isinstance(value, dict):
            raise ValueError(f"JSON object (mapping/dictionary) expected, but got {type(value).__name__}")
        assert "type" in value, "No 'type' field"
        geojson_type = value["type"]
        assert isinstance(geojson_type, str), f"Invalid 'type' type: {type(geojson_type).__name__}"
        if allowed_types and geojson_type not in allowed_types:
            raise ValueError(f"Found type {geojson_type!r}, but expects one of {sorted(allowed_types)}")
        if geojson_type in GEOJSON_GEOMETRY_TYPES_BASIC:
            assert "coordinates" in value, f"No 'coordinates' field (type {geojson_type!r})"
        elif geojson_type in {"GeometryCollection"}:
            assert "geometries" in value, f"No 'geometries' field (type {geojson_type!r})"
            # TODO: recursively check sub-geometries?
        elif geojson_type in {"Feature"}:
            assert "geometry" in value, f"No 'geometry' field (type {geojson_type!r})"
            assert "properties" in value, f"No 'properties' field (type {geojson_type!r})"
            if recurse:
                validate_geojson_basic(
                    value["geometry"], recurse=True, allowed_types=GEOJSON_GEOMETRY_TYPES_EXTENDED, raise_exception=True
                )
        elif geojson_type in {"FeatureCollection"}:
            assert "features" in value, f"No 'features' field (type {geojson_type!r})"
            if recurse:
                for f in value["features"]:
                    validate_geojson_basic(f, recurse=True, allowed_types=["Feature"], raise_exception=True)
        else:
            raise ValueError(f"Invalid type {geojson_type!r}")

    except Exception as e:
        if raise_exception:
            raise
        return [str(e)]
    return []


def validate_geojson_coordinates(geojson: dict):
    def _validate_coordinates(coordinates, initial_run=True):
        max_evaluations = 20
        message = f"Failed to parse Geojson. Coordinates are invalid."
        if not isinstance(coordinates, (list, Tuple)) or len(coordinates) == 0:
            raise OpenEOApiException(status_code=400, message=message)
        if not isinstance(coordinates[0], (float, int)):
            # Flatten until elements are floats or ints.
            eval_count = 0
            for sub in coordinates:
                eval_count += _validate_coordinates(sub, False)
                if eval_count > max_evaluations:
                    break
            return eval_count
        if len(coordinates) < 2:
            raise OpenEOApiException(status_code=400, message=message)
        if not (-180 <= coordinates[0] <= 180 and -90 <= coordinates[1] <= 90):
            message = (
                f"Failed to parse Geojson. Invalid coordinate: {coordinates}. "
                f"X value must be between -180 and 180, Y value must be between -90 and 90."
            )
            raise OpenEOApiException(status_code=400, message=message)
        return 1

    def _validate_geometry_collection(geojson):
        if geojson["type"] != "GeometryCollection":
            return
        for geometry in geojson["geometries"]:
            if geometry["type"] == "GeometryCollection":
                _validate_geometry_collection(geometry)
            else:
                _validate_coordinates(geometry["coordinates"])

    def _validate_feature_collection(geojson):
        if geojson["type"] != "FeatureCollection":
            return
        for feature in geojson["features"]:
            if feature["geometry"]["type"] == "GeometryCollection":
                _validate_geometry_collection(feature["geometry"])
            else:
                _validate_coordinates(feature["geometry"]["coordinates"])

    if not isinstance(geojson, dict):
        raise ValueError(f"Invalid GeoJSON: not a dict: {repr_truncate(geojson)}")
    if "type" not in geojson:
        raise ValueError(f"Invalid GeoJSON: missing 'type' field: {repr_truncate(geojson)}")

    if geojson["type"] == "Feature":
        geojson = geojson["geometry"]

    if geojson["type"] == "GeometryCollection":
        _validate_geometry_collection(geojson)
    elif geojson["type"] == "FeatureCollection":
        _validate_feature_collection(geojson)
    else:
        _validate_coordinates(geojson["coordinates"])


def geojson_to_geometry(geojson: dict) -> BaseGeometry:
    """Convert GeoJSON object to shapely geometry object"""
    # TODO #71 #114 EP-3981 standardize on using (FeatureCollection like) vector cubes  instead of GeometryCollection?
    validate_geojson_coordinates(geojson)
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
    validate_geojson_coordinates(geojson)
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


def reproject_bounding_box(bbox: dict, from_crs: Optional[str], to_crs: str) -> dict:
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


def reproject_geometry(
    geometry: BaseGeometry,
    from_crs: Union[pyproj.CRS, str],
    to_crs: Union[pyproj.CRS, str],
) -> BaseGeometry:
    """
    Reproject given shapely geometry

    :param geometry: shapely geometry
    :param from_crs: source CRS
    :param to_crs: target CRS
    :return: reprojected shapely geometry
    """
    transformer = pyproj.Transformer.from_crs(
        crs_from=from_crs, crs_to=to_crs, always_xy=True
    )
    return shapely.ops.transform(transformer.transform, geometry)


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


class BoundingBoxException(ValueError):
    pass


class CrsRequired(BoundingBoxException):
    pass


@dataclasses.dataclass(frozen=True)
class BoundingBox:
    """
    Bounding box with west, south, east, north coordinates
    optionally (geo)referenced.
    """

    west: float
    south: float
    east: float
    north: float
    crs: Optional[str] = dataclasses.field()

    def __init__(
        self,
        west: float,
        south: float,
        east: float,
        north: float,
        *,
        crs: Optional[Union[str, int]] = None,
    ):
        missing = [
            k
            for k, v in zip(
                ("west", "south", "east", "north"), (west, south, east, north)
            )
            if v is None
        ]
        if missing:
            raise BoundingBoxException(f"Missing bounds: {missing}.")
        # __setattr__ workaround to initialize read-only attributes
        super().__setattr__("west", west)
        super().__setattr__("south", south)
        super().__setattr__("east", east)
        super().__setattr__("north", north)
        super().__setattr__("crs", self.normalize_crs(crs) if crs is not None else None)

    @staticmethod
    def normalize_crs(crs: Union[str, int]) -> str:
        if isinstance(crs, int):
            return f"EPSG:{crs}"
        elif isinstance(crs, str):
            # TODO: support other CRS'es too?
            if not re.match("^epsg:\d+$", crs, flags=re.IGNORECASE):
                raise BoundingBoxException(f"Invalid CRS {crs!r}")
            return crs.upper()
        raise BoundingBoxException(f"Invalid CRS {crs!r}")

    @classmethod
    def from_dict(
        cls, d: Mapping, *, default_crs: Optional[Union[str, int]] = None
    ) -> "BoundingBox":
        """
        Extract bounding box from given mapping/dict (required fields "west", "south", "east", "north"),
        with optional CRS (field "crs").

        :param d: dictionary with at least fields "west", "south", "east", "north", and optionally "crs"
        :param default_crs: fallback CRS to use if not present in dictionary
        :return:
        """
        bounds = {k: d.get(k) for k in ["west", "south", "east", "north"]}
        return cls(**bounds, crs=d.get("crs", default_crs))

    @classmethod
    def from_dict_or_none(
        cls, d: Mapping, *, default_crs: Optional[Union[str, int]] = None
    ) -> Union["BoundingBox", None]:
        """
        Like `from_dict`, but returns `None`
        when no valid bounding box could be loaded from dict
        """
        try:
            return cls.from_dict(d=d, default_crs=default_crs)
        except BoundingBoxException:
            # TODO: option to log something?
            return None

    @classmethod
    def from_wsen_tuple(
        cls, wsen: Sequence[float], crs: Optional[Union[str, int]] = None
    ):
        """Build bounding box from tuple of west, south, east and north bounds (and optional crs"""
        assert len(wsen) == 4
        return cls(*wsen, crs=crs)

    def is_georeferenced(self) -> bool:
        return self.crs is not None

    def assert_crs(self):
        if self.crs is None:
            raise CrsRequired(f"A CRS is required, but not available in {self}.")

    def as_dict(self) -> dict:
        return {
            "west": self.west,
            "south": self.south,
            "east": self.east,
            "north": self.north,
            "crs": self.crs,
        }

    def as_tuple(self) -> Tuple[float, float, float, float, Union[str, None]]:
        return (self.west, self.south, self.east, self.north, self.crs)

    def as_wsen_tuple(self) -> Tuple[float, float, float, float]:
        return (self.west, self.south, self.east, self.north)

    def as_polygon(self) -> shapely.geometry.Polygon:
        """Get bounding box as a shapely Polygon"""
        return shapely.geometry.box(
            minx=self.west, miny=self.south, maxx=self.east, maxy=self.north
        )

    def contains(self, x: float, y: float) -> bool:
        """Check if given point is inside the bounding box"""
        return (self.west <= x <= self.east) and (self.south <= y <= self.north)

    def reproject(self, crs) -> "BoundingBox":
        """
        Reproject bounding box to given CRS to a new bounding box.

        Note that bounding box of the reprojected geometry
        typically has a larger spatial coverage than the
        original bounding box.
        """
        self.assert_crs()
        crs = self.normalize_crs(crs)
        if crs == self.crs:
            return self
        transform = pyproj.Transformer.from_crs(
            crs_from=self.crs, crs_to=crs, always_xy=True
        ).transform
        reprojected = shapely.ops.transform(transform, self.as_polygon())
        return BoundingBox(*reprojected.bounds, crs=crs)

    def best_utm(self) -> int:
        """
        Determine the best UTM zone for this bbox

        :return: EPSG code of UTM zone, e.g. 32631 for Belgian bounding boxes
        """
        self.assert_crs()
        return auto_utm_epsg_for_geometry(self.as_polygon(), crs=self.crs)

    def reproject_to_best_utm(self):
        return self.reproject(crs=self.best_utm())
