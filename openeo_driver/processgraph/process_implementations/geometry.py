"""Vector/geometry operation process implementations."""
import logging

import geopandas as gpd
import numpy as np
import pyproj
import shapely.geometry

from openeo_driver.datacube import DriverDataCube, DriverVectorCube
from openeo_driver.delayed_vector import DelayedVector
from openeo_driver.dry_run import DryRunDataCube, DryRunDataTracer
from openeo_driver.errors import FeatureUnsupportedException, ProcessParameterInvalidException
from openeo_driver.processes import ProcessArgs, ProcessSpec
from openeo_driver.processgraph.registry import (
    ENV_DRY_RUN_TRACER,
    non_standard_process,
    process_registry_100,
    process_registry_2xx,
)
from openeo_driver.specs import read_spec
from openeo_driver.util.utm import auto_utm_epsg_for_geometry
from openeo_driver.utils import EvalEnv

_log = logging.getLogger(__name__)


def _check_geometry_path_assumption(path: str, process: str, parameter: str):
    from openeo.util import str_truncate
    if isinstance(path, str) and path.lstrip().startswith("{"):
        raise ProcessParameterInvalidException(
            parameter=parameter,
            process=process,
            reason=f"provided a string (to be handled as path/URL), but it looks like (Geo)JSON encoded data: {str_truncate(path, width=32)!r}.",
        )


@non_standard_process(
    ProcessSpec(id='vector_buffer', description="Add a buffer around a geometry.")
        .param(name='geometries', description="Input geometry to add buffer to, vector-cube or GeoJSON(deprecated).",
               schema=[{
                "type": "object",
                "subtype": "datacube",
                "dimensions": [{"type": "geometry"}]
            }, {"type": "object", "subtype": "geojson"}])
        .param(name='distance',
               description="The distance of the buffer in meters.",
               schema={"type": "number"}, required=True)
        .returns(description="Output geometry (GeoJSON object) with the added or subtracted buffer",
                 schema={"type": "object", "subtype": "geojson"})
)
def vector_buffer(args: ProcessArgs, env: EvalEnv) -> dict:
    if "geometry" in args and "geometries" not in args:
        _log.warning("DEPRECATED: vector_buffer expects `geometries` argument, not `geometry`")
        geometry = args.get_required("geometry")
    else:
        geometry = args.get_required("geometries")
    distance = args.get_required("distance", expected_type=(int, float))
    if "unit" in args:
        _log.warning("vector_buffer: usage of non-standard 'unit' parameter")
    unit = args.get_optional("unit", default="meter")
    input_crs = output_crs = 'epsg:4326'
    buffer_resolution = 3

    if isinstance(geometry, DriverVectorCube):
        geoms = geometry.get_geometries()
        input_crs = geometry.get_crs()
    elif isinstance(geometry, str):
        _check_geometry_path_assumption(path=geometry, process="vector_buffer", parameter="geometry")
        geoms = list(DelayedVector(geometry).geometries)
    elif isinstance(geometry, dict) and "type" in geometry:
        geometry_type = geometry["type"]
        if geometry_type == "FeatureCollection":
            geoms = [shapely.geometry.shape(feat["geometry"]) for feat in geometry["features"]]
        elif geometry_type == "GeometryCollection":
            geoms = [shapely.geometry.shape(geom) for geom in geometry["geometries"]]
        elif geometry_type in {"Polygon", "MultiPolygon", "Point", "MultiPoint", "LineString"}:
            geoms = [shapely.geometry.shape(geometry)]
        elif geometry_type == "Feature":
            geoms = [shapely.geometry.shape(geometry["geometry"])]
        else:
            raise ProcessParameterInvalidException(
                parameter="geometry", process="vector_buffer", reason=f"Invalid geometry type {geometry_type}."
            )
        if "crs" in geometry:
            _log.warning("Handling GeoJSON dict with (non-standard) crs field")
            try:
                crs_name = geometry["crs"]["properties"]["name"]
                input_crs = pyproj.crs.CRS.from_string(crs_name)
            except Exception:
                _log.error(f"Failed to parse input geometry CRS {crs_name!r}", exc_info=True)
                raise ProcessParameterInvalidException(
                    parameter="geometry", process="vector_buffer", reason=f"Failed to parse input geometry CRS."
                )
    else:
        raise ProcessParameterInvalidException(
            parameter="geometry", process="vector_buffer", reason="The input geometry cannot be parsed"
        )
    geoms = gpd.GeoSeries(geoms, crs=input_crs)

    unit_scaling = {"meter": 1, "kilometer": 1000}
    if unit not in unit_scaling:
        raise ProcessParameterInvalidException(
            parameter="unit", process="vector_buffer",
            reason=f"Invalid unit {unit!r}. Should be one of {list(unit_scaling.keys())}."
        )
    distance = distance * unit_scaling[unit]

    epsg_utmzone = auto_utm_epsg_for_geometry(geoms.geometry[0])
    poly_buff_latlon = geoms.to_crs(epsg_utmzone).buffer(distance, resolution=buffer_resolution).to_crs(output_crs)

    empty_result_indices = np.where(poly_buff_latlon.is_empty)[0]
    if empty_result_indices.size > 0:
        raise ProcessParameterInvalidException(
            parameter="geometry", process="vector_buffer",
            reason=f"Buffering with distance {distance} {unit} resulted in empty geometries "
                   f"at position(s) {empty_result_indices}"
        )

    return (
        shapely.geometry.mapping(poly_buff_latlon[0])
        if len(poly_buff_latlon) == 1
        else shapely.geometry.mapping(poly_buff_latlon)
    )


@non_standard_process(
    ProcessSpec("get_geometries", description="Reads vector data from a file or a URL or get geometries from a FeatureCollection")
        .param('filename', description="filename or http url of a vector file", schema={"type": "string"}, required=False)
        .param('feature_collection', description="feature collection", schema={"type": "object"}, required=False)
        .returns("TODO", schema={"type": "object", "subtype": "vector-cube"})
)
def get_geometries(args: ProcessArgs, env: EvalEnv):
    feature_collection = args.get_optional("feature_collection")
    path = args.get_optional("filename", None)
    if path is not None:
        _check_geometry_path_assumption(path=path, process="get_geometries", parameter="filename")
        return DelayedVector(path)
    else:
        return feature_collection


@non_standard_process(
    ProcessSpec("read_vector", description="Reads vector data from a file or a URL.")
        .param(
            "filename",
            description="Vector file reference: a HTTP(S) URL or a file path.",
            schema=[
                {
                    "description": "Public URL to a vector file.",
                    "type": "string", "subtype": "uri",
                    "pattern": "^https?://",
                },
                {
                    "description": "File path (resolvable back-end-side) to a vector file.",
                    "type": "string", "subtype": "file-path",
                    "pattern": "^[^\r\n\\:'\"]+$",
                },
            ])
        .returns("GeoJSON-style feature collection", schema={"type": "object", "subtype": "geojson"})
)
def read_vector(args: ProcessArgs, env: EvalEnv) -> DelayedVector:
    _log.warning("DEPRECATED: read_vector usage")
    path = args.get_required("filename")
    _check_geometry_path_assumption(path=path, process="read_vector", parameter="filename")
    return DelayedVector(path)


@non_standard_process(
    ProcessSpec(
        id="to_vector_cube",
        description="[EXPERIMENTAL:] Converts given data (e.g. GeoJson object) to a vector cube."
    )
    .param('data', description="GeoJson object.", schema={"type": "object", "subtype": "geojson"})
    .returns("vector-cube", schema={"type": "object", "subtype": "vector-cube"})
)
def to_vector_cube(args: ProcessArgs, env: EvalEnv):
    _log.warning("DEPRECATED: process to_vector_cube is deprecated, use load_geojson instead")
    data = args.get_required("data")
    if isinstance(data, dict) and data.get("type") in {"Polygon", "MultiPolygon", "Feature", "FeatureCollection"}:
        return env.backend_implementation.vector_cube_cls.from_geojson(data)
    raise FeatureUnsupportedException(f"Converting {type(data)} to vector cube is not supported")


@non_standard_process(
    ProcessSpec("raster_to_vector",
                description="Converts this raster data cube into a vector data cube.",
                extra={"experimental": True})
        .param('data', description="A raster data cube.", schema={"type": "object", "subtype": "raster-cube"})
        .returns("vector-cube", schema={"type": "object", "subtype": "vector-cube"})
)
def raster_to_vector(args: ProcessArgs, env: EvalEnv):
    cube: DriverDataCube = args.get_required("data", expected_type=DriverDataCube)
    return cube.raster_to_vector()


@non_standard_process(
    ProcessSpec("vector_to_raster",
                description="Creates a raster cube as output based on a vector cube.",
                extra={"experimental": True})
        .param('data', description="A vector data cube.", schema={"type": "object", "subtype": "vector-cube"})
        .param('target', description="A raster data cube used as reference.",
               schema={"type": "object", "subtype": "raster-cube"})
        .returns("raster-cube", schema={"type": "object", "subtype": "raster-cube"})
)
def vector_to_raster(args: ProcessArgs, env: EvalEnv) -> DriverDataCube:
    input_vector_cube = args.get_required("data")
    dry_run_tracer: DryRunDataTracer = env.get(ENV_DRY_RUN_TRACER)
    if dry_run_tracer:
        if not isinstance(input_vector_cube, DryRunDataCube):
            raise ProcessParameterInvalidException(
                parameter="data",
                process="vector_to_raster",
                reason=f"Invalid data type {type(input_vector_cube)!r} expected vector-cube.",
            )
        return input_vector_cube

    target = args.get_required("target", expected_type=DriverDataCube)
    if not isinstance(input_vector_cube, DriverVectorCube) and not hasattr(input_vector_cube, "to_driver_vector_cube"):
        raise ProcessParameterInvalidException(
            parameter="data",
            process="vector_to_raster",
            reason=f"Invalid data type {type(input_vector_cube)!r} expected vector-cube.",
        )
    return env.backend_implementation.vector_to_raster(input_vector_cube, target)
