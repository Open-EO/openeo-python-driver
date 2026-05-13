"""SAR and atmospheric correction process implementations."""
import logging

from openeo_driver.datacube import DriverDataCube
from openeo_driver.datastructs import ResolutionMergeArgs, SarBackscatterArgs
from openeo_driver.processes import ProcessArgs
from openeo_driver.processgraph.registry import (
    custom_process_from_process_graph,
    non_standard_process,
    process_registry_100,
    process_registry_2xx,
)
from openeo_driver.processes import ProcessSpec
from openeo_driver.specs import read_spec
from openeo_driver.utils import EvalEnv

_log = logging.getLogger(__name__)


@non_standard_process(
    ProcessSpec(
        id='atmospheric_correction',
        description="Applies an atmospheric correction that converts top of atmosphere reflectance values into bottom of atmosphere/top of canopy reflectance values.",
        extra={
            "summary": "Apply atmospheric correction",
            "categories": ["cubes", "optical"],
            "experimental": True,
            "links": [
                {
                    "rel": "about",
                    "href": "https://bok.eo4geo.eu/IP1-7-1",
                    "title": "Atmospheric correction explained by EO4GEO body of knowledge."
                }
            ],
            "exceptions": {
                "DigitalElevationModelInvalid": {
                    "message": "The digital elevation model specified is either not a DEM or can't be used with the data cube given."
                }
            },
        }
    )
    .param('data', description="Data cube containing multi-spectral optical top of atmosphere reflectances to be corrected.", schema={"type": "object", "subtype": "raster-cube"})
    .param(name='method', description="The atmospheric correction method to use.", schema={"type": "string"}, required=False)
    .param(name='elevation_model', description="The digital elevation model to use.", schema={"type": "string"}, required=False)
    .param(name='missionId', description="non-standard mission Id, currently defaults to sentinel2", schema={"type": "string"}, required=False)
    .param(name='sza', description="non-standard if set, overrides sun zenith angle values [deg]", schema={"type": "number"}, required=False)
    .param(name='vza', description="non-standard if set, overrides sensor zenith angle values [deg]", schema={"type": "number"}, required=False)
    .param(name='raa', description="non-standard if set, overrides rel. azimuth angle values [deg]", schema={"type": "number"}, required=False)
    .param(name='gnd', description="non-standard if set, overrides ground elevation [km]", schema={"type": "number"}, required=False)
    .param(name='aot', description="non-standard if set, overrides aerosol optical thickness [], usually 0.1..0.2", schema={"type": "number"}, required=False)
    .param(name='cwv', description="non-standard if set, overrides water vapor [], usually 0..7", schema={"type": "number"}, required=False)
    .param(name='appendDebugBands', description="non-standard if set to 1, saves debug bands", schema={"type": "number"}, required=False)
    .returns(description="the corrected data as a data cube", schema={"type": "object", "subtype": "raster-cube"})
)
def atmospheric_correction(args: ProcessArgs, env: EvalEnv) -> DriverDataCube:
    cube: DriverDataCube = args.get_required("data", expected_type=DriverDataCube)
    method = args.get_optional("method", expected_type=str)
    elevation_model = args.get_optional("elevation_model", expected_type=str)
    mission_id = args.get_optional("missionId", expected_type=str)
    sza = args.get_optional("sza", expected_type=float)
    vza = args.get_optional("vza", expected_type=float)
    raa = args.get_optional("raa", expected_type=float)
    gnd = args.get_optional("gnd", expected_type=float)
    aot = args.get_optional("aot", expected_type=float)
    cwv = args.get_optional("cwv", expected_type=float)
    append_debug_bands = args.get_optional("appendDebugBands", expected_type=int)
    return cube.atmospheric_correction(
        method=method,
        elevation_model=elevation_model,
        options={
            "mission_id": mission_id,
            "sza": sza,
            "vza": vza,
            "raa": raa,
            "gnd": gnd,
            "aot": aot,
            "cwv": cwv,
            "append_debug_bands": append_debug_bands,
        },
    )


@process_registry_100.add_function(spec=read_spec("openeo-processes/1.x/proposals/sar_backscatter.json"))
@process_registry_2xx.add_function(spec=read_spec("openeo-processes/2.x/proposals/sar_backscatter.json"))
def sar_backscatter(args: ProcessArgs, env: EvalEnv):
    cube: DriverDataCube = args.get_required("data", expected_type=DriverDataCube)
    kwargs = args.get_subset(
        names=[
            "coefficient",
            "elevation_model",
            "mask",
            "contributing_area",
            "local_incidence_angle",
            "ellipsoid_incidence_angle",
            "noise_removal",
            "options",
        ]
    )
    return cube.sar_backscatter(SarBackscatterArgs(**kwargs))


@process_registry_100.add_function(spec=read_spec("openeo-processes/experimental/resolution_merge.json"))
@process_registry_2xx.add_function(spec=read_spec("openeo-processes/experimental/resolution_merge.json"))
def resolution_merge(args: ProcessArgs, env: EvalEnv):
    cube: DriverDataCube = args.get_required("data", expected_type=DriverDataCube)
    kwargs = args.get_subset(names=["method", "high_resolution_bands", "low_resolution_bands", "options"])
    return cube.resolution_merge(ResolutionMergeArgs(**kwargs))


@process_registry_100.add_function(spec=read_spec("openeo-processes/experimental/mask_scl_dilation.json"))
@process_registry_2xx.add_function(spec=read_spec("openeo-processes/experimental/mask_scl_dilation.json"))
def mask_scl_dilation(args: ProcessArgs, env: EvalEnv):
    _log.warning(
        "mask_scl_dilation is an experimental process and deprecated (in favor of to_scl_dilation_mask). Support will be removed soon."
    )
    cube: DriverDataCube = args.get_required("data", expected_type=DriverDataCube)
    if hasattr(cube, "mask_scl_dilation"):
        the_args = args.copy()
        del the_args["data"]
        return cube.mask_scl_dilation(**the_args)
    else:
        return cube


@process_registry_100.add_function(spec=read_spec("openeo-processes/experimental/to_scl_dilation_mask.json"))
@process_registry_2xx.add_function(spec=read_spec("openeo-processes/experimental/to_scl_dilation_mask.json"))
def to_scl_dilation_mask(args: ProcessArgs, env: EvalEnv):
    cube: DriverDataCube = args.get_required("data", expected_type=DriverDataCube)
    spec = read_spec("openeo-processes/experimental/to_scl_dilation_mask.json")
    defaults = {param["name"]: param["default"] for param in spec["parameters"] if "default" in param}
    optionals = {
        arg: args.get_optional(arg, default=defaults[arg])
        for arg in [
            "erosion_kernel_size",
            "mask1_values",
            "mask2_values",
            "kernel1_size",
            "kernel2_size",
        ]
    }
    return cube.to_scl_dilation_mask(**optionals)


@process_registry_100.add_function(spec=read_spec("openeo-processes/experimental/mask_l1c.json"))
@process_registry_2xx.add_function(spec=read_spec("openeo-processes/experimental/mask_l1c.json"))
def mask_l1c(args: ProcessArgs, env: EvalEnv):
    cube: DriverDataCube = args.get_required("data", expected_type=DriverDataCube)
    if hasattr(cube, "mask_l1c"):
        return cube.mask_l1c()
    else:
        return cube


# Register process graph-based fallback for ard_normalized_radar_backscatter
custom_process_from_process_graph(read_spec("openeo-processes/1.x/proposals/ard_normalized_radar_backscatter.json"))
