"""Terrain analysis process implementations."""
from openeo_driver.datacube import DriverDataCube
from openeo_driver.processes import ProcessArgs, ProcessSpec
from openeo_driver.processgraph.registry import non_standard_process
from openeo_driver.utils import EvalEnv


@non_standard_process(
    ProcessSpec(
        id="aspect",
        description="Computes the aspect (in radians, from due North) from elevation data.",
        extra={
            "summary": "Compute aspect on elevation data",
            "categories": ["cubes", "elevation"],
            "experimental": True
        }
    )
    .param('data', description="Data cube containing elevation data.",
           schema={"type": "object", "subtype": "raster-cube"})
    .returns(
        description="A data cube with calculated aspects for each band, the band names are the original band names with a '_aspect' suffix.",
        schema={"type": "object", "subtype": "raster-cube"})
)
def aspect(args: ProcessArgs, env: EvalEnv) -> DriverDataCube:
    cube: DriverDataCube = args.get_required("data", expected_type=DriverDataCube)
    return cube.aspect()


@non_standard_process(
    ProcessSpec(
        id="slope",
        description="Computes the slope (in radians, relative to the horizontal plane) from elevation data.",
        extra={
            "summary": "Compute slope on elevation data",
            "categories": ["cubes", "math", "elevation"],
            "experimental": True
        }
    )
    .param('data', description="Data cube containing elevation data.",
           schema={"type": "object", "subtype": "raster-cube"})
    .returns(
        description="A data cube with calculated slopes for each band, the band names are the original band names with a '_slope' suffix.",
        schema={"type": "object", "subtype": "raster-cube"})
)
def slope(args: ProcessArgs, env: EvalEnv) -> DriverDataCube:
    cube: DriverDataCube = args.get_required("data", expected_type=DriverDataCube)
    return cube.slope()


@non_standard_process(
    ProcessSpec(
        id="convert_data_type",
        description="Converts the data type of the cube to the desired data type.",
        extra={
            "summary": "Converts the data type of the cube",
            "categories": ["cubes"],
            "experimental": True
        }
    )
    .param('data', description="The data cube.",
           schema={"type": "object", "subtype": "raster-cube"})
    .param('data_type', description="The desired data type, represented as a string.",
           schema={"type": "string"}, required=True)
    .returns(
        description="A data cube with desired data type.",
        schema={"type": "object", "subtype": "raster-cube"})
)
def convert_data_type(args: ProcessArgs, env: EvalEnv) -> DriverDataCube:
    cube: DriverDataCube = args.get_required("data", expected_type=DriverDataCube)
    data_type = args.get_required("data_type", expected_type=str)
    return cube.convert_data_type(data_type=data_type)
