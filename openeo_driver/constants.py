class STAC_EXTENSION:
    PROCESSING = "https://stac-extensions.github.io/processing/v1.1.0/schema.json"
    EO = "https://stac-extensions.github.io/eo/v1.1.0/schema.json"
    EO_V110 = "https://stac-extensions.github.io/eo/v1.1.0/schema.json"
    EO_V200 = "https://stac-extensions.github.io/eo/v2.0.0/schema.json"
    FILEINFO = "https://stac-extensions.github.io/file/v2.1.0/schema.json"
    PROJECTION = "https://stac-extensions.github.io/projection/v1.1.0/schema.json"
    DATACUBE = "https://stac-extensions.github.io/datacube/v2.2.0/schema.json"
    MLMODEL = "https://stac-extensions.github.io/ml-model/v1.0.0/schema.json"
    CARD4LOPTICAL = "https://stac-extensions.github.io/card4l/v0.1.0/optical/schema.json"
    CARD4LSAR = "https://stac-extensions.github.io/card4l/v0.1.0/sar/schema.json"
    RASTER_V110 = "https://stac-extensions.github.io/raster/v1.1.0/schema.json"
    RASTER_V200 = "https://stac-extensions.github.io/raster/v2.0.0/schema.json"


class JOB_STATUS:
    """
    Container of batch job status constants.

    Allows to easily find places where batch job status is checked/set/updated.
    """

    CREATED = "created"
    QUEUED = "queued"
    RUNNING = "running"
    CANCELED = "canceled"
    FINISHED = "finished"
    ERROR = "error"


# Resample methods as used in official specs of `resample_spatial` and `resample_cube_spatial`
RESAMPLE_SPATIAL_METHODS = [
    "average",
    "bilinear",
    "cubic",
    "cubicspline",
    "lanczos",
    "max",
    "med",
    "min",
    "mode",
    "near",
    "q1",
    "q3",
    "rms",
    "sum",
]

# Align options as used in official spec of `resample_spatial`
RESAMPLE_SPATIAL_ALIGNS = [
    "lower-left",
    "upper-left",
    "lower-right",
    "upper-right",
]


# Default value for `log_level` parameter in `POST /result`, `POST /jobs`, ... requests
DEFAULT_LOG_LEVEL_PROCESSING = "info"
# Default value for `level in `GET /jobs/{job_id}/logs`, `GET /services/{service_id}/logs` requests
DEFAULT_LOG_LEVEL_RETRIEVAL = "debug"
