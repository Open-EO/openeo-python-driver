import glob
import os
import pathlib
import tempfile
import warnings
import logging
from pathlib import Path
from shutil import copy
from tempfile import mkstemp
from typing import Union, Dict, List, Optional, Any
from zipfile import ZipFile

import numpy as np
import pandas as pd
import typing
from flask import send_from_directory, jsonify, Response
from shapely.geometry import GeometryCollection, mapping
from shapely.geometry.base import BaseGeometry
import xarray

from openeo.metadata import CollectionMetadata
from openeo_driver.datacube import DriverDataCube, DriverVectorCube, DriverMlModel
from openeo_driver.datastructs import StacAsset
from openeo_driver.delayed_vector import DelayedVector
from openeo_driver.errors import OpenEOApiException, FeatureUnsupportedException, InternalException
from openeo_driver.util.ioformats import IOFORMATS
from openeo_driver.utils import replace_nan_values

_log = logging.getLogger(__name__)


class SaveResult:
    """
    Encapsulation of a processing result (raster data cube, vector cube, array, ...)
    and how it should be saved (format and additional options).

    To be delivered to the user through a Flask response (synchronous mode) or
    assets download URLs (batch mode).
    """

    DEFAULT_FORMAT = None

    def __init__(self, format: Optional[str] = None, options: Optional[dict] = None):
        self.format = format or self.DEFAULT_FORMAT
        self.options = options or {}

    def is_format(self, *args):
        return self.format.lower() in {f.lower() for f in args}

    def set_format(self, format: str, options: dict = None):
        self.format = format.lower()
        self.options = options or {}

    def write_assets(self, directory: Union[str, Path]) -> Dict[str, StacAsset]:
        raise NotImplementedError

    def create_flask_response(self) -> Response:
        """
        Returns a Flask compatible response. The view is unaware of the output format; rather, it is derived from the
        process graph.

        :return: A response that can be handled by Flask
        """
        raise NotImplementedError

    def flask_response_from_write_assets(self) -> Response:
        """Helper to generate a Flask response from `write_assets` result."""
        with tempfile.TemporaryDirectory(prefix="openeo-pydrvr-") as tmp_dir:
            assets = self.write_assets(directory=tmp_dir)
            if len(assets) == 0:
                raise InternalException("No assets written")
            if len(assets) > 1:
                # TODO support zipping multiple assets
                raise FeatureUnsupportedException("Multi-file responses not yet supported")
            asset = assets.popitem()[1]
            path = Path(asset["href"])
            mimetype = asset.get("type")
            return send_from_directory(path.parent, path.name, mimetype=mimetype)

    def get_mimetype(self, default="application/octet-stream"):
        return IOFORMATS.get_mimetype(self.format)


def get_temp_file(suffix="", prefix="openeo-pydrvr-"):
    # TODO: make sure temp files are cleaned up when read
    _, filename = tempfile.mkstemp(suffix=suffix, prefix=prefix)
    return filename


class ImageCollectionResult(SaveResult):

    DEFAULT_FORMAT = "GTiff"

    def __init__(self, cube: DriverDataCube, format: Optional[str] = None, options: Optional[dict] = None):
        super().__init__(format=format, options=options)
        self.cube = cube

    # TODO: simplify the back and forth between save_result and write_assets?

    def save_result(self, filename: str) -> str:
        # TODO: port to write_assets
        return self.cube.save_result(filename=filename, format=self.format, format_options=self.options)

    def write_assets(self, directory: Union[str, Path]) -> Dict[str, StacAsset]:
        """
        Save generated assets into a directory, return asset metadata.
        TODO: can an asset also be a full STAC item? In principle, one openEO job can either generate a full STAC collection, or one STAC item with multiple assets...

        :return: STAC assets dictionary: https://github.com/radiantearth/stac-spec/blob/master/item-spec/item-spec.md#assets
        """
        if hasattr(self.cube, "write_assets"):
            # TODO: code smell: filename=directory?
            return self.cube.write_assets(filename=directory, format=self.format, format_options=self.options)
        else:
            filename = self.cube.save_result(filename=directory, format=self.format, format_options=self.options)
            return {filename:{"href":filename}}

    def create_flask_response(self) -> Response:
        # TODO: clean up temp file
        # TODO: port to write_assets
        filename = get_temp_file(suffix=".save_result.{e}".format(e=self.format.lower()))
        filename = self.save_result(filename)
        mimetype = self.get_mimetype()
        return send_from_directory(os.path.dirname(filename), os.path.basename(filename), mimetype=mimetype)


class VectorCubeResult(SaveResult):
    # TODO merge implementation with ImageCollectionResult?

    DEFAULT_FORMAT = "GeoJSON"

    def __init__(self, cube: DriverVectorCube, format: Optional[str], options: Optional[dict] = None):
        super().__init__(format=format, options=options)
        self.cube = cube

    def write_assets(self, directory: Union[str, Path]) -> Dict[str, StacAsset]:
        return self.cube.write_assets(directory=directory, format=self.format, options=self.options)

    def create_flask_response(self) -> Response:
        return self.flask_response_from_write_assets()


class MlModelResult(SaveResult):
    # TODO merge implementation with ImageCollectionResult?

    def __init__(self, ml_model: DriverMlModel, options: Optional[dict] = None):
        super().__init__(options=options)
        self.ml_model = ml_model

    def write_assets(self, directory: Union[str, Path]) -> Dict[str, StacAsset]:
        return self.ml_model.write_assets(directory=directory)

    def get_model_metadata(self, directory: Union[str, Path]) -> Dict[str, typing.Any]:
        return self.ml_model.get_model_metadata(directory=directory)

    def create_flask_response(self) -> Response:
        return self.flask_response_from_write_assets()


class JSONResult(SaveResult):

    def __init__(self, data, format: str = "json", options: dict = None):
        super().__init__(format=format, options=options)
        self.data = data

    def write_assets(self, directory: Union[str, Path]) -> Dict[str, StacAsset]:
        """
        Save generated assets into a directory, return asset metadata.
        TODO: can an asset also be a full STAC item? In principle, one openEO job can either generate a full STAC collection, or one STAC item with multiple assets...

        :return: STAC assets dictionary: https://github.com/radiantearth/stac-spec/blob/master/item-spec/item-spec.md#assets
        """
        # TODO: There is something wrong here: arg is called `directory`,
        #       but implementation and actual usage handles it as a file path (take parent to get directory)
        output_dir = Path(directory).parent
        output_file = output_dir / "result.json"
        with open(output_file, 'w') as f:
            import json
            json.dump(self.prepare_for_json(), f)
        return {"result.json":{
            "href":str(output_file),
            "roles": ["data"],
            "type": "application/json",
            "description": "json result generated by openEO"
            }}

    def get_data(self):
        return self.data

    def prepare_for_json(self):
        return replace_nan_values(self.get_data())

    def create_flask_response(self) -> Response:
        return jsonify(self.prepare_for_json())


class AggregatePolygonResult(JSONResult):  # TODO: if it supports NetCDF and CSV, it's not a JSONResult
    """
    Container for timeseries result of `aggregate_polygon` process (aka "zonal stats")

    Expects internal representation of timeseries as nested structure:

        dict mapping timestamp (str) to:
            a list, one item per polygon:
                a list, one float per band

    """

    # TODO #71 #114 EP-3981 port this to proper vector cube support

    def __init__(self, timeseries: dict, regions: Union[GeometryCollection, DriverVectorCube],
                 metadata: CollectionMetadata = None):
        super().__init__(data=timeseries)
        if not isinstance(regions, (GeometryCollection, DriverVectorCube)):
            # TODO: raise exception instead of warning?
            warnings.warn("AggregatePolygonResult: GeometryCollection or DriverVectorCube expected but got {t}".format(t=type(regions)))
        self._regions = regions
        self._metadata = metadata
        self.raster_bands = None

    def get_data(self):
        if self.is_format('covjson', 'coveragejson'):
            return self.to_covjson()
        # By default, keep original (proprietary) result format
        return self.data

    def write_assets(self, directory: Union[str, Path]) -> Dict[str, StacAsset]:
        """
        Save generated assets into a directory, return asset metadata.
        TODO: can an asset also be a full STAC item? In principle, one openEO job can either generate a full STAC collection, or one STAC item with multiple assets...

        :return: STAC assets dictionary: https://github.com/radiantearth/stac-spec/blob/master/item-spec/item-spec.md#assets
        """
        # TODO: There is something wrong here: arg is called `directory`,
        #       but implementation and actual usage handles it as a file path (take parent to get directory)
        directory = pathlib.Path(directory).parent
        filename = str(Path(directory)/"timeseries.json")
        asset = {
            "roles": ["data"],
            "type": "application/json"
        }
        if self.is_format('netcdf', 'ncdf'):
            filename = str(Path(directory) / "timeseries.nc")
            self.to_netcdf(filename)
            asset["type"] = "application/x-netcdf"
        elif self.is_format('csv'):
            filename = str(Path(directory) / "timeseries.csv")
            self.to_csv(filename)
            asset["type"] = "text/csv"
        else:
            import json
            with open(filename, 'w') as f:
                json.dump(self.prepare_for_json(), f)
        asset["href"] = filename
        if self._metadata is not None and self._metadata.has_band_dimension():
            bands = [b._asdict() for b in self._metadata.bands]
            asset["bands"] = bands

        the_file = pathlib.Path(filename)
        if the_file.exists():
            size_in_bytes = the_file.stat().st_size
            asset["file:size"] = size_in_bytes

        if self.raster_bands is not None:
            asset["raster:bands"] = self.raster_bands


        return {str(Path(filename).name): asset}

    def create_flask_response(self) -> Response:
        if self.is_format('netcdf', 'ncdf'):
            filename = self.to_netcdf()
            return send_from_directory(
                os.path.dirname(filename),
                os.path.basename(filename),
                mimetype=IOFORMATS.get_mimetype(self.format),
            )

        if self.is_format('csv'):
            filename = self.to_csv()
            return send_from_directory(
                os.path.dirname(filename),
                os.path.basename(filename),
                mimetype=IOFORMATS.get_mimetype(self.format),
            )

        return super().create_flask_response()

    def _create_point_timeseries_xarray(self, feature_ids, timestamps, lats, lons, averages_by_feature):
        #xarray breaks with timezone aware dates: https://github.com/pydata/xarray/issues/1490
        band_names = [f"band_{band}" for band in range(averages_by_feature.shape[2])] if len(averages_by_feature.shape) > 2 else ["band_0"]

        if self._metadata is not None and self._metadata.has_band_dimension():
            band_names = self._metadata.band_names

        time_index = pd.to_datetime(timestamps,utc=False).tz_convert(None)
        if len(averages_by_feature.shape) == 3:
            array_definition = {'t': time_index}
            for band in range(averages_by_feature.shape[2]):
                data = averages_by_feature[:, :, band]
                array_definition[band_names[band]] =  (('feature', 't'), data)
            the_array = xarray.Dataset(array_definition)
        else:
            the_array = xarray.Dataset({
                'avg': (('feature', 't'), averages_by_feature),
                't': time_index})

        the_array = the_array.dropna(dim='t',how='all')
        the_array = the_array.sortby('t')

        the_array.coords['lat'] = (('feature'),lats)
        the_array.coords['lon'] = (('feature'), lons)
        the_array.coords['feature_names'] = (('feature'), feature_ids)

        the_array.variables['lat'].attrs['units'] = 'degrees_north'
        the_array.variables['lat'].attrs['standard_name'] = 'latitude'
        the_array.variables['lon'].attrs['units'] = 'degrees_east'
        the_array.variables['lon'].attrs['standard_name'] = 'longitude'
        the_array.variables['t'].attrs['standard_name'] = 'time'
        the_array.attrs['Conventions'] = 'CF-1.8'
        the_array.attrs['source'] = 'Aggregated timeseries generated by openEO GeoPySpark backend.'
        return the_array

    def to_netcdf(self, destination: Optional[str] = None) -> str:
        def features_ids_from_index(geometries):
            return ['feature_%d' % i for i in range(len(geometries))]

        if isinstance(self._regions, GeometryCollection):
            points = [r.representative_point() for r in self._regions]
            feature_ids = features_ids_from_index(self._regions)
        else:
            points = [r.representative_point() for r in self._regions.get_geometries()]
            feature_ids = self._regions.get_ids()
            feature_ids = (list(feature_ids) if feature_ids is not None
                           else features_ids_from_index(self._regions.get_geometries()))

        lats = [p.y for p in points]
        lons = [p.x for p in points]

        values = self.get_data().values()
        if self._metadata is not None and self._metadata.has_band_dimension():
            bandcount = len(self._metadata.bands)
        else:
            bandcount = max([max([len(bands)  for bands in feature_bands]) for feature_bands in values])
            if bandcount == 0:
                bandcount = 1
        fill_value = np.full((bandcount),fill_value=np.nan)

        cleaned_values =  [[bands if len(bands)>0 else fill_value for bands in feature_bands ] for feature_bands in values]
        time_feature_bands_array = np.array(list(cleaned_values))
        #time_feature_bands_array[time_feature_bands_array == []] = [np.nan, np.nan]
        assert len(feature_ids) == time_feature_bands_array.shape[1]
        #if len(time_feature_bands_array.shape) == 3:
        #    nb_bands = time_feature_bands_array.shape[2]
        feature_time_bands_array = np.swapaxes(time_feature_bands_array,0,1)
        array = self._create_point_timeseries_xarray(feature_ids, list(self.get_data().keys()), lats, lons, feature_time_bands_array)
        comp = dict(zlib=True, complevel=5)
        encoding = {var: comp for var in array.data_vars}
        if destination is None:
            filename = get_temp_file(suffix=".netcdf")
        else:
            filename = destination
        try:
            array.to_netcdf(filename,encoding=encoding)
        except Exception as e:
            _log.error(f"Writing failed for this array: {array}", exc_info=True)
            raise OpenEOApiException(f"Failed to write aggregated timeseries to netCDF file at {destination} due to {e}. Check the logging for more info.")
        return filename

    def to_csv(self, destination=None):
        nb_bands = max([len(item) for sublist in self.get_data().values() for item in sublist])

        date_band_dict = {}

        for date, polygon_results in self.get_data().items():
            if nb_bands > 1:
                for i in range(0, nb_bands):
                    date_band = date + '__' + str(i + 1).zfill(2)
                    date_band_dict[date_band] = [(p[i] if p else None) for p in polygon_results]
            else:
                date_band_dict[date] = [(p[0] if p else None) for p in polygon_results]

        if destination is None:
            filename = get_temp_file(suffix=".csv")
        else:
            filename = destination

        pd.DataFrame(date_band_dict).to_csv(filename, index=False)

        return filename

    def to_covjson(self) -> dict:
        """
        Convert internal timeseries structure to Coverage JSON structured dict
        """
        if self.data is None:
            return {}

        if isinstance(self._regions, GeometryCollection):
            # Convert GeometryCollection to list of GeoJSON Polygon coordinate arrays
            polygons = [p["coordinates"] for p in mapping(self._regions)["geometries"]]
        else:
            polygons = [mapping(g)["coordinates"] for g in self._regions.get_geometries()]

        # TODO make sure timestamps are ISO8601 (https://covjson.org/spec/#temporal-reference-systems)
        timestamps = sorted(self.data.keys())

        # Count bands in timestamp data
        # TODO get band count and names from metadata
        band_counts = set(len(polygon_data) for ts_data in self.data.values() for polygon_data in ts_data)
        band_counts.discard(0)
        if len(band_counts) != 1:
            raise ValueError("Multiple band counts in data: {c}".format(c=band_counts))
        band_count = band_counts.pop()

        # build parameter value array (one for each band)
        actual_timestamps = []
        param_values = [[] for _ in range(band_count)]
        for ts in timestamps:
            ts_data = self.data[ts]
            if len(ts_data) != len(polygons):
                warnings.warn("Expected {e} polygon results, but got {g}".format(e=len(polygons), g=len(ts_data)))
                continue
            if all(len(polygon_data) != band_count for polygon_data in ts_data):
                # Skip timestamps without any complete data
                # TODO: also skip timestamps with only NaNs?
                continue
            actual_timestamps.append(ts)
            for polygon_data in ts_data:
                if len(polygon_data) == band_count:
                    for b, v in enumerate(polygon_data):
                        param_values[b].append(v)
                else:
                    for b in range(band_count):
                        param_values[b].append(np.nan)

        domain = {
            "type": "Domain",
            "domainType": "MultiPolygonSeries",
            "axes": {
                "t": {"values": actual_timestamps},
                "composite": {
                    "dataType": "polygon",
                    "coordinates": ["x", "y"],
                    "values": polygons,
                },
            },
            "referencing": [
                {
                    "coordinates": ["x", "y"],
                    "system": {
                        "type": "GeographicCRS",
                        "id": "http://www.opengis.net/def/crs/OGC/1.3/CRS84",  # TODO check CRS
                    }
                },
                {
                    "coordinates": ["t"],
                    "system": {"type": "TemporalRS", "calendar": "Gregorian"}
                }
            ]
        }
        parameters = {
            "band{b}".format(b=band): {
                "type": "Parameter",
                "observedProperty": {"label": {"en": "Band {b}".format(b=band)}},
                # TODO: also add unit properties?
            }
            for band in range(band_count)
        }
        shape = (len(actual_timestamps), len(polygons))
        ranges = {
            "band{b}".format(b=band): {
                "type": "NdArray",
                "dataType": "float",
                "axisNames": ["t", "composite"],
                "shape": shape,
                "values": param_values[band],
            }
            for band in range(band_count)
        }
        return {
            "type": "Coverage",
            "domain": domain,
            "parameters": parameters,
            "ranges": ranges
        }


class AggregatePolygonResultCSV(AggregatePolygonResult):
    # TODO #71 #114 EP-3981 port this to proper vector cube support
    # TODO: this is a openeo-geopyspark-driver related/specific implementation, move it over there?

    def __init__(self, csv_dir, regions: Union[GeometryCollection, DriverVectorCube, DelayedVector, BaseGeometry], metadata: CollectionMetadata = None):
        super().__init__(timeseries=None, regions=regions, metadata=metadata)
        self._csv_dir = csv_dir
        self.raster_bands = None

    def get_data(self):
        if self.data is None:
            paths = list(glob.glob(os.path.join(self._csv_dir, "*.csv")))
            _log.info(f"Parsing intermediate timeseries results: {paths}")
            if len(paths) == 0:
                raise OpenEOApiException(status_code = 500, code = "EmptyResult",
                    message = f"aggregate_spatial did not generate any output, intermediate output path on the server: {self._csv_dir}")
            df = pd.concat(map(pd.read_csv, paths))
            features = df.feature_index.unique()
            if str(features.dtype) == 'int64':
                # TODO: This logic might get cleaned up when one kind ove vector cube is used everywhere
                if isinstance(self._regions, DriverVectorCube):
                    amount_of_regions = len(self._regions.get_geometries())
                elif isinstance(self._regions, str) or isinstance(self._regions, DelayedVector):
                    regions = self._regions
                    if isinstance(self._regions, str):
                        regions = DelayedVector(self._regions)
                    geometries = list(regions.geometries)
                    amount_of_regions = len(geometries)
                elif isinstance(self._regions, GeometryCollection):
                    amount_of_regions = len(self._regions)
                else:
                    _log.warning("Using polygon with largest index to estimate how many input polygons there where.")
                    amount_of_regions = features.max() + 1
                features = np.arange(0, amount_of_regions)
            else:
                features.sort()

            def _flatten_df(df):
                df.index = df.feature_index
                df = df.reindex(features)
                return df.drop(columns = "feature_index").values.tolist()

            self.data = {
                pd.to_datetime(date)
                .tz_convert("UTC")
                .strftime("%Y-%m-%dT%XZ"): _flatten_df(
                    df[df.date == date].drop(columns="date")
                )
                for date in df.date.unique()
            }

            # Compute stats.
            bands = df.columns[2:].values
            def stats(band):
                series = df[band]
                stats = {}
                stats["mean"] = series.mean()
                stats["minimum"] = series.min()
                stats["maximum"] = series.max()
                stats["stddev"] = series.std()
                stats["valid_percent"] = ((100.0 * len(series.dropna()) / len(series)) if len(series) else None)
                return {"statistics": stats}
            self.raster_bands = [stats(b) for b in bands]

        if self.is_format('covjson', 'coveragejson'):
            return self.to_covjson()
        return self.data

    def to_csv(self, destination=None):
        csv_paths = glob.glob(self._csv_dir + "/*.csv")
        # TODO: assumption there is only one CSV?
        if(destination == None):
            return csv_paths[0]
        else:
            copy(csv_paths[0],destination)
            return destination


class AggregatePolygonSpatialResult(SaveResult):
    """
    Container for result of `aggregate_polygon` process (aka "zonal stats") for a spatial layer.
    """
    # TODO #71 #114 EP-3981 replace with proper VectorCube implementation

    DEFAULT_FORMAT = "JSON"

    def __init__(self, csv_dir: Union[str, Path], regions: GeometryCollection, metadata: CollectionMetadata = None,
                 format: Optional[str] = None, options: Optional[dict] = None):
        super().__init__(format, options)
        self._csv_dir = Path(csv_dir)
        self._regions = regions
        self._metadata = metadata

    @staticmethod
    def _band_values_by_geometry(df: pd.DataFrame) -> List[List[float]]:
        df.index = df.feature_index
        df.sort_index(inplace=True)
        return df.drop(columns="feature_index").values.tolist()

    def prepare_for_json(self) -> List[List[float]]:
        df = pd.read_csv(self._csv_path())
        return self._band_values_by_geometry(df)

    def _csv_path(self) -> str:
        csv_paths = glob.glob(f"{self._csv_dir}/*.csv")
        # could support multiple files but currently assumes coalesce(1)
        assert len(csv_paths) == 1, f"expected exactly one CSV file at {self._csv_dir}"
        return csv_paths[0]

    def create_flask_response(self) -> Response:
        if self.is_format("json"):
            return jsonify(self.prepare_for_json())
        elif self.is_format("csv"):
            csv_path = self._csv_path()
            return send_from_directory(
                os.path.dirname(csv_path),
                os.path.basename(csv_path),
                mimetypes=IOFORMATS.get_mimetype(self.format),
            )
        else:
            raise FeatureUnsupportedException(f"Unsupported output format {self.format}; supported are: JSON and CSV")

    def write_assets(self, directory: Union[str, Path]) -> Dict[str, StacAsset]:
        # TODO: There is something wrong here: arg is called `directory`,
        #       but implementation and actual usage handles it as a file path (take parent to get directory)
        directory = pathlib.Path(directory).parent

        asset = {
            "roles": ["data"]
        }

        if self.is_format("json"):
            asset["type"] = "application/json"
            filename = str(Path(directory) / "timeseries.json")

            import json
            with open(filename, 'w') as f:
                json.dump(self.prepare_for_json(), f)
        elif self.is_format("csv"):
            filename = str(Path(directory) / "timeseries.csv")
            asset["type"] = "text/csv"

            copy(self._csv_path(), filename)
        else:
            raise FeatureUnsupportedException(f"Unsupported output format {self.format}; supported are: JSON and CSV")

        asset["href"] = filename

        if self._metadata is not None and self._metadata.has_band_dimension():
            bands = [b._asdict() for b in self._metadata.bands]
            asset["bands"] = bands

        return {str(Path(filename).name): asset}

    def fit_class_random_forest(
            self,
            target: dict,
            num_trees: int = 100,
            max_variables: Optional[Union[int, str]] = None,
            seed: Optional[int] = None
    ) -> DriverMlModel:
        # TODO: this method belongs eventually under DriverVectorCube
        raise NotImplementedError


class MultipleFilesResult(SaveResult):
    def __init__(self, format: str, *files: Path):
        super().__init__(format=format)
        self.files = list(files)

    def write_assets(self, directory: Union[str, Path]) -> Dict[str, StacAsset]:
        """
        Save generated assets into a directory, return asset metadata.
        TODO: can an asset also be a full STAC item? In principle, one openEO job can either generate a full STAC collection, or one STAC item with multiple assets...

        :return: STAC assets dictionary: https://github.com/radiantearth/stac-spec/blob/master/item-spec/item-spec.md#assets
        """
        # TODO: There is something wrong here: arg is called `directory`,
        #       but implementation and actual usage handles it as a file path (take parent to get directory)
        output_dir = Path(directory).parent
        output_file = output_dir / "result.zip"

        #TODO: this creates a zip, but we can perhaps also support multiple assets?
        self.reduce(output_file,delete_originals=True)

        return {"result.json":{
            "href":str(output_file),
            "roles": ["data"],
            "type": "application/zip",
            "description": "zip result generated by openEO"
            }}

    def reduce(self, output_file: Union[str, Path], delete_originals: bool):
        with ZipFile(output_file, "w") as zip_file:
            for file in self.files:
                zip_file.write(filename=file, arcname=file.name)

        if delete_originals:
            for file in self.files:
                file.unlink()

    def create_flask_response(self) -> Response:
        from flask import current_app

        _, temp_file_name = mkstemp(suffix=".zip", dir=Path.cwd())
        temp_file = Path(temp_file_name)

        def stream_and_remove():
            self.reduce(temp_file, delete_originals=True)

            with open(temp_file, "rb") as f:
                yield from f

            temp_file.unlink()

        resp = current_app.response_class(stream_and_remove(), mimetype="application/zip")
        resp.headers.set('Content-Disposition', 'attachment', filename=temp_file.name)

        return resp


class NullResult(SaveResult):

    def write_assets(self, directory: Union[str, Path]) -> Dict[str, StacAsset]:
        return {}

    def create_flask_response(self) -> Response:
        return jsonify(None)


def to_save_result(data: Any, format: Optional[str] = None, options: Optional[dict] = None) -> SaveResult:
    """
    Convert a process graph result to a SaveResult object
    """
    options = options or {}
    if isinstance(data, SaveResult):
        return data
    elif isinstance(data, DriverDataCube):
        return ImageCollectionResult(data, format=format, options=options)
    elif isinstance(data, DriverVectorCube):
        return VectorCubeResult(cube=data, format=format, options=options)
    elif isinstance(data, DelayedVector):
        # TODO #114 EP-3981 add vector cube support: keep features from feature collection
        geojsons = [mapping(geometry) for geometry in data.geometries_wgs84]
        return JSONResult(geojsons, format=format, options=options)
    elif isinstance(data, DriverMlModel):
        return MlModelResult(ml_model = data)
    elif isinstance(data, np.ndarray):
        return JSONResult(data.tolist())
    elif isinstance(data, np.generic):
        # Convert numpy datatype to native Python datatype first
        return JSONResult(data.item())
    elif isinstance(data, (list, tuple, dict, str, int, float)):
        # Generic JSON result
        return JSONResult(data)
    elif data is None:
        return NullResult()
    else:
        raise ValueError(f"No save result support for type {type(data)}")
