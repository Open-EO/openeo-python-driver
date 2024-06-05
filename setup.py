from setuptools import setup, find_packages

# Load the openeo version info.
#
# Note that we cannot simply import the module, since dependencies listed
# in setup() will very likely not be installed yet when setup.py run.
#
# See:
#   https://packaging.python.org/guides/single-sourcing-package-version

__version__ = None

with open("openeo_driver/_version.py") as fp:
    exec(fp.read())

version = __version__

tests_require = [
    "pytest",
    "mock",
    "requests-mock",
    "pylint>=2.5.0",
    "astroid>=2.4.0",
    "openeo_udf>=1.0.0rc3",
    "boto3[s3]>=1.26.17",
    "moto[s3]>=5.0.0",
    "time-machine>=2.8.0",
    "netCDF4>=1.5.4",
    "re-assert",
    "pyarrow>=10.0.0",
    "pystac",
    "jsonschema",
    "dirty-equals>=0.6",
    "typing-extensions<4.12.1",  # Workaround for https://github.com/eu-cdse/openeo-cdse-infra/issues/161
]

setup(
    name="openeo_driver",
    version=version,
    author="Jeroen Dries",
    author_email="jeroen.dries@vito.be",
    description="Flask based frontend for openEO API implementations in Python.",
    long_description="Flask based frontend for openEO API implementations in Python.",
    url="https://github.com/Open-EO/openeo-python-driver",
    packages=find_packages(include=["openeo_driver*"]),
    include_package_data=True,
    data_files=[
        ("openeo-python-driver-data", ["CHANGELOG.md"]),
    ],
    python_requires=">=3.8",
    setup_requires=["pytest-runner"],
    tests_require=tests_require,
    install_requires=[
        "flask>=2.0.0",
        "werkzeug>=2.3.8,<3.0.0",  # https://github.com/Open-EO/openeo-python-driver/issues/243
        "requests>=2.28.0",
        "openeo>=0.25.0",
        "openeo_processes==0.0.4",  # 0.0.4 is special build/release, also see https://github.com/Open-EO/openeo-python-driver/issues/152
        "gunicorn>=20.0.1",
        "numpy>=1.22.0",
        "shapely<2.0.0",  # https://github.com/Open-EO/openeo-python-driver/issues/158
        "pandas>=1.4.0",
        "geopandas>=0.11.0",  # 0.11.0 fixes https://github.com/geopandas/geopandas/pull/2243
        "xarray>=0.16.2",
        "flask-cors",
        "pyproj>=2.1.0",
        "python-dateutil",
        "python-json-logger>=2.0.0",
        "deprecated>=1.2.12",
        "importlib_resources; python_version<'3.10'",
        "attrs>=23.1.0",
        "fiona>=1.9.0",
        "reretry~=0.11.8",
        "markdown>3.4",
    ],
    extras_require={
        "dev": tests_require,
        "s3": ["boto3[s3]>=1.26.17", "botocore"],
        "vault": ["hvac>=1.0.2"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Development Status :: 4 - Beta",
        "Operating System :: OS Independent",
    ],
)
