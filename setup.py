from setuptools import setup, find_packages

# Load the openeo version info.
#
# Note that we cannot simply import the module, since dependencies listed
# in setup() will very likely not be installed yet when setup.py run.
#
# See:
#   https://packaging.python.org/guides/single-sourcing-package-version

__version__ = None

with open('openeo_driver/_version.py') as fp:
    exec(fp.read())

version = __version__

tests_require = [
    'pytest',
    'mock',
    'requests-mock',
    'pylint>=2.5.0',
    'astroid>=2.4.0',
    'openeo_udf>=1.0.0rc3',
    "time-machine",
]

setup(
    name='openeo_driver',
    version=version,
    author='Jeroen Dries',
    author_email='jeroen.dries@vito.be',
    description='Python implementation of openEO web service, with abstract implementation of processes.',
    url='https://github.com/Open-EO/openeo-python-driver',
    packages=find_packages(include=['openeo_driver*']),
    include_package_data=True,
    python_requires=">=3.8",
    setup_requires=['pytest-runner'],
    tests_require=tests_require,
    install_requires=[
        'flask',
        'werkzeug>=1.0.1',
        "requests>=2.28.0",
        'openeo>=0.13.1a2.dev',
        'openeo_processes==0.0.4',  # 0.0.4 is special build/release, also see https://github.com/Open-EO/openeo-python-driver/issues/152
        'gunicorn>=20.0.1',
        'numpy>=1.22.0',
        'shapely',
        'pandas',
        'geopandas>=0.11.0',  # 0.11.0 fixes https://github.com/geopandas/geopandas/pull/2243
        'xarray~=0.16.2',
        'netCDF4~=1.5.4',
        'flask-cors',
        'pyproj',
        'python-dateutil',
        "python-json-logger>=2.0.0",
        'deprecated>=1.2.12',
    ],
    extras_require={
        "dev": tests_require,
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Development Status :: 4 - Beta',
        'Operating System :: OS Independent'
    ]
)
