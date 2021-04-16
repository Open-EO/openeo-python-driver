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
    setup_requires=['pytest-runner'],
    tests_require=tests_require,
    install_requires=[
        'flask',
        'werkzeug>=1.0.1',
        'requests',
        'openeo>=0.5.0a1.*',
        'openeo_processes>=0.0.4',
        'gunicorn>=20.0.1',
        'numpy>=1.17.0,<1.19.0',#tensorflow 2.3 requires numpy <1.19
        'shapely',
        'geopandas~=0.7.0',
        'xarray~=0.16.2',
        'netCDF4~=1.5.4',
        'openeo_udf>=1.0.0rc3',
        'flask-cors',
        'pyproj',
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
