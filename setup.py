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
        'werkzeug',
        'requests',
        'openeo>=0.3.0a1.*',
        'gunicorn==19.9.0',
        'shapely',
        'geopandas==0.6.2',
        'xarray==0.12.3',
        'netCDF4==1.5.1.2',
        'openeo_udf>=0.0.9.post0',
        'flask-cors',
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
