from setuptools import setup, find_packages

setup(
    name='openeo_driver',
    packages=find_packages(include=['openeo*']),
    include_package_data=True,
    install_requires=[
        'flask',
    ],
)