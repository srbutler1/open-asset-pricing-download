from setuptools import setup, find_packages

setup(
    name='openassetpricing',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'polars',
        'pandas',
        'requests',
        'tabulate'
    ],
)
