from setuptools import setup, find_packages

setup(
    name='openassetpricing',
    version='0.0.1',
    author='Peng Li, Andrew Chen, Tom Zimmermann',
    author_email='pl750@bath.ac.uk, andrew.y.chen@frb.gov, tom.zimmermann@uni-koeln.de',
    license='GPLv2',
    packages=find_packages(),
    install_requires=[
        'polars',
        'pandas',
        'requests',
        'tabulate',
        'wrds',
        'pyarrow',
        'beautifulsoup4'
    ],
)
