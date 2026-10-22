"""Package build: pure Python (no compiled extensions); project metadata is in pyproject.toml."""
from setuptools import find_packages, setup

setup(
    packages=find_packages(exclude=("benchmarks", "benchmarks.*")),
    include_package_data=True,
)
