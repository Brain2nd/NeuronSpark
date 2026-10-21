"""Package build: pure Python (no compiled extensions); project metadata is in pyproject.toml."""
from setuptools import find_packages, setup

setup(
    packages=find_packages(exclude=("tests", "tests.*", "benchmarks", "benchmarks.*")),
    # the license and provenance record of the vendored mamba_ssm kernels travel with them
    package_data={"neuronspark.ops.triton.mamba": ["LICENSE", "PROVENANCE.md"]},
    include_package_data=True,
)
