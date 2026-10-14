from pathlib import Path

from setuptools import find_packages, setup
from setuptools_rust import RustExtension

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

with Path(__file__).with_name("Cargo.toml").open("rb") as manifest:
    crate_version = tomllib.load(manifest)["package"]["version"]

setup(
    name="bevy_zeroverse_ffi",
    version=crate_version,
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    rust_extensions=[
        RustExtension("bevy_zeroverse_ffi")
    ],
    include_package_data=True,
    zip_safe=False,
)
