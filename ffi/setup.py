from setuptools import find_packages, setup
from setuptools_rust import RustExtension


setup(
    name="bevy_zeroverse_ffi",
    version="0.1",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    rust_extensions=[
        RustExtension("bevy_zeroverse_ffi")
    ],
    include_package_data=True,
    zip_safe=False,
)
