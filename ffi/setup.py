import os
from setuptools import find_packages, setup
from setuptools_rust import RustExtension


# TODO: remove tch-rs dependency and output raw bytes (do tensor conversion in python, this has the overhead of GPU -> CPU -> GPU transfer)
os.environ["LIBTORCH_USE_PYTORCH"] = "1"

libtorch_path = os.popen("python -c 'import site; print(site.getsitepackages()[0] + \"/torch/lib\")'").read().strip()
os.environ["LD_LIBRARY_PATH"] = libtorch_path + ":" + os.environ.get("LD_LIBRARY_PATH", "")


setup(
    name="bevy_zeroverse",
    version="0.1",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    rust_extensions=[
        RustExtension("bevy_zeroverse")
    ],
)
