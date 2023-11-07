# Available at setup time due to pyproject.toml
from glob import glob
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

__version__ = "0.0.1"

ext_modules = [
    Pybind11Extension("kernelo",
        sorted(glob("src/*.cpp")),
        # Example: passing in the version to the compiled code
        define_macros = [('VERSION_INFO', __version__)],
        extra_compile_args=[
                '-Ofast', '-DARMA_NO_DEBUG', '-std=c++14'
        ]
        ),
]

setup(
    name="kernelo_lib",
    version=__version__,
    description="A test project using pybind11",
    long_description="",
    ext_modules=ext_modules,
    extras_require={"test": "pytest"},
    # Currently, build_ext only provides an optional "highest supported C++
    # level" feature, but in the future it may provide more features.
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
)