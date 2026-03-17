
# xLLiM

`xLLiM` is a C++ with Python bindings implementation of Gaussian Locally-Linear Mapping.
It can be used to perform efficient inversion of high-dimensional models. 
`xLLiM` integrates features such as:
* Forward model functionals. These can be implemented as C++ or as pure Python functions. The forward model can be used to generate data following a distribution and refine GLLiM results by sampling the PDF using various strategies.
* Multi-initialization options
* Post-GLLiM refinement methods such as Importance Sampling (IS) and Iterative Mixture Importance Sampling (IMIS)
* Post-processing analysis, including confidence quantification on predictions, detection of multiple solutions, and permutation of predictions in case of signal regularity.
Currently, `xLLiM` supports only Gaussian probability distributions, but other distributions may be added to it in the future.

`xLLiM` is distributed as a compiled shared library and can be installed via conda, pip, or as a Docker container. It is integrated in the [Planet-GLLiM](https://gitlab.inria.fr/xllim/planet-gllim) astrophysics application, that is distributed as a Docker image for local use and as a data processing service
on [Allgo-18](https://allgo18.inria.fr/).

> **Repository migration:** `xLLiM` has moved to GitHub. The current repository is [https://github.com/xllim-tools/xllim](https://github.com/xllim-tools/xllim). The former GitLab repository ([https://gitlab.inria.fr/xllim/xllim](https://gitlab.inria.fr/xllim/xllim)) is no longer maintained.

The `xLLiM` method was previously implemented in `R` and is available on [CRAN](https://cran.r-project.org/web/packages/xLLiM/index.html).
Other implementations of GLLiM also exist:
- A pure Python implementation: [pyGLLiM](https://github.com/Chutlhu/pyGLLiM)
- A Julia implementation with IS and IMIS: [Fast Bayesian Inversion](https://gitlab.inria.fr/bkugler/fastbayesianinversion)

This `xLLiM` implementation is derived from [Kernelo](https://gitlab.inria.fr/kernelo-mistis/kernelo-gllim-is).
`Kernelo` is obsolete and is no longer maintained.

## Table of Contents

- [Documentation and API reference](#documentation-and-api-reference)
- [Runtime Dependencies](#runtime-dependencies)
  - [Mandatory](#mandatory)
  - [Optional](#optional)
- [xLLiM installation options](#xllim-installation-options)
  - [1. Conda-forge (recommended)](#1-conda-forge-recommended)
  - [2. PyPI](#2-pypi)
  - [3. Docker](#3-docker)
  - [4. Manual installation (optional)](#4-manual-installation-optional)
- [Licence](#licence)

# Documentation and API reference

The API reference of the xLLiM module is available at [xLLiM API reference](https://xllim-tools.github.io/xllim/). 
For more information you can find a complete scientific documentation in the [Planet-GLLiM documentation](https://xllim.gitlabpages.inria.fr/planet-gllim/).

# Runtime Dependencies

## Mandatory:
These packages are required for the core functionality of the library.
| Name                 | version           |
|----------------------|-------------------|
| Python               | >=3.11, < 3.13    |
| Numpy                | >=2.0, < 3        |
| h5py                 | >=3.8, < 4        |

For other python versions, see [Other python versions](#other-python-versions).

## Optional:

### ENVI / GDAL Support
xllim requires GDAL specifically for ENVI hyperspectral file support. Because GDAL depends on native system libraries, installation can be complex.

#### Recommended (Most reliable)
Using Conda is the simplest way to handle linked C libraries:
```bash
conda install gdal
```

#### Alternative: pip + system package (Advanced users)
If installing via apt or brew, the Python GDAL version must match the system GDAL version.
If the versions do not match, the installation will fail.

Example on Ubuntu:

1. Install system libraries
```bash
sudo apt install libgdal-dev gdal-bin
```
2. Check the system version
```bash
gdal-config --version
```
3. Install the matching python wrapper (e.g., if version is 3.8.4)
```bash
pip install "gdal==3.8.4"
```

# xLLiM installation options

`xLLiM` can be installed and used in several ways, depending on your environment and preference.

## 1. Conda-forge (recommended)
The easiest way to install `xLLiM` is via conda.

First create and activate an environment if you don't already have one:
```bash
conda create -n <my-env> python=3.12
conda activate <my-env>
```
(you can use python 3.11 or 3.12)

Then install `xLLiM`. This will also install the required dependencies needed.
```bash
conda install xllim
```

## 2. PyPI

#### Available wheels

We've built and tested wheels for **Linux x86_64** (manylinux_2_28) - **Python 3.11** and **3.12**. If you're in one of these configurations, you can install xllim easily with:

```bash
pip install xllim
```

#### Mac/Windows

If you're on Mac or Windows, and are fine with Python 3.11/12, we recommend you either:
- install with conda ([Section 1.](#1-conda-forge-recommended))
- use our pre-built docker image ([Section 3.](#3-docker))

#### Other Python versions

If you want to use a different Python version (3.10, 3.13, 3.14...):
They've not been tested as some of `xLLiM` dependencies exclude these versions. It might be possible though. Then you may try:
- Building from the sdist (```pip install xllim```): when no pre-built wheel matches your Python version or platform, pip automatically falls back to downloading the source distribution and compiling the C++ extension on your machine. The Python-side build tools (scikit-build-core, cmake, ninja, numpy) are installed automatically. However, you must first install the native system libraries manually (compiler, Armadillo, OpenBLAS, Boost, LAPACK — see [Section 4.](#4-manual-installation-optional) for details). If those are present, the build should succeed.
.
- Compiling, building and installing from scratch (advanced users) ([Section 4.](#4-manual-installation-optional))


## 3. Docker

A minimal Docker image is available based on python:3.11-slim, which already includes xLLiM installed via the wheel generated by our CI pipeline:

### Prerequisite
Docker Engine is available on a variety of Linux platforms, macOS and Windows 10. You can find the installation documentation on [Docker's documentation website](https://docs.docker.com/)
#### Linux & MacOs
* [Docker Engine](https://docs.docker.com/engine/) ([Installation instructions](https://docs.docker.com/engine/install/))
* [Docker Compose](https://docs.docker.com/compose/) ([Installation instructions](https://docs.docker.com/compose/install/))

#### Windows
* Docker Desktop that contains everything to run Docker ([Installation instructions](https://docs.docker.com/desktop/windows/install/)).

### Using the Docker image

Pull the image:

```bash
docker pull ghcr.io/xllim-tools/xllim/xllim:latest
```
> Note: you can also pull specific versions of xLLiM. Please refer to [xLLiM GHCR](https://github.com/xllim-tools/xllim/pkgs/container/xllim%2Fxllim/versions).

Run your project by mounting your source directory:

```bash
docker run -it \
  -v $(pwd):/workspace \
  -w /workspace \
  ghcr.io/xllim-tools/xllim/xllim:latest \
  python main.py
```
This allows you to use `xLLiM` without manually managing dependencies or Python environments.

You can also open an interactive shell:

```bash
docker run -it -v $(pwd):/workspace -w /workspace ghcr.io/xllim-tools/xllim/xllim:latest bash
```

### The Jupyter notebook image

The image is built from the official [jupyter/scipy-notebook](https://jupyter-docker-stacks.readthedocs.io/en/latest/) image adapted to xLLiM dependencies. This image offers the familiar JupyterLab user interface within a Python-based datascience environment.

#### First steps

1. Connect to Inria's GitLab
```
docker login registry.gitlab.inria.fr
```
2. Get the Dockerfile. You achieve this by using curl or wget.
```
curl --location -o jupyter.Dockerfile "https://gitlab.inria.fr/xllim/xllim/-/raw/master/jupyter.Dockerfile?ref_type=heads&inline=false"
```
3. Build the docker image named *xllim_jupyter_notebook*
```
docker build -f jupyter.Dockerfile -t "xllim_jupyter_notebook" --no-cache-filter install .
```
4. Create and run the container *xllim_notebook* and bind the volume to your current working directory.
```
docker run -it --name xllim_notebook -p 8888:8888 -v "${PWD}":/home/jovyan/work xllim_jupyter_notebook
```
5. You can find the JupyterLab server address (*http://127.0.0.1:8888/lab?token=[some-token]*) in the logs. Make sure there is not another Jupyter server running. Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
Have fun !

#### Use your container

Once your container is set up it is very easy to use your xLLiM environment. All changes made to the docker container (installing packages, etc.) are **persistent**. Be careful not to delete your container, otherwise all modifications made within it would be lost. You can start and stop the container with the two simple commands below:
```
docker stop xllim_notebook
docker start xllim_notebook
```
#### ⚠️ File permission issues

Depending on your OS and Docker version you may face permission issues on the mounted volume (/home/jovyan/work/ directory).
You can overcome this issue by granting file access to the virtual user (*jovyan*) when creating the container.
```
docker run -it --name xllim_notebook -p 8888:8888 -v "${PWD}":/home/jovyan/work xllim_jupyter_notebook bash -c "chown -R jovyan:users /home/jovyan/work && start-notebook.py"
```
After stopping the Docker container you may need to grant back file access to your host machine's user.
```
sudo chown -R $(id -u):$(id -g) "${PWD}"
```

## 4. Manual installation (optional)
If you need to compile `xLLiM` from source (e.g., for an unsupported platform or Python version, or for development), please refer to the requirements below.

### Build Dependencies

These are required only if you are compiling the library from source.
| Name                 | Version           | Notes                                                              |
|----------------------|-------------------|--------------------------------------------------------------------|
| C++ compiler         | C++17 support     | System-installed (e.g., g++ >= 9, clang++ >= 5)                   |
| CMake                | >= 3.21           | Auto-installed by pip when using `pip install .`                   |
| Ninja                | any               | Auto-installed by pip when using `pip install .`                   |
| Python               | >= 3.11, < 3.13   | System-installed; see [Other Python versions](#other-python-versions) for other versions |
| Armadillo            | >= 12.6, < 13     | System-installed                                                   |
| Boost                | >= 1.78, < 2      | System-installed; components: `system`, `thread`, `random`         |
| OpenBLAS             | >= 0.3.15, < 1    | System-installed (BLAS + LAPACK backends)                          |
| Pybind11             | 2.13.6            | Vendored as Git submodule — no manual install needed               |
| Carma                | 0.8.0             | Vendored as Git submodule — no manual install needed               |

### Build from source

Since `xLLiM` contains C++ extensions with Python bindings, building from source requires native system libraries that pip cannot install automatically. The instructions below use a Debian/Ubuntu-based system as example - adapt package names for your distribution.

> **Note:** Pybind11 and Carma are included as Git submodules in `extern/` and do not need to be installed separately.

1. Clone the project with submodules
```bash
git clone --recurse-submodules https://github.com/xllim-tools/xllim.git
cd xllim
```

2. Install system dependencies
```bash
sudo apt update

# Compilation tools
sudo apt-get install -y --no-install-recommends g++ cmake ninja-build

# Armadillo and linear algebra backends
sudo apt-get install -y --no-install-recommends libopenblas-dev liblapack-dev libarpack2-dev libsuperlu-dev libarmadillo-dev

# Python development headers
sudo apt-get install -y --no-install-recommends python3-dev

# Boost (components: system, thread, random)
sudo apt-get install -y --no-install-recommends libboost-dev libboost-system-dev libboost-thread-dev libboost-random-dev
```

> ⚠️ **Boost version:** `xLLiM` requires Boost >= 1.78. Ubuntu 24.04+ ships a compatible version out of the box. On older distributions (e.g., Ubuntu 22.04 ships 1.74), you will need to build Boost >= 1.78 from source — see `.github/workflows/build_publish.yml` for an example.

3. Build and install
```bash
pip install .
```
This uses [scikit-build-core](https://scikit-build-core.readthedocs.io/) to drive the CMake build automatically. The Python-side build tools (`scikit-build-core`, `cmake`, `ninja`, `numpy`) are fetched by pip as needed.

4. Verify the installation
```
python3
>>> import xllim
```

#### Manual CMake build (for development)

For C++ development or debugging, you can drive CMake directly:
```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DXLLIM_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build
```
The build type can be either *Debug* (`-O0 -g`) or *Release* (`-O2 -DNDEBUG`). To install the compiled module into your Python environment, use `pip install .` as described above.

# Licence
This software is licensed under the [BSD 3-Clause License](LICENCE.txt).
While the software is free, we would appreciate it if you send us an email at `xllim-contact@inria.fr` to let us know how you use it.
Also, please contact us if the licence does not meet your needs.
