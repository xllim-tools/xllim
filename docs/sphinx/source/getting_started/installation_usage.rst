.. _installation-usage:

Installation
------------

``xLLiM`` can be installed in several ways depending on your environment and preference.
The recommended approach is via **conda-forge**, which handles all native dependencies automatically.

.. contents:: Installation options
   :local:
   :depth: 1


.. _install-conda:

1. Conda-forge (recommended)
****************************

``xLLiM`` is published on `conda-forge <https://conda-forge.org/>`_, a community-maintained channel that provides
up-to-date, cross-platform packages. The corresponding
`xllim-feedstock <https://github.com/conda-forge/xllim-feedstock>`_ contains the conda packaging recipe
and tracks each release.

Installing a conda distribution
================================

If you do not already have conda installed, several distributions are available:

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - Distribution
     - Description
     - Docs
   * - **Micromamba** ⭐
     - Standalone, fast C++ reimplementation of conda. No base environment, no Python required. Defaults to conda-forge.
     - `mamba.readthedocs.io <https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html>`_
   * - **Miniforge**
     - Community-maintained minimal installer. Defaults to conda-forge.
     - `github.com/conda-forge/miniforge <https://github.com/conda-forge/miniforge>`_
   * - **Miniconda**
     - Minimal installer from Anaconda. Defaults to the ``defaults`` channel (see warning below).
     - `docs.conda.io <https://docs.conda.io/en/latest/miniconda.html>`_
   * - **Anaconda**
     - Full-featured distribution with many pre-installed packages and a GUI. Defaults to ``defaults`` channel (see warning below). Heavier download.
     - `docs.anaconda.com <https://docs.anaconda.com/anaconda/install/>`_

.. note::

   **Recommended: Micromamba or Miniforge.** Both default to conda-forge exclusively, so environments
   are clean and consistent. They are also significantly faster than the classic ``conda`` solver.

Example: installing Micromamba on Linux or macOS:

.. code-block:: bash

   "${SHELL}" <(curl -L micro.mamba.pm/install.sh)

Restart your shell (or ``source ~/.bashrc``), then verify:

.. code-block:: bash

   micromamba --version

On Windows, see the `Micromamba installation docs <https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html>`_.

.. warning::

   **Anaconda/Miniconda users: channel conflict.** Anaconda and Miniconda default to Anaconda's proprietary
   ``defaults`` channel, which is **not** the same as conda-forge. Mixing channels can lead to subtle
   dependency conflicts and broken environments, because the two channels build packages independently and
   their ABI guarantees are not always compatible.

   If you use Anaconda or Miniconda, create a **conda-forge-only** environment before installing ``xLLiM``:

   .. code-block:: bash

      conda create -n <my-env> -c conda-forge --override-channels
      conda activate <my-env>
      conda config --env --add channels conda-forge
      conda config --env --set channel_priority strict

   See the `conda-forge documentation on channel conflicts <https://mamba.readthedocs.io/en/latest/user_guide/troubleshooting.html#defaults-channels>`_ for more details.

Installing xLLiM
=================

Create and activate an environment (``<my-env>`` is a placeholder), then install ``xLLiM``:

.. code-block:: bash

   # With conda or mamba
   conda create -n <my-env>
   conda activate <my-env>
   conda install -c conda-forge xllim

.. code-block:: bash

   # With micromamba
   micromamba create -n <my-env>
   micromamba activate <my-env>
   micromamba install -c conda-forge xllim


.. _install-pypi:

2. PyPI
*******

Pre-built wheels are available for **Linux x86_64** (manylinux_2_28) with **Python 3.11** and **3.12**:

.. code-block:: bash

   pip install xllim

**Mac/Windows:** conda (section :ref:`install-conda`) or Docker (section :ref:`install-docker`) are recommended.

**Other Python versions (3.10, 3.13…):** not officially tested. You can try building from the source
distribution — ``pip`` will fall back to it automatically when no wheel matches. However, you must first
install the native system libraries manually (see section :ref:`install-manual`).


.. _install-docker:

3. Docker
*********

A minimal Docker image based on ``python:3.11-slim`` is available, with ``xLLiM`` pre-installed.

**Prerequisites:** install `Docker Engine <https://docs.docker.com/engine/install/>`_ (Linux/macOS) or
`Docker Desktop <https://docs.docker.com/desktop/windows/install/>`_ (Windows).

Pull the image:

.. code-block:: bash

   docker pull ghcr.io/xllim-tools/xllim/xllim:latest

Run your project by mounting your source directory:

.. code-block:: bash

   docker run -it \
     -v $(pwd):/workspace \
     -w /workspace \
     ghcr.io/xllim-tools/xllim/xllim:latest \
     python main.py

Open an interactive shell:

.. code-block:: bash

   docker run -it -v $(pwd):/workspace -w /workspace ghcr.io/xllim-tools/xllim/xllim:latest bash

.. note::

   Specific versions of the image are also available. See the
   `xLLiM GHCR registry <https://github.com/xllim-tools/xllim/pkgs/container/xllim%2Fxllim/versions>`_
   for the full list.


.. _install-manual:

4. Manual installation (build from source)
******************************************

Required only for unsupported platforms, Python versions, or development purposes.

Build dependencies
==================

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Name
     - Version
     - Notes
   * - C++ compiler
     - C++17 support
     - System-installed (e.g., g++ >= 9, clang++ >= 5)
   * - CMake
     - >= 3.21
     - Auto-installed by pip
   * - Ninja
     - any
     - Auto-installed by pip
   * - scikit-build-core
     - >= 0.7.0, < 1
     - Auto-installed by pip
   * - Python
     - >= 3.11, < 3.13
     - System-installed
   * - Numpy
     - >= 2
     - Auto-installed by pip
   * - Pybind11
     - >= 2.12
     - Auto-installed by pip
   * - Armadillo
     - >= 12.6, < 13
     - System-installed
   * - Boost
     - >= 1.78, < 2
     - System-installed; components: ``system``, ``thread``, ``random``
   * - BLAS / LAPACK
     - any
     - System-installed (OpenBLAS, MKL, Apple Accelerate…)
   * - Carma
     - >= 0.8.0
     - System-installed (see below)

Build steps (Debian/Ubuntu)
============================

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/xllim-tools/xllim.git
      cd xllim

2. Install system dependencies:

   .. code-block:: bash

      sudo apt update

      # Compilation tools
      sudo apt-get install -y --no-install-recommends g++ cmake ninja-build

      # BLAS/LAPACK (OpenBLAS shown; any conforming implementation works)
      sudo apt-get install -y --no-install-recommends libopenblas-dev liblapack-dev

      # Armadillo
      sudo apt-get install -y --no-install-recommends libarmadillo-dev

      # Python development headers
      sudo apt-get install -y --no-install-recommends python3-dev

      # Boost (components: system, thread, random)
      sudo apt-get install -y --no-install-recommends libboost-dev libboost-system-dev libboost-thread-dev libboost-random-dev

   .. warning::

      **Boost version:** ``xLLiM`` requires Boost >= 1.78.0 (enforced by CMake). Ubuntu 24.04+ ships a
      compatible version out of the box. On older distributions (e.g., Ubuntu 22.04 ships 1.74), build
      Boost from source:

      .. code-block:: bash

         curl -L https://archives.boost.io/release/1.78.0/source/boost_1_78_0.tar.gz -o boost.tar.gz
         tar -xzf boost.tar.gz
         cd boost_1_78_0
         ./bootstrap.sh --prefix=/usr/local
         ./b2 --with-system --with-thread --with-random link=static variant=release -j$(nproc) install
         cd .. && rm -rf boost_1_78_0 boost.tar.gz

      Then set ``CMAKE_ARGS`` so CMake finds this installation instead of the system one:

      .. code-block:: bash

         export CMAKE_ARGS="-DBoost_ROOT=/usr/local -DBoost_NO_SYSTEM_PATHS=ON -DBoost_USE_STATIC_LIBS=ON"

   **carma** — not available in standard apt repositories; build from source:

   .. code-block:: bash

      curl -L https://github.com/RUrlus/carma/archive/refs/tags/v0.8.0.tar.gz -o carma.tar.gz
      tar -xzf carma.tar.gz
      cd carma-0.8.0 && mkdir build && cd build
      cmake -DCARMA_INSTALL_LIB=ON ..
      cmake --build . --config Release --target install
      cd ../..
      rm -rf carma-0.8.0 carma.tar.gz

   .. tip::

      If you are using conda, all of the above can be installed in one step:

      .. code-block:: bash

         conda install armadillo boost libblas liblapack carma

3. Build and install:

   .. code-block:: bash

      pip install .

4. Verify the installation:

   .. code-block:: python

      import xllim

Manual CMake build (for development)
======================================

For C++ development or debugging, drive CMake directly:

.. code-block:: bash

   cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DXLLIM_BUILD_TESTS=ON
   cmake --build build
   ctest --test-dir build
