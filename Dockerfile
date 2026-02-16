# syntax=docker/dockerfile:1


FROM ubuntu:jammy AS base_image

# Install xllim run-time dependencies
RUN apt-get update
RUN apt-get install -y --no-install-recommends \
	apt-utils \
	python3 \
	python3-numpy \
	python3-pybind11 \
	python3-h5py \
	libarmadillo10 \
	libgomp1 \
	libpython3.10

# (optional) Copy some python scripts to test xllim in docker container
RUN mkdir /home/pythonTests && mkdir /home/dataRef
COPY tests/pythonTests /home/pythonTests/
COPY tests/dataRef /home/dataRef


FROM base_image AS runner
COPY xllim*.so /usr/lib/python3/dist-packages/xllim.so
COPY xllim_cli.py /usr/bin/xllim_cli
ENTRYPOINT [ "/usr/bin/xllim_cli" ]


FROM base_image AS builder

# Install xllim compilation dependencies
RUN apt-get update

# Install compilation tools
RUN apt-get install -y --no-install-recommends g++ cmake make

# Install Armadillo-related dependencies
RUN apt-get install -y --no-install-recommends libopenblas-dev liblapack-dev libarpack2-dev libsuperlu-dev libarmadillo-dev

# Install Python-related dependencies
RUN apt-get install -y --no-install-recommends python3-dev python3-pip

# Install documention related packages
RUN pip install -U sphinx sphinx-rtd-theme myst_parser sphinx-copybutton

# Install python tests related packages
RUN pip install -U pytest

# ! Boost required but only used for boost/property_tree
RUN apt-get install -y --no-install-recommends libboost-dev
