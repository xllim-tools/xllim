# syntax=docker/dockerfile:1


FROM ubuntu:jammy AS runner

# Install xllim run-time dependencies
RUN apt-get update
RUN apt-get install -y --no-install-recommends \
	apt-utils \
	python3 \
	python3-numpy \
	python3-pybind11 \
	libarmadillo10 \
	libgomp1

# (optional) Copy some python scripts to test xllim in docker container
RUN mkdir /home/pythonTests && mkdir /home/dataRef
COPY tests/pythonTests /home/pythonTests/
COPY tests/dataRef /home/dataRef


FROM runner AS builder

# Install xllim compilation dependencies
RUN apt-get update

# Install compilation tools
RUN apt-get install -y --no-install-recommends g++ cmake make

# Install Armadillo-related dependencies
RUN apt-get install -y --no-install-recommends libopenblas-dev liblapack-dev libarpack2-dev libsuperlu-dev libarmadillo-dev

# Install Python-related dependencies
RUN apt-get install -y --no-install-recommends python3-dev

# Set timezone non-interactively for python3-sphinx
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata

# Install documention related packages
RUN apt-get install -y --no-install-recommends python3-sphinx python3-sphinx-rtd-theme

# Install python tests related packages
RUN apt-get install -y --no-install-recommends python3-pytest


# ! Boost required but only used for boost/property_tree
RUN apt-get install -y --no-install-recommends libboost-dev
