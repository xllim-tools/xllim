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
COPY tests/pythonTests/*_script.py /home/pythonTests/
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

# python3-pip is for documention and pip install sphinx
RUN apt-get install -y --no-install-recommends python3-pip
RUN pip3 install -U sphinx sphinx-rtd-theme

# ! Boost required but only used for boost/property_tree
RUN apt-get install -y --no-install-recommends libboost-dev


FROM runner AS minimal

# Download and place xllim shared library in python global dist packages
RUN apt update
RUN apt install -y --no-install-recommends curl
RUN curl --location -o artefact.zip "https://gitlab.inria.fr/api/v4/projects/kernelo-mistis%2Fkernelo-gllim-is/jobs/artifacts/v2/download?job=build_job"
RUN unzip artefact.zip
RUN rm artefact.zip
RUN mv *.so /usr/lib/python3/dist-packages/xllim.so