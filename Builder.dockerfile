# syntax=docker/dockerfile:1
FROM scratch
MAINTAINER stan.borkowski
USER root

FROM ubuntu:focal AS runner
RUN apt-get update
# install kernelo dependencies
RUN apt-get install -y --no-install-recommends \
	python3 python3-numpy \
	libatlas3-base libarmadillo9

# install kernelo build dependencies
FROM runner AS builder
RUN apt-get update
RUN apt-get install -y --no-install-recommends gcc g++ cmake \
	python3-dev cython3 python3-numpy python3-pip \
	libatlas-base-dev libarmadillo-dev libboost-dev
RUN pip3 install cyarma

# install sonarcube and it's dependencies
FROM builder AS tester
RUN apt-get update
RUN apt-get install -y --no-install-recommends unzip wget nodejs pylint
RUN wget --no-check-certificate https://binaries.sonarsource.com/Distribution/sonar-scanner-cli/sonar-scanner-cli-4.7.0.2747-linux.zip
RUN unzip sonar-scanner-cli-4.7.0.2747-linux.zip
ENV PATH "$PATH:/sonar-scanner-4.7.0.2747-linux/bin/"

# prefetch sonar-scanner dependencies
# TODO

FROM builder AS buildso
RUN python3 setup.py build_ext --inplace -vvv

# run build and compile tests 
FROM tester AS runtests
RUN mkdir -p build; cd build; cmake ../; cmake --build .