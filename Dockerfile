FROM ubuntu:jammy AS runner
RUN apt-get update
# install kernelo dependencies
# RUN apt-get install -y --no-install-recommends \
# 	python3 python3-numpy \
# 	libatlas3-base libarmadillo9
RUN apt-get install -y --no-install-recommends python3
RUN apt-get install -y --no-install-recommends python3-numpy
RUN apt-get install -y --no-install-recommends python3-pybind11
# RUN apt-get install -y --no-install-recommends libatlas3-base
RUN apt-get install -y --no-install-recommends libarmadillo10
RUN apt-get install -y --no-install-recommends libgomp1
RUN ls
# COPY *.so /usr/lib/python3/dist-packages/
# Copy  some python script to test xllim in docker container
RUN mkdir /home/pythonTests && mkdir /home/dataRef
COPY tests/pythonTests/*_script.py /home/pythonTests/
COPY tests/dataRef /home/dataRef
RUN ls /home/pythonTests

# install kernelo build dependencies
FROM runner AS builder
# RUN apt-get install -y --no-install-recommends gcc g++ cmake make \
# 	python3-dev cython3 python3-numpy python3-pip \
# 	libatlas-base-dev libarmadillo-dev libboost-dev
RUN apt-get install -y --no-install-recommends g++
RUN apt-get install -y --no-install-recommends cmake
RUN apt-get install -y --no-install-recommends make
RUN apt-get install -y --no-install-recommends python3-dev
RUN apt-get install -y --no-install-recommends python3-numpy
# RUN apt-get install -y --no-install-recommends python3-pip
RUN apt-get install -y --no-install-recommends python3-pybind11
# RUN apt-get install -y --no-install-recommends libatlas-base-dev
RUN apt-get install -y --no-install-recommends libopenblas-dev liblapack-dev libarpack2-dev libsuperlu-dev
RUN apt-get install -y --no-install-recommends libarmadillo-dev
RUN apt-get install -y --no-install-recommends libboost-dev
# TODO try to not use boost library

# RUN apt list --installed

##################################################
# replaced by gitlab ci script

# RUN mkdir /usr/xllim_lib
# WORKDIR /usr/xllim_lib
# RUN ls
# COPY . .

# # RUN cd extern/carma && mkdir build
# # # optionally with -DCMAKE_INSTALL_PREFIX:PATH=
# # RUN cd extern/carma/build && cmake -DCARMA_INSTALL_LIB=ON ..
# # RUN cd extern/carma/build && cmake --build . --config Release --target install

# RUN rm -rf build && mkdir build
# RUN cd build
# RUN ls /usr/lib/python3/dist-packages
# RUN cmake -S . -B build -DPYTHON_LIBRARY_DIR="/usr/lib/python3/dist-packages" -DCMAKE_BUILD_TYPE=Release
# #-DPYTHON_EXECUTABLE="/usr/bin/python3" -DPython3_NumPy_INCLUDE_DIR="/home/luc/.local/lib/python3.10/site-packages/numpy/core/include" -DCMAKE_BUILD_TYPE=Debug
# RUN cd build && make install

# # install sonarcube and it's dependencies
# FROM builder AS coveragetester
# RUN apt-get install -y --no-install-recommends unzip wget nodejs pylint
# RUN wget --no-check-certificate https://binaries.sonarsource.com/Distribution/sonar-scanner-cli/sonar-scanner-cli-4.7.0.2747-linux.zip
# RUN unzip sonar-scanner-cli-4.7.0.2747-linux.zip
# ENV PATH "$PATH:/sonar-scanner-4.7.0.2747-linux/bin/"
# RUN pip3 install gcovr
# # prefetch sonar-scanner dependencies
# # TODO
