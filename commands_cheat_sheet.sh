#!/bin/bash

# Old kernelo
rm -rf dist/ && rm -rf build/ && /usr/bin/python3 setup.py sdist bdist_wheel -vvv && /usr/bin/pip install dist/kernelo*.whl --force-reinstall


# Build docker images
cd ~/Documents/dev/kernelo-gllim-is
docker build -t "xllim_v2_test_runner" --target runner . --progress=plain
docker build -t "xllim_v2_test_builder" --target builder . --progress=plain
# for verbosity, add "--progress=plain"

# Run docker and connect to its terminal 
docker run -i -t xllim_v2_test_runner bash


# Compile library on host
cd ~/Documents/dev/kernelo-gllim-is/build/debug
sudo cmake -S ../.. -B . -DPYTHON_LIBRARY_DIR="/home/luc/.local/lib/python3.10/site-packages" -DPYTHON_EXECUTABLE="/usr/bin/python3" -DPython3_NumPy_INCLUDE_DIR="/home/luc/.local/lib/python3.10/site-packages/numpy/core/include" -DCMAKE_BUILD_TYPE=Debug
sudo make install

# Compile library within builder docker image
cd ~/Documents/dev/kernelo-gllim-is
sh build_lib_docker.sh /home/luc/.local/lib/python3.10/site-packages /home/luc/Documents/kernelo/libraries_backup/


# Run tests on host
cd ~/Documents/dev/kernelo-gllim-is/build/debug
ctest
ctest -V -R TestModelTest.GetLDimension

cd ~/Documents/dev/kernelo-gllim-is/tests/pythonTests
/usr/bin/python3 gllim_script.py

# Run tests within docker image
cd ~/Documents/dev/kernelo-gllim-is
sh build_lib_docker.sh /home/luc/.local/lib/python3.10/site-packages /home/luc/Documents/kernelo/libraries_backup/
