#!/bin/bash
set -e  # exit if any command fails
# trap 'echo "\"${last_command}\" command filed with exit code $?."' EXIT

echo "Setting up symbolic links to test data"
ln -sf cpptest/functionalModel_tests/test_hapke.json .
ln -sf cpptest/functionalModel_tests/test_shkuratov.json .

echo "Bulding cpp tests"
mkdir -p build
cd build
cmake --build .

echo "Running cpp tests ----------"
./generation_test > /dev/null
./learning_test > /dev/null
./main_test > /dev/null
cd -
echo "Cpp tests completed --------"

echo "Building Python module -----"
python3 setup.py build_ext --inplace -vvv
echo "Running Python tests -------"
python3 run.py
echo "Python tests completed -----"
