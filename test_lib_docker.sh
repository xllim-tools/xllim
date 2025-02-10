#!/bin/bash

# --image option = Docker image name with tag in which the tests will be runned
# --lib-path option = path to the xllim shared linrary .so used for the Python tests

# Parse options
ARGS=$(getopt -o "" --long image:,lib-path: -- "$@")
eval set -- "$ARGS"

# Default values
image=""
lib_path=""

# Read the arguments
while true; do
  case "$1" in
    --image) image="$2"; shift 2 ;;
    --lib-path) lib_path="$2"; shift 2 ;;
    --) shift; break ;;  # End of options
    *) break ;;
  esac
done


echo "\033[35m \n-> Run the builder docker container bounded to xllim app : $image \033[0m"
docker run -it -d --rm --name xllim_builder_temp_container -v $(pwd):/home $image


echo "\033[35m \n-> CPP unit tests \033[0m"
docker exec -i xllim_builder_temp_container bash -c "cd /home/build && ctest"
# docker exec -i xllim_builder_temp_container bash -c "cd /home/build && ctest -V -R TestModelTest.GetLDimension"

echo "\033[35m \n-> Python integration tests with the xllim compiled lib at $lib_path \033[0m"
docker cp $lib_path xllim_builder_temp_container:/usr/lib/python3/dist-packages/xllim.so
docker exec -i xllim_builder_temp_container bash -c "cd /home/tests/pythonTests && pytest -v --tb=short --color=yes" # -s for stdout and prints

echo "\033[35m \n-> Stop the docker container \033[0m"
docker stop xllim_builder_temp_container