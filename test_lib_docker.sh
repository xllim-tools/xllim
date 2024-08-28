#!/bin/bash

# 1st argument = copy directory path = $1
# 2nd argument = backup directory path = $2

echo "\033[35m \n-> Run the builder docker container bounded to xllim app \033[0m"
docker run -it -d --rm --name xllim_builder_temp_container -v $(pwd):/home xllim_v2_test_builder


echo "\033[35m \n-> Build xllim in docker container \033[0m"
docker exec -i xllim_builder_temp_container bash -c "cd /home/build && ctest"
docker exec -i xllim_builder_temp_container bash -c "cd /home/build && ctest -V -R ShkuratovModelTest.GetLDimension"

echo "\033[35m \n-> Stop the docker container \033[0m"
docker stop xllim_builder_temp_container