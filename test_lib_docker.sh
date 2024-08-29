#!/bin/bash

# 1st argument = copy directory path = $1
# 2nd argument = backup directory path = $2

echo "\033[35m \n-> Run the builder docker container bounded to xllim app \033[0m"
docker run -it -d --rm --name xllim_builder_temp_container -v $(pwd):/home xllim_v2_jammy_builder


echo "\033[35m \n-> Build xllim in docker container \033[0m"
# docker exec -i xllim_builder_temp_container bash -c "cd /home/build && ctest"
docker exec -i xllim_builder_temp_container bash -c "cd /home/build && ctest -V -R TestModelTest.GetLDimension"
# docker exec -i xllim_builder_temp_container bash -c "cd /home/build && ctest -V -R PerformanceTest.TestModel"

docker cp /home/luc/.local/lib/python3.10/site-packages/xllim.so xllim_builder_temp_container:/usr/lib/python3/dist-packages/xllim.so
docker exec -i xllim_builder_temp_container bash -c "cd /home/tests/pythonTests && python3 gllim_script.py"

echo "\033[35m \n-> Stop the docker container \033[0m"
docker stop xllim_builder_temp_container