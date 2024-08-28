#!/bin/bash

# 1st argument = copy directory path = $1
# 2nd argument = backup directory path = $2

echo "\033[35m \n-> Remove previous xllim libraries \033[0m"
rm -rf $1/xllim*.so -v

echo "\033[35m \n-> Run the builder docker container bounded to xllim app \033[0m"
docker run -it -d --rm --name xllim_builder_temp_container -v $(pwd):/home xllim_v2_test_builder

# Idée pour artefact CI : faire sortir la library .so dans .
# on a bien build/ qui est modifié dans host donc pourquoi pas la lib ?

echo "\033[35m \n-> Build xllim in docker container \033[0m"
docker exec -i xllim_builder_temp_container bash -c "\
    cd /home/build && \
    cmake -S .. -B . -DPYTHON_LIBRARY_DIR="/usr/lib/python3/dist-packages" -DCMAKE_BUILD_TYPE=Release && \
    make install "

echo "\033[35m \n-> Copy xllim library to host\033[0m"
docker cp xllim_builder_temp_container:/usr/lib/python3/dist-packages/xllim.cpython-312-x86_64-linux-gnu.so .

echo "\033[35m \n-> Rename library and move it to python host site-packages ($1) and back-up ($2) directories \033[0m"
mv xllim*.so xllim.$(date +%F_%T).so -v
cp *.so $1 -v
cp *.so $2 -v
rm -rf *.so -v

BACK_UP_LIBS=$(ls $2 | wc -l)
echo "\033[33m There are ($BACK_UP_LIBS) back-up xllim libraries in ($2) \033[0m"

echo "\033[35m \n-> Stop the docker container \033[0m"
docker stop xllim_builder_temp_container
