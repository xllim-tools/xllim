#!/bin/bash

# 1st argument = copy directory path = $1
# 2nd argument = backup directory path = $2

echo " -> Remove previous kernelo libraries"
rm *.so
rm $1/*.so

echo " -> Build kernelo in docker container"
docker run -it -d --rm --name kernelo_builder_temp_container -v .:/app kernelo_builder
docker exec -i kernelo_builder_temp_container bash -c "cd /app && python3 setup.py build_ext --inplace -vvv"
docker stop kernelo_builder_temp_container

echo " -> Rename library, copy it to 'dev/planetgllim' directory and 'kernelo/kernelo_library' back-up folder"
mv *.so kernelo_$(date +%F_%T).so
cp *.so $1
cp *.so $2