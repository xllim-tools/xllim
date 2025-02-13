#!/bin/bash

# --image option = Docker image name with tag in which the tests will be runned
# --lib-dir option = path to directory where the xllim shared linrary .so will be installed
# --backup-dir option = path to directory in which a backup of the library will be saved with date and time.


# Parse options
ARGS=$(getopt -o "" --long image:,lib-dir:,backup-dir: -- "$@")
eval set -- "$ARGS"

# Default values
image=""
lib_dir=""
backup_dir=""

# Read the arguments
while true; do
  case "$1" in
    --image) image="$2"; shift 2 ;;
    --lib-dir) lib_dir="$2"; shift 2 ;;
    --backup-dir) backup_dir="$2"; shift 2 ;;
    --) shift; break ;;  # End of options
    *) break ;;
  esac
done

# Ensure paths DO NOT end with '/'
if [ -n "$lib_dir" ] && [ "${lib_dir%/}" != "$lib_dir" ]; then
  lib_dir="${lib_dir%/}"
fi
if [ -n "$backup_dir" ] && [ "${backup_dir%/}" != "$backup_dir" ]; then
  backup_dir="${backup_dir%/}"
fi



echo "\033[35m \n-> Run the builder docker container bounded to xllim app : $image \033[0m"
docker run -it -d --rm --name xllim_builder_temp_container -v $(pwd):/home $image


echo "\033[35m \n-> Build xllim in docker container \033[0m"
docker exec -i xllim_builder_temp_container bash -c "\
    cd /home/build && \
    cmake -S .. -B . -DPYTHON_LIBRARY_DIR="/usr/lib/python3/dist-packages" -DCMAKE_BUILD_TYPE=Release && \
    make install "


echo "\033[35m \n-> Copy xllim library to host python site-packages at $lib_dir \033[0m"
file="xllim.cpython-310-x86_64-linux-gnu.so"
# file="xllim.cpython-312-x86_64-linux-gnu.so" # depending on python version
docker cp xllim_builder_temp_container:/usr/lib/python3/dist-packages/$file $lib_dir


echo "\033[35m \n-> Rename library and move it to back-up ($backup_dir) directories \033[0m"
cp $lib_dir/$file $backup_dir/${file%.so}.$(date +%F_%T).so -v
BACK_UP_LIBS=$(ls $backup_dir | wc -l)
echo "\033[33m There are ($BACK_UP_LIBS) back-up xllim libraries in ($backup_dir) \033[0m"


echo "\033[35m \n-> Stop the docker container \033[0m"
docker stop xllim_builder_temp_container
