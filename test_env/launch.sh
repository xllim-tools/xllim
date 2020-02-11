#!/bin/bash

jupyter serverextension enable --py jupyterlab --sys-prefix

mkdir -p root
cd root

ls /builds/fs_djouadi/kernelo/test_env

curl --output kernelo.cpython-37m-x86_64-linux-gnu.so --location --header "PRIVATE-TOKEN:e1eoD8G9B8MmRMitas_y" "https://gitlab.com//api/v4/projects/16161938/jobs/artifacts/cicd_config/raw/kernelo.cpython-37m-x86_64-linux-gnu.so?job=build"

cp ../test_hapke.json ./test_hapke.json
cp ../test_script.ipynb ./test_script.ipynb


export JUPYTER_CONFIG_DIR=/app
jupyter lab --port=${PORT}
