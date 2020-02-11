#!/bin/bash

jupyter serverextension enable --py jupyterlab --sys-prefix

mkdir -p root
cd root

ls /builds/fs_djouadi/kernelo/test_env

curl --output kernelo.cpython-37m-x86_64-linux-gnu.so --header "PRIVATE-TOKEN: pxwCWfSgX95FVAkQ4USr" "https://gitlab.com/fs_djouadi/kernelo/-/jobs/artifacts/cicd_config/raw/kernelo.cpython-37m-x86_64-linux-gnu.so?job=build
"

cp ../test_hapke.json ./test_hapke.json
cp ../test_script.ipynb ./test_script.ipynb


export JUPYTER_CONFIG_DIR=/app
jupyter lab --port=${PORT}
