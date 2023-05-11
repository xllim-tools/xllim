#!/bin/bash
rm -rf dist/ build/ kernelo.egg-info/
rm -rf cython/kernelo.cpp
python3 setup.py sdist bdist_wheel -vvv
pip install dist/kernelo-0.1-cp310-cp310-linux_x86_64.whl --force-reinstall
