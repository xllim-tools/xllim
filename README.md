# Building on Ubuntu 20.04

## Install dependecies
```
sudo apt install gcc cmake python3-dev libarmadillo-dev libboost-dev libgtest-dev
```
Apparently the cmake setup is not building the googletest code present in ``external/lib``.
The ``libgtest-dev`` of the system must be used istead.

## Build the project
```
$ mkdir build ; cd build
$ cmake ../
$ cmake --build .
```