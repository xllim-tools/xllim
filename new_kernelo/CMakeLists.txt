cmake_minimum_required(VERSION 3.4...3.18)
project(newkernelo)

find_package(Armadillo REQUIRED)
add_library(hapke STATIC src/TestModel.cpp)
set_target_properties(hapke PROPERTIES POSITION_INDEPENDENT_CODE ON)

# find_package(pybind11 REQUIRED) # -- Found pybind11: /home/luc/.local/include (found version "2.11.1")
add_subdirectory(extern/pybind11) # in subdirectory
pybind11_add_module(newkernelo python/TestModel.cpp)
target_link_libraries(newkernelo PRIVATE hapke)

find_package(carma CONFIG REQUIRED) # -- Found carma: /usr/local/include/carma (found version "0.6.7")
target_link_libraries(newkernelo PRIVATE carma::carma)