cmake_minimum_required(VERSION 3.4...3.18)
project(cmake_example)

add_library(hapke STATIC src/TestModel.cpp)
set_target_properties(hapke PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_subdirectory(python/pybind11)
pybind11_add_module(cmake_example python/TestModel.cpp)
target_link_libraries(cmake_example PRIVATE hapke)

find_package(carma CONFIG REQUIRED)
target_link_libraries(cmake_example PRIVATE carma::carma)