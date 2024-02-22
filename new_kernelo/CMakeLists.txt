cmake_minimum_required(VERSION 3.4...3.18)
project(newkernelo)


# GoogleTest requires at least C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_package(Armadillo REQUIRED)
add_library(functionalModel STATIC 
    src/TestModel.cpp
    src/ShkuratovModel.cpp
    src/HapkeModel.cpp
    src/ExternalPythonModel.cpp
)
set_target_properties(functionalModel PROPERTIES POSITION_INDEPENDENT_CODE ON)

# 2 methodes pour inclure CARMA:
#      soit C/C dans le code en dur carma/include et faire #include <carma.carma.h> dans le C++ pybind11
#      soit comme dans la docu officiel avec cmake (instalation dans /usr/local/include/) ... mouais
#  -> c'est plus clair si le carma est directement dans le code, discretmement dans python/carma, à côté des binding files
# Même "problème" avec pybind11 (submodule or pip/cmake installation)
# find_package(pybind11 REQUIRED) # -- Found pybind11: /home/luc/.local/include (found version "2.11.1")
add_subdirectory(extern/pybind11) # in subdirectory
pybind11_add_module(newkernelo python/functionalModel.cpp)
target_link_libraries(newkernelo PRIVATE functionalModel)
target_link_libraries(functionalModel PRIVATE pybind11::embed carma::carma) # link pybind11::embed and carma::carma to ExternalModel

find_package(carma CONFIG REQUIRED) # -- Found carma: /usr/local/include/carma (found version "0.6.7")
target_link_libraries(newkernelo PRIVATE carma::carma)


#################################################################
#                           Google Test                         #
#################################################################

### get Google test
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
###


enable_testing()
add_executable(TestModel_test cpptest/TestModel_test.cpp)
target_link_libraries(TestModel_test gtest_main functionalModel) # gtest ${ARMADILLO_LIBRARIES}

include(GoogleTest)
gtest_discover_tests(TestModel_test)



#################################################################