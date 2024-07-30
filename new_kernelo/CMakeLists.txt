cmake_minimum_required(VERSION 3.4...3.18) # check version
project(newkernelo)



# GoogleTest requires at least C++14
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set compiler flags for different build types
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g") # -fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer -DARMA_EXTRA_DEBUG
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")

find_package(OpenMP REQUIRED)

find_package(Armadillo REQUIRED)
add_library(functionalModel STATIC 
    src/functionalModel/FunctionalModel.cpp
    src/functionalModel/TestModel.cpp
    src/functionalModel/ShkuratovModel.cpp
    src/functionalModel/HapkeModel.cpp
    src/functionalModel/ExternalPythonModel.cpp
    # src/utils/utils.cpp
)
set_target_properties(functionalModel PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(dataGeneration STATIC 
    src/dataGeneration/generator/GeneratorFactory.cpp
    src/dataGeneration/generator/RandomGenerator.cpp
    src/dataGeneration/generator/SobolGenerator.cpp
    src/dataGeneration/generator/LatinCubeGenerator.cpp
    src/dataGeneration/statModel/GaussianStatModel.cpp
    src/dataGeneration/statModel/DependentGaussianStatModel.cpp
)
set_target_properties(dataGeneration PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(utils STATIC 
    src/utils/utils.cpp
)
set_target_properties(utils PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(xllimSolver STATIC 
    src/xllimSolver/gllim.cpp
    src/xllimSolver/jgmm.cpp
    src/xllimSolver/emEstimator.cpp
    src/xllimSolver/covariances/fullCovariance.cpp
    src/xllimSolver/covariances/diagCovariance.cpp
    src/xllimSolver/covariances/isoCovariance.cpp
)
set_target_properties(xllimSolver PROPERTIES POSITION_INDEPENDENT_CODE ON)

# 2 methodes pour inclure CARMA:
#      soit C/C dans le code en dur carma/include et faire #include <carma.carma.h> dans le C++ pybind11
#      soit comme dans la docu officiel avec cmake (instalation dans /usr/local/include/) ... mouais
#  -> c'est plus clair si le carma est directement dans le code, discretmement dans python/carma, à côté des binding files
# Même "problème" avec pybind11 (submodule or pip/cmake installation)
# find_package(pybind11 REQUIRED) # -- Found pybind11: /home/luc/.local/include (found version "2.11.1")
add_subdirectory(extern/pybind11) # in subdirectory
pybind11_add_module(newkernelo python/kernelo.cpp)
target_link_libraries(newkernelo PRIVATE functionalModel dataGeneration utils xllimSolver)
target_link_libraries(functionalModel PRIVATE utils pybind11::embed carma::carma OpenMP::OpenMP_CXX) # link pybind11::embed and carma::carma to ExternalModel
target_link_libraries(dataGeneration PRIVATE pybind11::embed carma::carma OpenMP::OpenMP_CXX) # link pybind11::embed and carma::carma to ExternalModel

find_package(carma CONFIG REQUIRED) # -- Found carma: /usr/local/include/carma (found version "0.6.7")
target_link_libraries(newkernelo PRIVATE carma::carma OpenMP::OpenMP_CXX)


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
add_executable(functionalModel_test 
    cpptest/functionalModel/TestModel_test.cpp
    cpptest/functionalModel/ShkuratovModel_test.cpp
    cpptest/functionalModel/HapkeModel_test.cpp
    cpptest/functionalModel/ExternalPythonModel_test.cpp
    cpptest/functionalModel/ImportanceSampling_test.cpp)
target_link_libraries(functionalModel_test PRIVATE gtest_main pybind11::embed functionalModel dataGeneration) # gtest ${ARMADILLO_LIBRARIES} # link pybind11::embed to ExternalModel

add_executable(dataGeneration_test 
    cpptest/dataGeneration/Generator_test.cpp
    cpptest/dataGeneration/GaussianStatModel_test.cpp
    cpptest/dataGeneration/DependentGaussianStatModel_test.cpp)
target_link_libraries(dataGeneration_test PRIVATE gtest_main pybind11::embed dataGeneration functionalModel) # gtest ${ARMADILLO_LIBRARIES} # link pybind11::embed to ExternalModel

add_executable(xllimSolver_test 
    cpptest/xllimSolver/covariancesTest/fullCovarianceTest.cpp
    cpptest/xllimSolver/covariancesTest/diagCovarianceTest.cpp
    cpptest/xllimSolver/covariancesTest/isoCovarianceTest.cpp)
target_link_libraries(xllimSolver_test PRIVATE gtest_main xllimSolver armadillo)

add_executable(utils_test 
    cpptest/utils_test.cpp)
target_link_libraries(utils_test PRIVATE gtest_main utils armadillo)

include(GoogleTest)
gtest_discover_tests(functionalModel_test)
gtest_discover_tests(dataGeneration_test)
gtest_discover_tests(xllimSolver_test)
gtest_discover_tests(utils_test)



#################################################################


install(TARGETS newkernelo
  COMPONENT python
  LIBRARY DESTINATION "${PYTHON_LIBRARY_DIR}"
  )

# Release compilation
# $ cmake -S . -B build/release/ -DPYTHON_LIBRARY_DIR="/home/luc/.local/lib/python3.10/site-packages" -DPYTHON_EXECUTABLE="/usr/bin/python3" -DCMAKE_BUILD_TYPE=Release
# (/build/release/)$ make install

# Debug release / ctest
# $ cmake -S . -B build/debug/ -DCMAKE_BUILD_TYPE=Debug
# (/build/debug/)$ cmake --build . -j
# (/build/debug)$ ctest
