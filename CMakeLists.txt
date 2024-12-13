cmake_minimum_required(VERSION 3.4...3.18)

set(XLLIM_VERSION_MAJOR 2)
set(XLLIM_VERSION_MINOR 0)
set(XLLIM_VERSION_PATCH 0)
set(XLLIM_VERSION_INFO "${XLLIM_VERSION_MAJOR}.${XLLIM_VERSION_MINOR}.${XLLIM_VERSION_PATCH}")

project(
  pybind11
  LANGUAGES CXX
  VERSION ${XLLIM_VERSION_INFO})


# ##############################################################################
#                                   SETTINGS                                   #
# ##############################################################################

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set compiler flags for different build types
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g") # -fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer -DARMA_EXTRA_DEBUG
set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -DNDEBUG -DARMA_NO_DEBUG -DARMA_DONT_USE_OPENMP") # -DNOPROGRESS


# ##############################################################################
#                                 REQUIREMENTS                                 #
# ##############################################################################
# (OpenMP, Armadillo, pybind11, carma)

# OpenMP
find_package(OpenMP REQUIRED)

# Armadillo
find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

# find_package(Python3 COMPONENTS Interpreter Development.Module NumPy REQUIRED)

# pybind11
add_subdirectory(extern/pybind11)

# carma
set(ARMADILLO_ROOT_DIR "/usr/include") # to help carma finding Armadillo
add_subdirectory(extern/carma)

file(READ "${carma_SOURCE_DIR}/CMakeLists.txt" ver)

string(REGEX MATCH "VERSION_MAJOR ([0-9]*)" _ ${ver})
set(ver_major ${CMAKE_MATCH_1})

string(REGEX MATCH "VERSION_MINOR ([0-9]*)" _ ${ver})
set(ver_minor ${CMAKE_MATCH_1})

string(REGEX MATCH "VERSION_PATCH ([0-9]*)" _ ${ver})
set(ver_patch ${CMAKE_MATCH_1})

message("-- carma version: ${ver_major}.${ver_minor}.${ver_patch}") # print carma version as it is not print by their cmake project



# ##############################################################################
#                                   TARGETS                                    #
# ##############################################################################

add_library(utils STATIC 
    src/utils/utils.cpp
)
set_target_properties(utils PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(utils PRIVATE OpenMP::OpenMP_CXX ${ARMADILLO_LIBRARIES})

add_library(logging INTERFACE)
target_include_directories(logging INTERFACE src/logging)

set_target_properties(logging PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(generator STATIC 
    src/generator/RandomGenerator.cpp
    src/generator/SobolGenerator.cpp
    src/generator/LatinCubeGenerator.cpp
)
set_target_properties(generator PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(functionalModel STATIC 
    src/functionalModel/FunctionalModel.cpp
    src/functionalModel/TestModel.cpp
    src/functionalModel/ShkuratovModel.cpp
    src/functionalModel/HapkeModel.cpp
    # src/functionalModel/ExternalPythonModel.cpp
)
set_target_properties(functionalModel PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(functionalModel PRIVATE utils OpenMP::OpenMP_CXX) # link pybind11::embed and carma::carma to ExternalModel

add_library(pythonEmbed STATIC 
    src/functionalModel/FunctionalModel.cpp
    src/functionalModel/ExternalPythonModel.cpp
)
set_target_properties(pythonEmbed PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(pythonEmbed PRIVATE utils pybind11::embed carma::carma) # link pybind11::embed and carma::carma to ExternalModel


add_library(xllimSolver STATIC 
    src/xllimSolver/gllim.cpp
    src/xllimSolver/jgmm.cpp
    src/xllimSolver/emEstimator.cpp
    src/xllimSolver/covariances/fullCovariance.cpp
    src/xllimSolver/covariances/diagCovariance.cpp
    src/xllimSolver/covariances/isoCovariance.cpp
)
set_target_properties(xllimSolver PROPERTIES POSITION_INDEPENDENT_CODE ON)

pybind11_add_module(xllim bindings/xllim_bindings.cpp)
target_link_libraries(xllim PRIVATE functionalModel pythonEmbed xllimSolver generator utils logging carma::carma OpenMP::OpenMP_CXX ${ARMADILLO_LIBRARIES})



# ##############################################################################
#                                    TESTS                                     #
# ##############################################################################

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

add_executable(utils_test 
    tests/cppTests/utils_test.cpp)
target_link_libraries(utils_test PRIVATE gtest_main utils ${ARMADILLO_LIBRARIES})

add_executable(generator_test 
    tests/cppTests/generator/Generator_test.cpp)
target_link_libraries(generator_test PRIVATE gtest_main generator)

add_executable(functionalModel_test 
    tests/cppTests/functionalModel/TestModel_test.cpp
    tests/cppTests/functionalModel/ShkuratovModel_test.cpp
    tests/cppTests/functionalModel/HapkeModel_test.cpp
    # tests/cppTests/functionalModel/ExternalPythonModel_test.cpp
    tests/cppTests/functionalModel/DataGeneration_test.cpp
    tests/cppTests/functionalModel/ImportanceSampling_test.cpp)
target_link_libraries(functionalModel_test PRIVATE gtest_main functionalModel generator logging ${ARMADILLO_LIBRARIES})

add_executable(pythonEmbed_test 
    tests/cppTests/functionalModel/ExternalPythonModel_test.cpp)
target_link_libraries(pythonEmbed_test PRIVATE gtest_main pythonEmbed pybind11::embed logging ${ARMADILLO_LIBRARIES})


add_executable(xllimSolver_test 
    tests/cppTests/xllimSolver/covariancesTest/fullCovarianceTest.cpp
    tests/cppTests/xllimSolver/covariancesTest/diagCovarianceTest.cpp
    tests/cppTests/xllimSolver/covariancesTest/isoCovarianceTest.cpp
    # tests/cppTests/xllimSolver/estimatorTest/jgmm_test.cpp
    tests/cppTests/xllimSolver/estimatorTest/emEstimator_test.cpp)
target_link_libraries(xllimSolver_test PRIVATE gtest_main xllimSolver utils logging ${ARMADILLO_LIBRARIES})

add_executable(integration_test 
    tests/cppTests/integration_test/performanceTests.cpp)
target_link_libraries(integration_test PRIVATE gtest_main utils xllimSolver functionalModel generator logging ${ARMADILLO_LIBRARIES})

include(GoogleTest)
gtest_discover_tests(functionalModel_test)
gtest_discover_tests(pythonEmbed_test)
gtest_discover_tests(generator_test)
gtest_discover_tests(xllimSolver_test)
gtest_discover_tests(utils_test)
gtest_discover_tests(integration_test)



# ##############################################################################
#                               INSTALLATION                                   #
# ##############################################################################

install(TARGETS xllim
  COMPONENT python
  LIBRARY DESTINATION "${PYTHON_LIBRARY_DIR}"
  )