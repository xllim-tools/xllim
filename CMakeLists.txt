cmake_minimum_required(VERSION 3.18) # Updated for modern CMake practices, 3.4 is very old

# Project name for the C++ build part.
# The version here is for the C++ build, Python package version is in pyproject.toml
project(xllim_cpp_core LANGUAGES CXX VERSION 2.0.0)

# ##############################################################################
#                                   SETTINGS                                   #
# ##############################################################################

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) # Good practice

# Set compiler flags for different build types
# These can also be set via CMAKE_BUILD_TYPE by scikit-build-core
set(CMAKE_CXX_FLAGS_DEBUG_INIT "-O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE_INIT "-O2 -DNDEBUG")
# Add Armadillo specific flags if needed, but prefer to do this via target_compile_definitions
# if(CMAKE_BUILD_TYPE STREQUAL "Release")
#   add_compile_definitions(ARMA_NO_DEBUG ARMA_DONT_USE_OPENMP) # NOPROGRESS can also be added
# endif()
# For finer control, scikit-build-core usually passes -DCMAKE_BUILD_TYPE=Release or Debug

# config for carma's cmake
set(USE_ARMA_VERSION "12.6.x")
# set(PYBIND11_ROOT_DIR "extern/pybind11")
set(USE_PYBIND11_VERSION "v2.11.1")

# Option to control test building (good for users installing the Python package)
option(XLLIM_BUILD_TESTS "Build C++ tests" OFF) # Default to OFF for package builds

# ##############################################################################
#                                 REQUIREMENTS                                 #
# ##############################################################################


find_package(Boost 1.65.0 REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

if(BLAS_FOUND AND LAPACK_FOUND)
    message(STATUS "BLAS libraries: ${BLAS_LIBRARIES}")
    message(STATUS "LAPACK libraries: ${LAPACK_LIBRARIES}")
else()
    message(FATAL_ERROR "BLAS or LAPACK not found by CMake. Armadillo needs them.")
endif()

# pybind11 (vendored)
# scikit-build-core can provide pybind11. If you use the vendored one,
# ensure pybind11_add_module is available.
# If you want to let scikit-build-core find pybind11, you'd use:
# find_package(pybind11 REQUIRED)
# But since you have it in extern/, let's use that.
add_subdirectory(extern/pybind11) # This makes pybind11::module etc. available

# carma (vendored)
# Note: ARMADILLO_ROOT_DIR might not be needed if Armadillo is found correctly by find_package.
# If carma's CMake needs it, it's okay.
# set(ARMADILLO_ROOT_DIR ${ARMADILLO_INCLUDE_DIRS}) # More robust if Armadillo_INCLUDE_DIRS is a list
add_subdirectory(extern/carma) # This makes carma::carma available

# Print carma version (from your original)
file(READ "${carma_SOURCE_DIR}/CMakeLists.txt" ver)
string(REGEX MATCH "VERSION_MAJOR ([0-9]*)" _ ${ver})
set(ver_major ${CMAKE_MATCH_1})
string(REGEX MATCH "VERSION_MINOR ([0-9]*)" _ ${ver})
set(ver_minor ${CMAKE_MATCH_1})
string(REGEX MATCH "VERSION_PATCH ([0-9]*)" _ ${ver})
set(ver_patch ${CMAKE_MATCH_1})
message(STATUS "-- carma version: ${ver_major}.${ver_minor}.${ver_patch}")

if(NOT Python_INCLUDE_DIRS AND NOT Python3_INCLUDE_DIRS) # CMake uses Python3_ for Python 3
    # scikit-build-core might set PYTHON_INCLUDE_DIR directly too
    if(DEFINED PYTHON_INCLUDE_DIR)
        set(EFFECTIVE_PYTHON_INCLUDE_DIRS ${PYTHON_INCLUDE_DIR})
    else()
        message(FATAL_ERROR "Python include directories not found. Needed for compiling xllim_pythonEmbed.")
    endif()
else()
    if(Python3_INCLUDE_DIRS)
      set(EFFECTIVE_PYTHON_INCLUDE_DIRS ${Python3_INCLUDE_DIRS})
    else()
      set(EFFECTIVE_PYTHON_INCLUDE_DIRS ${Python_INCLUDE_DIRS})
    endif()
endif()
message(STATUS "Effective Python include directories: ${EFFECTIVE_PYTHON_INCLUDE_DIRS}")

# Main include directory for your project headers (e.g., include/gllim.hpp)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# ##############################################################################
#                           INTERNAL C++ LIBRARIES                             #
# ##############################################################################
# These will be static libraries linked into the final Python module.

add_library(xllim_utils STATIC
    src/utils/utils.cpp
)
set_target_properties(xllim_utils PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(xllim_utils PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include # If utils.h is in include/
    # Or PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/utils if utils.h is there
)
# target_link_libraries(xllim_utils PUBLIC OpenMP::OpenMP_CXX armadillo::armadillo) # Use imported targets
target_link_libraries(xllim_utils PUBLIC armadillo::armadillo) # Use imported targets

add_library(xllim_logging INTERFACE)
target_include_directories(xllim_logging INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/logging # As per your original
)
# No PIC needed for INTERFACE library, no link libs unless they are INTERFACE usage requirements

add_library(xllim_generator STATIC
    src/generator/RandomGenerator.cpp
    src/generator/SobolGenerator.cpp
    src/generator/LatinCubeGenerator.cpp
)
set_target_properties(xllim_generator PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(xllim_generator PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include) # If headers are in include/
target_link_libraries(xllim_generator PUBLIC armadillo::armadillo Boost::boost)

add_library(xllim_functionalModel STATIC
    src/functionalModel/FunctionalModel.cpp
    src/functionalModel/TestModel.cpp
    src/functionalModel/ShkuratovModel.cpp
    src/functionalModel/HapkeModel.cpp
)
set_target_properties(xllim_functionalModel PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(xllim_functionalModel PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(xllim_functionalModel PUBLIC xllim_utils) # utils is also linked via xllim_utils
# target_link_libraries(xllim_functionalModel PUBLIC xllim_utils OpenMP::OpenMP_CXX) # utils is also linked via xllim_utils

set(CARMA_BINARY_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/extern/carma/include)
add_library(xllim_pythonEmbed STATIC
    # src/functionalModel/FunctionalModel.cpp # Already in xllim_functionalModel, avoid ODR issues
    src/functionalModel/ExternalPythonModel.cpp
)
set_target_properties(xllim_pythonEmbed PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(xllim_pythonEmbed PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${EFFECTIVE_PYTHON_INCLUDE_DIRS} 
    ${CARMA_BINARY_INCLUDE_DIR}
    )
# target_link_libraries(xllim_pythonEmbed PUBLIC xllim_utils pybind11::embed carma::carma)
target_link_libraries(xllim_pythonEmbed PUBLIC xllim_utils pybind11::headers carma::carma)

add_library(xllim_solver STATIC
    src/xllimSolver/gllim.cpp
    src/xllimSolver/jgmm.cpp
    src/xllimSolver/emEstimator.cpp
    src/xllimSolver/covariances/fullCovariance.cpp
    src/xllimSolver/covariances/diagCovariance.cpp
    src/xllimSolver/covariances/isoCovariance.cpp
)
set_target_properties(xllim_solver PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(xllim_solver PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(xllim_solver PUBLIC armadillo::armadillo)
# target_link_libraries(xllim_solver PUBLIC OpenMP::OpenMP_CXX armadillo::armadillo)

# ##############################################################################
#                               PYTHON MODULE                                  #
# ##############################################################################

# The Python module target. Conventionally named `_core`.
# This will produce xllim/_core.abi3.so or similar in the wheel.
pybind11_add_module(_core MODULE
    bindings/xllim_bindings.cpp
)

target_link_libraries(_core PRIVATE
    xllim_functionalModel
    xllim_pythonEmbed # If ExternalPythonModel is used by bindings
    xllim_solver
    xllim_generator
    xllim_utils     # Already linked by others, but explicit is fine
    xllim_logging   # Interface library, its include dirs and link libs are propagated
    carma::carma    # This should bring in Armadillo linkage and includes
    Boost::boost
    pybind11::module   # Link pybind11's module part
    ${LAPACK_LIBRARIES} # Explicitly link LAPACK
    ${BLAS_LIBRARIES}   # Explicitly link BLAS
)

# Ensure Armadillo include dirs are available to _core if not propagated correctly
# target_include_directories(_core PRIVATE ${ARMADILLO_INCLUDE_DIRS}) # Already via armadillo::armadillo

# Release build definitions for Armadillo (if not handled by CMAKE_BUILD_TYPE effects)
if(CMAKE_BUILD_TYPE STREQUAL "Release" OR NOT CMAKE_BUILD_TYPE)
    target_compile_definitions(_core PRIVATE ARMA_NO_DEBUG ARMA_DONT_USE_OPENMP)
    # NOPROGRESS can be added if it's a define
endif()


# ##############################################################################
#                               INSTALLATION (for Python Package)              #
# ##############################################################################
# This section is for scikit-build-core to package the Python module.
# The `DESTINATION "."` combined with `install-dir = "xllim"` in pyproject.toml
# means the .so/.pyd file will end up as `xllim/_core.xxx.so` in the package.

install(TARGETS _core
    LIBRARY DESTINATION "."  # Relative to install-dir from pyproject.toml
    RUNTIME DESTINATION "."  # For Windows DLLs if any, though usually not needed for .pyd
    COMPONENT python_module  # Optional, but good practice
)

# If you wanted to install headers for C++ developers (not for typical Python package user):
# install(
#     DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
#     DESTINATION include/xllim # Standard practice: include/<library_name>
#     COMPONENT देव
#     FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h"
# )


# ##############################################################################
#                                    TESTS (Optional)                          #
# ##############################################################################

if(XLLIM_BUILD_TESTS)
    message(STATUS "Building C++ tests as XLLIM_BUILD_TESTS is ON")
    enable_testing()

    # Google test
    include(FetchContent)
    FetchContent_Declare(
      googletest
      URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip # Or a more recent stable tag
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE) # For Windows
    FetchContent_MakeAvailable(googletest)

    # --- Test Executables ---
    # Note: Link PRIVATE to GTest (gtest_main or GTest::GTest GTest::Main)
    # and PUBLIC/PRIVATE to your libraries under test.

    add_executable(utils_test tests/cppTests/utils_test.cpp)
    target_link_libraries(utils_test PRIVATE GTest::gtest_main xllim_utils armadillo::armadillo)

    add_executable(generator_test tests/cppTests/generator/Generator_test.cpp)
    target_link_libraries(generator_test PRIVATE GTest::gtest_main xllim_generator)

    add_executable(functionalModel_test
        tests/cppTests/functionalModel/TestModel_test.cpp
        # ... other test files ...
    )
    target_link_libraries(functionalModel_test PRIVATE GTest::gtest_main xllim_functionalModel xllim_generator xllim_logging armadillo::armadillo)

    add_executable(pythonEmbed_test tests/cppTests/functionalModel/ExternalPythonModel_test.cpp)
    target_link_libraries(pythonEmbed_test PRIVATE GTest::gtest_main xllim_pythonEmbed pybind11::embed xllim_logging armadillo::armadillo) # pybind11::embed for tests using it

    add_executable(xllimSolver_test
        tests/cppTests/xllimSolver/covariancesTest/fullCovarianceTest.cpp
        # ... other test files ...
    )
    target_link_libraries(xllimSolver_test PRIVATE GTest::gtest_main xllim_solver xllim_utils xllim_logging armadillo::armadillo)

    add_executable(integration_test tests/cppTests/integration_test/performanceTests.cpp)
    target_link_libraries(integration_test PRIVATE GTest::gtest_main xllim_utils xllim_solver xllim_functionalModel xllim_generator xllim_logging armadillo::armadillo)

    include(GoogleTest) # For gtest_discover_tests
    if(TARGET functionalModel_test)
      gtest_discover_tests(functionalModel_test)
    endif()
    if(TARGET pythonEmbed_test)
      gtest_discover_tests(pythonEmbed_test)
    endif()
    # ... and for other tests
    if(TARGET utils_test)
      gtest_discover_tests(utils_test)
    endif()
    # ... etc.

else()
    message(STATUS "Skipping C++ tests build as XLLIM_BUILD_TESTS is OFF (or not set)")
endif()

message(STATUS "XLLiM C++ core configured.")
message(STATUS "  Python module target: _core")
message(STATUS "  Build tests: ${XLLIM_BUILD_TESTS}")
message(STATUS "  Armadillo includes: ${ARMADILLO_INCLUDE_DIRS}")
message(STATUS "  Armadillo libraries: ${ARMADILLO_LIBRARIES}")
message(STATUS "  Pybind11 include dir (from vendored): ${pybind11_INCLUDE_DIR}") # Check actual var from add_sub(pybind11)
message(STATUS "  Carma include dir (from vendored): ${carma_INCLUDE_DIR}")     # Check actual var

install(TARGETS _core
    LIBRARY DESTINATION xllim  # <--- Install into a subdirectory named "xllim"
    RUNTIME DESTINATION xllim  # <--- (for DLLs on Windows if _core has them)
    COMPONENT python_module
)