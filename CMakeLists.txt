cmake_minimum_required(VERSION 3.21)

project(xllim_cpp_core LANGUAGES CXX VERSION 2.0.0)

# ##############################################################################
#                                   SETTINGS                                   #
# ##############################################################################

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS_DEBUG_INIT "-O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE_INIT "-O2 -DNDEBUG")

option(XLLIM_BUILD_TESTS "Build C++ tests" OFF) # Default to OFF for package builds

set(CMAKE_FIND_PACKAGE_PREFER_CONFIG ON) # Prefer <Package>Config.cmake over Find<Package>.cmake
if(DEFINED ENV{CONDA_PREFIX})
    list(PREPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
    message(STATUS "Prepended CONDA_PREFIX to CMAKE_PREFIX_PATH: $ENV{CONDA_PREFIX}")
else()
    message(WARNING "CONDA_PREFIX environment variable not set. Dependency resolution might be less predictable.")
endif()
# ##############################################################################
#                                 REQUIREMENTS                                 #
# ##############################################################################
# Boost 1.78: matches the version built for the PyPI wheel, ensuring ABI consistency across CI and conda.
find_package(Boost 1.78.0 REQUIRED COMPONENTS system thread random)
message(STATUS "Found Boost: Version ${Boost_VERSION}, Libs: ${Boost_LIBRARIES}, Includes: ${Boost_INCLUDE_DIRS}")

# OpenBLAS selected explicitly. Version is not strictly pinned: BLAS/LAPACK ABI is historically stable,
# so a loose bound avoids over-constraining while still selecting the correct backend.
set(BLA_VENDOR "OpenBLAS" CACHE STRING "Preferred BLAS vendor")
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
message(STATUS "BLAS libraries: ${BLAS_LIBRARIES}")
message(STATUS "LAPACK libraries: ${LAPACK_LIBRARIES}")

# --- Armadillo (from Conda) ---
add_definitions(-DARMA_DONT_USE_WRAPPER)
if(DEFINED ENV{CONDA_PREFIX})
    set(Armadillo_DIR "$ENV{CONDA_PREFIX}/share/Armadillo/CMake" CACHE PATH "Directory containing ArmadilloConfig.cmake")
    message(STATUS "Set Armadillo_DIR to: ${Armadillo_DIR}")
endif()
# 12.6 is the minimum version validated against in the manylinux image in CI.
find_package(Armadillo 12.6 REQUIRED)
if(TARGET armadillo AND NOT TARGET armadillo::armadillo)
    add_library(armadillo::armadillo ALIAS armadillo)
    message(STATUS "Aliased 'armadillo' target to armadillo::armadillo.")
else()
    message(FATAL_ERROR "Armadillo not found")
endif()

# --- Pybind11 ---
# XLLIM_USE_SYSTEM_PYBIND11 allow conda-forge builds to use system-installed pybind11 instead of vendored.
# This satisfies conda-forge's non-vendorization policy while preserving
# vendored pybind11 for PyPI/local builds (default: OFF).
option(XLLIM_USE_SYSTEM_PYBIND11 "Use system-installed pybind11 instead of vendored" OFF)
if(XLLIM_USE_SYSTEM_PYBIND11)
    find_package(pybind11 REQUIRED CONFIG)
    message(STATUS "Using system pybind11: ${pybind11_VERSION}")
else()
    set(PYBIND11_USE_CMAKE_MODULE OFF) # We use add_subdirectory
    set(PYBIND11_TEST OFF) # Don't build pybind11 tests
    add_subdirectory(extern/pybind11)
endif()

# --- Carma ---
# Allow conda-forge builds to use system-installed carma instead of vendored.
# Carma will find Armadillo using find_package(Armadillo) internally.
# It should pick up the same Armadillo target we found above.
option(XLLIM_USE_SYSTEM_CARMA "Use system-installed carma instead of vendored" OFF)
if(XLLIM_USE_SYSTEM_CARMA)
    find_package(carma REQUIRED CONFIG)
    message(STATUS "Using system carma: ${carma_VERSION}")
else()
    set(CARMA_BUILD_TESTS OFF)
    set(CARMA_BUILD_EXAMPLES OFF)
    add_subdirectory(extern/carma)
endif()

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
# include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/include) # Project headers

if(APPLE)
    set(CMAKE_MACOSX_RPATH TRUE)
    # @loader_path allows finding libraries relative to the current binary/library loading them.
    # $ENV{CONDA_PREFIX}/lib allows finding other conda libraries.
    set(_INSTALL_RPATH "@loader_path")
    if(DEFINED ENV{CONDA_PREFIX})
        list(APPEND _INSTALL_RPATH "$ENV{CONDA_PREFIX}/lib")
    endif()
    set(CMAKE_INSTALL_RPATH "${_INSTALL_RPATH}")
    message(STATUS "macOS CMAKE_INSTALL_RPATH: ${CMAKE_INSTALL_RPATH}")
elseif(UNIX) # Linux
    # Commented code ensures only conda's libs are used, but it make xllim segfault
    # set(_INSTALL_RPATH "\$ORIGIN")
    # if(DEFINED ENV{CONDA_PREFIX})
    #     list(APPEND _INSTALL_RPATH "$ENV{CONDA_PREFIX}/lib")
    # endif()
    # list(APPEND _INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
    # string(REPLACE ";" ":" _INSTALL_RPATH_STR "${_INSTALL_RPATH}")
    # set(CMAKE_INSTALL_RPATH "${_INSTALL_RPATH_STR}")
    # message(STATUS "Linux CMAKE_INSTALL_RPATH: ${CMAKE_INSTALL_RPATH}")
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
endif()

if(UNIX)
    set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
endif()

# ##############################################################################
#                           INTERNAL C++ LIBRARIES                             #
# ##############################################################################
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

add_library(xllim_utils STATIC src/utils/utils.cpp)
target_include_directories(xllim_utils PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(xllim_utils PUBLIC armadillo::armadillo) # Use imported targets

add_library(xllim_logging INTERFACE)
target_include_directories(xllim_logging INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/logging>
    $<INSTALL_INTERFACE:include/logging>
)

add_library(xllim_generator STATIC
    src/generator/RandomGenerator.cpp
    src/generator/SobolGenerator.cpp
    src/generator/LatinCubeGenerator.cpp
)
set_target_properties(xllim_generator PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(xllim_generator PUBLIC armadillo::armadillo Boost::system Boost::thread Boost::random)

add_library(xllim_functionalModel STATIC
    src/functionalModel/FunctionalModel.cpp
    src/functionalModel/TestModel.cpp
    src/functionalModel/ShkuratovModel.cpp
    src/functionalModel/HapkeModel.cpp
)
set_target_properties(xllim_functionalModel PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(xllim_functionalModel PUBLIC xllim_utils)

set(CARMA_BINARY_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/extern/carma/include)
add_library(xllim_pythonEmbed STATIC
    # src/functionalModel/FunctionalModel.cpp # Already in xllim_functionalModel, avoid ODR issues
    src/functionalModel/ExternalPythonModel.cpp
)
target_include_directories(xllim_pythonEmbed PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${EFFECTIVE_PYTHON_INCLUDE_DIRS}
    ${carma_INCLUDE_DIR} # Use variable set by carma
)
set_target_properties(xllim_pythonEmbed PROPERTIES POSITION_INDEPENDENT_CODE ON)
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
target_include_directories(xllim_solver PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${carma_INCLUDE_DIR}
)
target_link_libraries(xllim_solver PUBLIC armadillo::armadillo pybind11::headers carma::carma)

# ##############################################################################
#                               PYTHON MODULE                                  #
# ##############################################################################

pybind11_add_module(_core MODULE bindings/xllim_bindings.cpp)
target_link_libraries(_core PRIVATE
    xllim_functionalModel
    xllim_pythonEmbed
    xllim_solver
    xllim_generator
    xllim_logging
    pybind11::module
    )

if(CMAKE_BUILD_TYPE STREQUAL "Release" OR NOT CMAKE_BUILD_TYPE)
    target_compile_definitions(_core PRIVATE ARMA_NO_DEBUG ARMA_DONT_USE_OPENMP)
    # NOPROGRESS can be added if it's a define
endif()


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

# ##############################################################################
#                                  INSTALL                                     #
# ##############################################################################

install(TARGETS _core
    LIBRARY DESTINATION xllim  # <--- Install into a subdirectory named "xllim"
    RUNTIME DESTINATION xllim  # <--- (for DLLs on Windows if _core has them)
    COMPONENT python_module
)

message(STATUS "XLLiM C++ core configured.")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  CONDA_PREFIX: $ENV{CONDA_PREFIX}")
message(STATUS "  CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")
message(STATUS "  Boost includes: ${Boost_INCLUDE_DIRS}, Boost libs: ${Boost_LIBRARIES}")
message(STATUS "  BLAS libraries: ${BLAS_LIBRARIES}")
message(STATUS "  LAPACK libraries: ${LAPACK_LIBRARIES}")
if(TARGET armadillo::armadillo)
    get_target_property(ARMA_INC armadillo::armadillo INTERFACE_INCLUDE_DIRECTORIES)
    get_target_property(ARMA_LIB armadillo::armadillo INTERFACE_LINK_LIBRARIES)
    message(STATUS "  armadillo::armadillo: Includes: ${ARMA_INC}, Link Libs: ${ARMA_LIB}")
else()
    message(WARNING "  armadillo::armadillo target not found after configuration.")
endif()
message(STATUS "  Pybind11 include dir: ${pybind11_INCLUDE_DIR}")
message(STATUS "  Carma include dir: ${carma_INCLUDE_DIR}")