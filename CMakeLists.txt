cmake_minimum_required(VERSION 3.18)

project(xllim_cpp_core LANGUAGES CXX VERSION 2.0.0)

# ##############################################################################
#                                   SETTINGS                                   #
# ##############################################################################

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS_DEBUG_INIT "-O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE_INIT "-O2 -DNDEBUG")

set(USE_PYBIND11_VERSION "v2.11.1")

option(XLLIM_BUILD_TESTS "Build C++ tests" OFF) # Default to OFF for package builds

# ##############################################################################
#                                 REQUIREMENTS                                 #
# ##############################################################################


find_package(Boost 1.65.0 REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(Armadillo REQUIRED)

if(BLAS_FOUND AND LAPACK_FOUND)
    message(STATUS "BLAS libraries: ${BLAS_LIBRARIES}")
    message(STATUS "LAPACK libraries: ${LAPACK_LIBRARIES}")
else()
    message(FATAL_ERROR "BLAS or LAPACK not found by CMake. Armadillo needs them.")
endif()

add_subdirectory(extern/pybind11)
add_subdirectory(extern/carma)

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

if(UNIX AND NOT APPLE)
    # For Linux, ensure RPATH points to the install lib dir and allows $ORIGIN
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
    # When building, RPATH should include the install RPATH so it can find libs
    # from the host environment if they are already in a layout similar to install.
    set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
    # Also add directories from the linker search path (like those from dependencies) to RPATH
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
endif()

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

add_library(xllim_generator STATIC
    src/generator/RandomGenerator.cpp
    src/generator/SobolGenerator.cpp
    src/generator/LatinCubeGenerator.cpp
)
set_target_properties(xllim_generator PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(xllim_generator PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(xllim_generator PUBLIC armadillo::armadillo Boost::boost)

add_library(xllim_functionalModel STATIC
    src/functionalModel/FunctionalModel.cpp
    src/functionalModel/TestModel.cpp
    src/functionalModel/ShkuratovModel.cpp
    src/functionalModel/HapkeModel.cpp
)
set_target_properties(xllim_functionalModel PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(xllim_functionalModel PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(xllim_functionalModel PUBLIC xllim_utils)

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
target_include_directories(xllim_solver PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CARMA_BINARY_INCLUDE_DIR})
target_link_libraries(xllim_solver PUBLIC armadillo::armadillo pybind11::headers carma::carma)

# ##############################################################################
#                               PYTHON MODULE                                  #
# ##############################################################################

pybind11_add_module(_core MODULE
    bindings/xllim_bindings.cpp
)

target_link_libraries(_core PRIVATE
    xllim_functionalModel
    xllim_pythonEmbed
    xllim_solver
    xllim_generator
    xllim_utils
    xllim_logging
    carma::carma
    Boost::boost
    pybind11::module
    ${LAPACK_LIBRARIES}
    ${BLAS_LIBRARIES}
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

install(TARGETS _core
    LIBRARY DESTINATION xllim  # <--- Install into a subdirectory named "xllim"
    RUNTIME DESTINATION xllim  # <--- (for DLLs on Windows if _core has them)
    COMPONENT python_module
)