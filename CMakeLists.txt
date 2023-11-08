cmake_minimum_required(VERSION 3.15)

set(CMAKE_CXX_STANDARD 11)
set(PROJECT_VERSION 1.0)

if(CMAKE_BUILD_TYPE STREQUAL "Coverage")
    SET(CMAKE_CXX_FLAGS "-g -O0 -fprofile-arcs -ftest-coverage")
    SET(CMAKE_C_FLAGS "-g -O0 -fprofile-arcs -ftest-coverage")
endif()

SET(GCC_COVERAGE_COMPILE_FLAGS "-fprofile-arcs -ftest-coverage")
SET(GCC_COVERAGE_LINK_FLAGS "-lgcov -coverage")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -larmadillo ${GCC_COVERAGE_COMPILE_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${GCC_COVERAGE_LINK_FLAGS} -std=c++11 -larmadillo -lstdc++ -fopenmp")

include(ExternalProject)

project(Kernelo VERSION ${PROJECT_VERSION} DESCRIPTION "This is Planeto kernel" LANGUAGES CXX)
set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -larmadillo -DARMA_NO_DEBUG -DARMA_DONT_USE_OPENMP -D_GLIBCXX_USE_CXX11_ABI=0")
set(CMAKE_CXX_FLAGS_DEBUG "-std=c++11 -larmadillo -O0 -g -ggdb")
#set(CMAKE_CXX_COMPILER "/usr/bin/g++")

# get googletest -------------------------------------
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/release-1.11.0.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
# ----------------------------------------------------

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories("/usr/lib/python3/dist-packages/numpy/core/include/")
#include_directories("extern/lib")

add_library(prediction STATIC
        src/prediction/IPredictor.h
        src/prediction/Predictor.cpp
        src/prediction/Predictor.h
        src/prediction/MultivariateGaussian.h
        src/importanceSampling/ImportanceSampler.cpp
        src/importanceSampling/ImportanceSampler.h
        src/importanceSampling/Imis.cpp
        src/importanceSampling/Imis.h
        src/importanceSampling/proposition/ISProposition.h
        src/importanceSampling/proposition/GaussianMixtureProposition.cpp
        src/importanceSampling/proposition/GaussianMixtureProposition.h
        src/importanceSampling/proposition/GaussianRegularizedProposition.cpp
        src/importanceSampling/proposition/GaussianRegularizedProposition.h
        src/importanceSampling/proposition/UniformProposition.cpp
        src/importanceSampling/proposition/UniformProposition.h
        src/importanceSampling/ISResult.h src/importanceSampling/ISDiagnostic.h
        src/importanceSampling/target/ISTarget.h
        src/importanceSampling/target/ISTargetDependent.cpp
        src/importanceSampling/target/ISTargetDependent.h
        src/prediction/PredictionResult.h
        src/prediction/creators.h
        src/importanceSampling/creators.h
        src/prediction/PredictionResultExport.h
        src/importanceSampling/ImportanceSamplingDiagnostic.h
        src/importanceSampling/ImportanceSamplingResult.h)

add_library(dataGeneration STATIC
        src/dataGeneration/StatModel.h
        src/dataGeneration/GaussianStatModel.cpp
        src/dataGeneration/GaussianStatModel.h
        src/dataGeneration/GeneratorFactory.h
        src/dataGeneration/GeneratorFactory.cpp
        src/dataGeneration/GeneratorStrategy.h
        src/dataGeneration/LatinCubeGenerator.cpp
        src/dataGeneration/LatinCubeGenerator.h
        src/dataGeneration/RandomGenerator.cpp
        src/dataGeneration/RandomGenerator.h
        src/dataGeneration/SobolGenerator.cpp
        src/dataGeneration/SobolGenerator.h
        src/dataGeneration/DependentGaussianStatModel.cpp
        src/dataGeneration/DependentGaussianStatModel.h
        src/dataGeneration/creators.h
        src/prediction/IPredictor.h
        src/logging/Logger.h
        src/logging/Logger.cpp)

add_library(functionalModel STATIC
        src/functionalModel/HapkeModel/HapkeVersions/HapkeModel.cpp
        src/functionalModel/HapkeModel/HapkeVersions/Hapke93Model.cpp
        src/functionalModel/HapkeModel/HapkeVersions/Hapke02Model.cpp
        src/functionalModel/Enumeration.h
        src/functionalModel/HapkeModel/HapkeModel.h
        src/functionalModel/FunctionalModel.h
        src/functionalModel/HapkeModel/HapkeVersions/Hapke93Model.h
        src/functionalModel/HapkeModel/HapkeVersions/Hapke02Model.h
        src/functionalModel/HapkeModel/HapkeAdapters/ThreeParamsModel.cpp
        src/functionalModel/HapkeModel/HapkeAdapters/ThreeParamsModel.h
        src/functionalModel/HapkeModel/HapkeAdapter.h
        src/functionalModel/HapkeModel/HapkeAdapters/SixParamsModel.cpp
        src/functionalModel/HapkeModel/HapkeAdapters/SixParamsModel.h
        src/functionalModel/HapkeModel/HapkeAdapters/FourParamsModel.cpp
        src/functionalModel/HapkeModel/HapkeAdapters/FourParamsModel.h
        src/functionalModel/ShkuratovModel/ShkuratovModel.cpp
        src/functionalModel/ShkuratovModel/ShkuratovModel.h
        src/functionalModel/TestModel/TestModel.cpp
        src/functionalModel/TestModel/TestModel.h)

add_library(learningModel STATIC
        src/learningModel/configs/LearningConfig.h
        src/learningModel/estimators/Estimators.h
        src/learningModel/gllim/GLLiMParameters.h
        src/learningModel/covariances/Icovariance.h
        src/learningModel/EstimatorFactory.tpp
        src/learningModel/EstimatorFactory.h
        src/learningModel/gllim/IGLLiMLearning.h
        src/learningModel/gllim/GLLiMLearning.tpp
        src/learningModel/gllim/GLLiMLearning.h
        src/learningModel/initializers/Initializers.h
        src/learningModel/covariances/DiagCovariance.cpp
        src/learningModel/covariances/Fullcovariance.cpp
        src/learningModel/estimators/GmmEstimator.h
        src/learningModel/estimators/EmEstimator.h
        src/learningModel/covariances/IsoCovariance.cpp
        src/learningModel/initializers/FixedInitializer.h
        src/learningModel/initializers/FixedInitializer.tpp
        src/learningModel/initializers/MultInitializer.h
        src/learningModel/initializers/MultInitializer.tpp
        src/learningModel/configs/InitConfig.h
        src/learningModel/gllim/GLLiM.h src/helpersFunctions/Helpers.h
        src/helpersFunctions/Helpers.cpp
        src/learningModel/LearningModelFactory.tpp
        src/learningModel/LearningModelFactory.h
        src/learningModel/InitializerFactory.h
        src/logging/Logger.h
        src/logging/Logger.cpp)


target_compile_features(functionalModel PUBLIC cxx_std_11)

add_executable(main_test main.cpp)
target_link_libraries(main_test PUBLIC functionalModel)

#add_executable(testEmbed embedingPython/test.cpp src/functionalModel/ExternalModel/pyhelper.hpp src/functionalModel/ExternalModel/ExternalFunctionalModel.cpp src/functionalModel/ExternalModel/ExternalFunctionalModel.h)
#target_link_libraries(testEmbed ${PYTHON_LIBRARIES})

add_executable(generation_test dataGenerationTest.cpp)
target_link_libraries(generation_test PUBLIC gtest_main functionalModel dataGeneration learningModel ${ARMADILLO_LIBRARIES})

add_executable(learning_test learning_test.cpp)
target_link_libraries(learning_test PUBLIC gtest_main functionalModel learningModel dataGeneration prediction ${ARMADILLO_LIBRARIES} ${PYTHON_LIBRARIES})


add_subdirectory(cpptest/functionalModel_tests)
add_subdirectory(cpptest/learningModelTests)
add_subdirectory(cpptest/importanceSamplingTests)
# add_subdirectory(cpptest/predictionTests)  <-- doesn't compile

# declare tests to ctest (optional as binaries can be executed one by one)
enable_testing()
include(GoogleTest)
gtest_discover_tests(learning_test DISCOVERY_TIMEOUT 60)
gtest_discover_tests(generation_test DISCOVERY_TIMEOUT 60)

# add_subdirectory(valgrind)
