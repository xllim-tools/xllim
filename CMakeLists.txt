cmake_minimum_required(VERSION 3.15)

set(CMAKE_CXX_STANDARD 11)
set(PROJECT_VERSION 1.0)

SET(GCC_COVERAGE_COMPILE_FLAGS "-fprofile-arcs -ftest-coverage")
SET(GCC_COVERAGE_LINK_FLAGS "-lgcov -coverage")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -larmadillo ${GCC_COVERAGE_COMPILE_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${GCC_COVERAGE_LINK_FLAGS} -std=c++11 -larmadillo -fopenmp")

include(ExternalProject)

project(Kernelo VERSION ${PROJECT_VERSION} DESCRIPTION "This is Planeto kernel" LANGUAGES CXX)
set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -larmadillo -DARMA_NO_DEBUG -DARMA_DONT_USE_OPENMP")
set(CMAKE_CXX_FLAGS_DEBUG "-std=c++11 -larmadillo")
#set(CMAKE_CXX_COMPILER "/usr/bin/g++")

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

include(CTest)

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
        src/dataGeneration/DependentGaussianStatModel.h src/dataGeneration/creators.h)

add_library(functionalModel STATIC
        src/physicalModel/HapkeModel/HapkeVersions/HapkeModel.cpp
        src/physicalModel/HapkeModel/HapkeVersions/Hapke93Model.cpp
        src/physicalModel/HapkeModel/HapkeVersions/Hapke02Model.cpp
        src/physicalModel/FunctionnalModelFactory.cpp
        src/physicalModel/Enumeration.h
        src/physicalModel/HapkeModel/HapkeModel.h
        src/physicalModel/FunctionalModel.h
        src/physicalModel/HapkeModel/HapkeVersions/Hapke93Model.h
        src/physicalModel/HapkeModel/HapkeVersions/Hapke02Model.h src/physicalModel/HapkeModel/HapkeAdapters/ThreeParamsModel.cpp src/physicalModel/HapkeModel/HapkeAdapters/ThreeParamsModel.h src/physicalModel/HapkeModel/HapkeAdapter.h src/physicalModel/HapkeModel/HapkeAdapters/HapkeAdapterFactory.cpp src/physicalModel/HapkeModel/HapkeAdapters/HapkeAdapterFactory.h src/physicalModel/HapkeModel/HapkeAdapters/SixParamsModel.cpp src/physicalModel/HapkeModel/HapkeAdapters/SixParamsModel.h src/physicalModel/HapkeModel/HapkeAdapters/FourParamsModel.cpp src/physicalModel/HapkeModel/HapkeAdapters/FourParamsModel.h src/physicalModel/ShkuratovModel/ShkuratovModel.cpp src/physicalModel/ShkuratovModel/ShkuratovModel.h)

add_library(learningModel STATIC
        src/learningModel/configs/LearningConfig.h
        src/learningModel/estimators/Estimators.h
        src/learningModel/gllim/GLLiMParameters.h
        src/learningModel/covariances/Icovariance.h
        src/learningModel/EstimatorFactory.cpp
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
        src/learningModel/gllim/GLLiM.h src/helpersFunctions/Helpers.h src/helpersFunctions/Helpers.cpp)

target_include_directories(functionalModel PUBLIC include)

target_compile_features(functionalModel PUBLIC cxx_std_11)

add_executable(Test main.cpp)

add_executable(TestGeneration dataGenerationTest.cpp)

add_executable(TestLearning learning_test.cpp)

target_link_libraries(TestLearning PUBLIC learningModel dataGeneration ${ARMADILLO_LIBRARIES})

target_link_libraries(TestGeneration PUBLIC functionalModel dataGeneration)


target_link_libraries(Test PUBLIC functionalModel)


add_subdirectory(cpptest/functionalModel_tests)

add_subdirectory(valgrind)

add_subdirectory(cpptest/learningModelTests)