cmake_minimum_required(VERSION 3.15)

set(CMAKE_CXX_STANDARD 11)
set(PROJECT_VERSION 1.0)

include(ExternalProject)

project(Kernelo VERSION ${PROJECT_VERSION} DESCRIPTION "This is Planeto kernel" LANGUAGES CXX)


add_library(functionalModel STATIC
        src/physicalModel/HapkeModel.cpp
        src/physicalModel/Hapke93Model.cpp
        src/physicalModel/Hapke02Model.cpp
        src/physicalModel/FunctionnalModelFactory.cpp
        src/physicalModel/Enumeration.h
        src/physicalModel/HapkeModel.h
        src/physicalModel/FunctionalModel.h
        src/physicalModel/Hapke93Model.h
        src/physicalModel/Hapke02Model.h src/physicalModel/ThreeParamsModel.cpp src/physicalModel/ThreeParamsModel.h src/physicalModel/HapkeAdapter.h src/physicalModel/HapkeAdapterFactory.cpp src/physicalModel/HapkeAdapterFactory.h src/physicalModel/SixParamsModel.cpp src/physicalModel/SixParamsModel.h src/physicalModel/FourParamsModel.cpp src/physicalModel/FourParamsModel.h)

target_include_directories(functionalModel PUBLIC include)

target_compile_features(functionalModel PUBLIC cxx_std_11)

add_executable(Test main.cpp)

target_link_libraries(Test PUBLIC functionalModel)