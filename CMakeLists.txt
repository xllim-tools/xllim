cmake_minimum_required(VERSION 3.15)

include(ExternalProject)

project(untitled)

set(CMAKE_CXX_STANDARD 14)

#INCLUDE_DIRECTORIES(/arma/build/mymachine/include)

# Génère et lie le module pylib
#add_library(Node SHARED
#        src/physicalModel/HapkeModel.cpp
#        src/physicalModel/HapkeModel.h
#        src/physicalModel/Enumeration.h main.cpp
#        src/physicalModel/FunctionnalModel.h
#        src/physicalModel/Hapke93Model.cpp
#        src/physicalModel/Hapke93Model.h
#        src/physicalModel/Hapke02Model.cpp
#        src/physicalModel/Hapke02Model.h)

add_executable(Test main.cpp src/physicalModel/Hapke02Model.cpp src/physicalModel/HapkeModel.cpp)