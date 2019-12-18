cmake_minimum_required(VERSION 3.15)

include(ExternalProject)

project(untitled)

set(CMAKE_CXX_STANDARD 14)

# Génère et lie le module pylib
add_library(Rectangle SHARED src/Rectangle.h src/Rectangle.cpp)
