cmake_minimum_required(VERSION 3.15)
project(object_postprocessor)

set(CMAKE_CXX_STANDARD 14)

include_directories(location_refiner)
include_directories( "/usr/include/eigen3" )
add_executable(object_postprocessor
        #location_refiner/location_refiner_obstacle_postprocessor.cc
        #location_refiner/location_refiner_obstacle_postprocessor.h
        i_blas.h
        object_postprocessor.h object_postprocessor.cpp)
        #location_refiner/obj_postprocessor.cc
        #location_refiner/obj_postprocessor.h object_postprocessor.cpp object_postprocessor.h)
