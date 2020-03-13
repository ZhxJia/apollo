cmake_minimum_required(VERSION 3.15)
project(io)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules")

include_directories(.
        cyber/common)

#aux_source_directory(./ ${io_src})

add_executable(io
        io_util.cc
        cyber/common/file.cc
        )

#locate absl
add_subdirectory(abseil-cpp)
target_link_libraries(io absl::strings)



#loate boost
find_package(Boost 1.55.0 REQUIRED COMPONENTS system filesystem)
include_directories(yolo_detector_test ${Boost_INCLUDE_DIRS})
link_directories(yolo_detector_test ${Boost_LIBRARY_DIRS})
target_link_libraries(io ${Boost_LIBRARIES})

# locate Eigen
include_directories(/usr/include/eigen3)

#locate glog
find_package(GLOG REQUIRED)
include_directories(${GLOG_INCLUDE_DIRS})
target_link_libraries(io /usr/local/lib/libglog.so) ##error using ${GLOG_LIBRARIES}
message("find glog:${GLOG_LIBRARIES}")

#locate gflag
find_package(GFlags REQUIRED)
include_directories(${GFLAGS_INCLUDE_DIRS})
target_link_libraries(io ${GFLAGS_LIBRARIES})
message("find gflag:${GFLAGS_LIBRARIES}")
message("version:${GFLAGS_VERSION}")

#locate gtest
find_package(GTest MODULE REQUIRED)
include_directories(${GTEST_INCLUDE_DIR})
target_link_libraries(io ${GTEST_LIBRARY})
message(gtest_include_dir=${GTEST_INCLUDE_DIR})
message(gtest_library=${GTEST_LIBRARY})

#locate protobuf
#locate proto
find_package(Protobuf REQUIRED)
include_directories(${PROTOBUF_INCLUDE_DIRS})
target_link_libraries(io ${PROTOBUF_LIBRARIES})
message("find Protobuf:${PROTOBUF_INCLUDE_DIRS}")