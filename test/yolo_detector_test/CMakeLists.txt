cmake_minimum_required(VERSION 3.15)
project(yolo_detector_test)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules")

include_directories(/home/jachin/space/apollo_hit/test/yolo_detector_test)
#add_executable(yolo_detector_test
#        yolo_detector.cpp yolo_detector.h opencv3t4.h yolo_obstacle_detector.cpp yolo_obstacle_detector.h interface/base_feature_extrator.h interface/base_obstacle_detector.h interface/base_init_options.h)
add_executable(yolo_detector_test
        proto/yolo.pb.h proto/yolo.pb.cc
        yolo_net.cpp yolo_net.h base/object_types.h region_output.h region_output.cc cyber/common/file.cc cyber/common/file.h inference/caffe/ inference/inference_factory.cc inference/inference_factory.h
        detector.cpp detector.h
        inference/caffe/caffe_net.cc inference/caffe/caffe_net.h
        common/util.cc common/util.h
        common/camera_frame.h
        base/object.cc base/object.h


        ) #注意依赖之间的先后顺序

#locate caffe
include_directories(/home/jachin/space/apollo_hit/test/yolo_detector_test/caffe-apollo/caffe-output/include
        /usr/local/include
        /usr/include
        /usr/include/eigen3
        caffe-apollo/src/caffe)
link_directories(yolo_detector_test /home/jachin/space/apollo_hit/test/yolo_detector_test/caffe-apollo/caffe-output/lib)
target_link_libraries(yolo_detector_test /home/jachin/space/apollo_hit/test/yolo_detector_test/caffe-apollo/caffe-output/lib/libcaffe.so)


find_package(Boost 1.55.0 REQUIRED COMPONENTS system filesystem)
include_directories(yolo_detector_test ${Boost_INCLUDE_DIRS})
link_directories(yolo_detector_test ${Boost_LIBRARY_DIRS})
target_link_libraries(yolo_detector_test ${Boost_LIBRARIES})

#locate proto
find_package(Protobuf REQUIRED)
include_directories(${PROTOBUF_INCLUDE_DIRS})
target_link_libraries(yolo_detector_test ${PROTOBUF_LIBRARIES})
message("find Protobuf:${PROTOBUF_INCLUDE_DIRS}")


#locate glog
find_package(GLOG REQUIRED)
include_directories(${GLOG_INCLUDE_DIRS})
target_link_libraries(yolo_detector_test /usr/local/lib/libglog.so) ##error using ${GLOG_LIBRARIES}
message("find glog:${GLOG_LIBRARIES}")

#locate gflag
find_package(GFlags REQUIRED)
include_directories(${GFLAGS_INCLUDE_DIRS})
target_link_libraries(yolo_detector_test ${GFLAGS_LIBRARIES})
message("find gflag:${GFLAGS_LIBRARIES}")
message("version:${GFLAGS_VERSION}")

#locate gtest
find_package(GTest MODULE REQUIRED)
include_directories(${GTEST_INCLUDE_DIR})
target_link_libraries(yolo_detector_test ${GTEST_LIBRARY})
message(gtest_include_dir=${GTEST_INCLUDE_DIR})
message(gtest_library=${GTEST_LIBRARY})

#locate opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(yolo_detector_test ${OpenCV_LIBS})






