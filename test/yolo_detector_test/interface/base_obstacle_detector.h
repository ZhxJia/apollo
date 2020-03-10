//
// Created by jachin on 2020/3/9.
//

#ifndef YOLO_DETECTOR_TEST_BASE_OBSTACLE_DETECTOR_H
#define YOLO_DETECTOR_TEST_BASE_OBSTACLE_DETECTOR_H
#pragma once

#include <memory>
#include <string>

#include "modules/perception/base/camera.h"
#include "modules/perception/camera/common/camera_frame.h"
#include "base_init_options.h"
#include <Eigen/Core>
#include <Eigen/Dense>



struct ObstacleDetectorInitOptions : public BaseInitOptions {
    std::shared_ptr<base::BaseCameraModel> base_camera_model = nullptr;
    Eigen::Matrix3f intrinsics;
};

class BaseObstacleDetector {
public:
    BaseObstacleDetector() = default;

    virtual ~BaseObstacleDetector() = default;

    virtual bool Init(const ObstacleDetectorInitOptions &options =
    ObstacleDetectorInitOptions()) = 0;

    // @brief: detect obstacle from image.
    // @param [in]: options
    // @param [in/out]: frame
    // obstacle type and 2D bbox should be filled, required,
    // 3D information of obstacle can be filled, optional.
    virtual bool Detect(const ObstacleDetectorOptions &options,
                        CameraFrame *frame) = 0;

    virtual std::string Name() const = 0;

    BaseObstacleDetector(const BaseObstacleDetector &) = delete;
    BaseObstacleDetector &operator=(const BaseObstacleDetector &) = delete;
};  // class BaseObstacleDetector




#endif //YOLO_DETECTOR_TEST_BASE_OBSTACLE_DETECTOR_H
