//
// Created by jachin on 2020/3/8.
//

#ifndef DETECTOR_TEST_DETECTOR_H
#define DETECTOR_TEST_DETECTOR_H

#include <boost/exception/detail/shared_ptr.hpp>
#include "base/object_types.h"
#include "yolo_net.h"

namespace apollo {
    namespace perception {
        namespace camera {

            std::shared_ptr<YoloObstacleDetector> detector_ = nullptr;

        }//camera
    }//perception
}//apollo

#endif //YOLO_DETECTOR_TEST_DETECTOR_H
