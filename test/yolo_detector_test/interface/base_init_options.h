//
// Created by jachin on 2020/3/9.
//

#ifndef YOLO_DETECTOR_TEST_BASE_INIT_OPTIONS_H
#define YOLO_DETECTOR_TEST_BASE_INIT_OPTIONS_H
#pragma once

#include <string>

namespace apollo {
    namespace perception {
        namespace camera {

            struct BaseInitOptions {
                std::string root_dir;
                std::string conf_file;
                int gpu_id = 0;
                bool use_cyber_work_root = false;
            };

        }  // namespace camera
    }  // namespace perception
}  // namespace apollo



#endif //YOLO_DETECTOR_TEST_BASE_INIT_OPTIONS_H
