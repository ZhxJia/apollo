//
// Created by jachin on 2020/3/9.
//

#ifndef YOLO_DETECTOR_TEST_YOLO_NET_H
#define YOLO_DETECTOR_TEST_YOLO_NET_H
#define CPU_ONLY

#include "caffe/caffe.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv3t4.h"
#include "iostream"
#include "iosfwd"
#include <vector>
#include <string>
#include "base/object_types.h"
#include "proto/yolo.pb.h"
#include "inference/inference_factory.h"
#include "region_output.h"

using namespace caffe;
//using namespace apollo::perception::camera;
//using namespace apollo::perception::base;
//using namespace apollo::perception;
//using namespace apollo::perception::inference;
namespace apollo {
    namespace perception {

        namespace camera {

            class YoloObstacleDetector {
            public:
                YoloObstacleDetector(); //proto_file和weight_file应该为全局路径

                ~YoloObstacleDetector() {};

                bool Init();

                bool Load();

            protected:

                void LoadInputShape(const yolo::ModelParam &model_param);
                void LoadParam(const yolo::YoloParam &yolo_param);

                bool InitNet(const yolo::YoloParam &yolo_param);

                void InitYoloBlob(const yolo::NetworkParam &net_param);


                std::shared_ptr<caffe::Net<float>> net_ = nullptr;

            private:
                std::string proto_file_;
                std::string weight_file_;

                yolo::YoloParam yolo_param_;
                std::shared_ptr<inference::Inference> inference_;

                std::vector<float> anchors_;
                std::vector<base::ObjectSubType> types_;

                MinDims min_dims_;
                YoloBlobs yolo_blobs_;

                //load shape
                int height_ = 800;
                int width_ = 1440;
                int offset_y_ = 0;

                //load param
                int ori_cycle_ = 1;
                float confidence_threshold_ = 0.f;
                float light_vis_conf_threshold_ = 0.f;
                float light_swt_conf_threshold_ = 0.f;
                
            };

        }//camera
    }//perception
}//apollo

#endif //YOLO_DETECTOR_TEST_YOLO_NET_H
