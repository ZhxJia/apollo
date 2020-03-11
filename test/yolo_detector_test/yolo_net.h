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

                bool Detect(const cv::Mat &img);

                void WrapInputLayer(std::vector<cv::Mat> *input_channels);

                void Preprocess(const cv::Mat &img, std::vector<cv::Mat> *input_channels);

                void get_objects_cpu(const YoloBlobs &yolo_blobs, const std::vector<base::ObjectSubType> &types,
                                     const NMSParam &nms, const yolo::ModelParam &model_param,
                                     float light_vis_conf_threshold, float light_swt_conf_threshold,
                                     caffe::Blob<float> *overlapped, caffe::Blob<float> *idx_sm,
                                     std::vector<base::ObjectPtr> *objects);

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
                std::vector<float> expands_;

                //load shape
                int height_ = 576;
                int width_ = 1440;
                int offset_y_ = 0;

                //load param
                int ori_cycle_ = 1;
                float confidence_threshold_ = 0.f;
                float light_vis_conf_threshold_ = 0.f;
                float light_swt_conf_threshold_ = 0.f;
                MinDims min_dims_;
                //nms param
                NMSParam nms_;
                int obj_k_ = kMaxObjSize;


                float border_ratio_ = 0.f;

                //InitYoloBlob
                YoloBlobs yolo_blobs_;
                std::shared_ptr<caffe::Blob<float>> overlapped_ = nullptr;
                std::shared_ptr<caffe::Blob<float>> idx_sm_ = nullptr;


                //input shape
                cv::Size input_geometry_ = cv::Size(1440, 576);
                //detected_objects
                std::vector<base::ObjectPtr> detected_objects_;

            };

        }//camera
    }//perception
}//apollo

#endif //YOLO_DETECTOR_TEST_YOLO_NET_H
