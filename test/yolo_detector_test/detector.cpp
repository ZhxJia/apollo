//
// Created by jachin on 2020/3/8.
//
#define CPU_ONLY

#include "detector.h"
#include "yolo_net.h"
#include <memory>
#include <fstream>
#include "caffe/caffe.hpp"

using namespace caffe;
using namespace std;

//using namespace apollo::perception::base;
namespace apollo {
    namespace perception {
        namespace camera {

            static const cv::Scalar kBoxColorMap[] = {
                    cv::Scalar(0, 0, 0),        // 0
                    cv::Scalar(128, 128, 128),  // 1
                    cv::Scalar(255, 0, 0),      // 2
                    cv::Scalar(0, 255, 0),      // 3
                    cv::Scalar(0, 0, 255),      // 4
                    cv::Scalar(255, 255, 0),    // 5
                    cv::Scalar(0, 255, 255),    // 6
                    cv::Scalar(255, 0, 255),    // 7
                    cv::Scalar(255, 255, 255),  // 8
            };

            static const cv::Scalar kFaceColorMap[] = {
                    cv::Scalar(255, 255, 255),  // 0
                    cv::Scalar(255, 0, 0),      // 1
                    cv::Scalar(0, 255, 0),      // 2
                    cv::Scalar(0, 0, 255),      // 3
            };

            base::ObjectSubType GetObjectSubType(const std::string &type_name) {
                if (type_name == "car") {
                    return base::ObjectSubType::CAR;
                } else if (type_name == "van") {
                    return base::ObjectSubType::VAN;
                } else if (type_name == "bus") {
                    return base::ObjectSubType::BUS;
                } else if (type_name == "truck") {
                    return base::ObjectSubType::TRUCK;
                } else if (type_name == "cyclist") {
                    return base::ObjectSubType::CYCLIST;
                } else if (type_name == "motorcyclist") {
                    return base::ObjectSubType::MOTORCYCLIST;
                } else if (type_name == "tricyclelist") {
                    return base::ObjectSubType::TRICYCLIST;
                } else if (type_name == "pedestrian") {
                    return base::ObjectSubType::PEDESTRIAN;
                } else if (type_name == "trafficcone") {
                    return base::ObjectSubType::TRAFFICCONE;
                } else {
                    // type_name is "" or unknown
                    return base::ObjectSubType::UNKNOWN;
                }
            }



            int main() {



                const std::string image_path = "../test.jpg";
                detector_.reset(new YoloObstacleDetector());
                detector_->Init();


                auto cv_img = cv::imread(image_path,cv::IMREAD_COLOR);

                detector_->Detect(cv_img);

                cout << "done!" << endl;
                return 0;
            }
        }//camera
    }//perception
}//apollo

int main() {

    cout << "detector test" << endl;
    return apollo::perception::camera::main();

}