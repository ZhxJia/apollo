//
// Created by jachin on 2020/3/8.
//
#define CPU_ONLY

#include "detector.h"
#include "yolo_net.h"
#include <memory>
#include <fstream>
#include <gtest/gtest.h>
#include "caffe/caffe.hpp"
#include "cyber/common/log.h"

using namespace caffe;
using namespace std;

//using namespace apollo::perception::base;
namespace apollo {
    namespace perception {
        namespace camera {

            static const std::map<base::ObjectSubType, cv::Scalar> kSubType2ColarMap = {
                    {base::ObjectSubType::UNKNOWN,           cv::Scalar(255, 255, 255)},
                    {base::ObjectSubType::UNKNOWN_MOVABLE,   cv::Scalar(128, 128, 128)},
                    {base::ObjectSubType::UNKNOWN_UNMOVABLE, cv::Scalar(255, 0, 0)},
                    {base::ObjectSubType::CAR,               cv::Scalar(255, 255, 128)},
                    {base::ObjectSubType::VAN,               cv::Scalar(255, 0, 0)},
                    {base::ObjectSubType::TRUCK,             cv::Scalar(0, 255, 0)},
                    {base::ObjectSubType::BUS,               cv::Scalar(0, 0, 255)},
                    {base::ObjectSubType::CYCLIST,           cv::Scalar(255, 255, 0)},
                    {base::ObjectSubType::MOTORCYCLIST,      cv::Scalar(0, 255, 255)},
                    {base::ObjectSubType::TRICYCLIST,        cv::Scalar(0, 0, 128)},
                    {base::ObjectSubType::PEDESTRIAN,        cv::Scalar(0, 128, 0)},
                    {base::ObjectSubType::TRAFFICCONE,       cv::Scalar(128, 0, 0)},
                    {base::ObjectSubType::MAX_OBJECT_TYPE,   cv::Scalar(0, 128, 128)},
            };

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

                const std::string image_name = "test3";
                //const std::string image_name = "empty.png";
                const std::string image_path = "../test_img/" + image_name + ".jpg";
                std::string result_path = "../output/" + image_name + ".txt";
                std::string vis_path = "../output/" + image_name + ".jpg";

                detector_.reset(new YoloObstacleDetector());
                detector_->Init();

                auto cv_img = cv::imread(image_path, cv::IMREAD_COLOR);

                EXPECT_TRUE(detector_->Detect(cv_img));

                FILE *fp = fopen(result_path.c_str(), "w");
                if (fp == nullptr) {
                    AERROR << "Failed to open result path: " << result_path;
                }

                int obj_id = 0;
                for (auto obj :detector_->detected_objects_) {
                    auto &supp = obj->camera_supplement;
                    auto &box = supp.box;
                    auto area_id = supp.area_id;
                    fprintf(fp,
                            "%s 0 0 %6.3f %8.2f %8.2f %8.2f %8.2f %6.3f %6.3f %6.3f "
                            "%6.3f %6.3f %6.3f %6.3f %6.3f "
                            "%4d %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f\n",
                            base::kSubType2NameMap.at(obj->sub_type).c_str(), supp.alpha,
                            supp.box.xmin, supp.box.ymin, supp.box.xmax, supp.box.ymax,
                            obj->size[2], obj->size[1], obj->size[0], obj->center[0],
                            obj->center[1] + obj->size[2] * .5, obj->center[2],
                            supp.alpha + atan2(obj->center[0], obj->center[2]),
                            obj->type_probs[static_cast<int>(obj->type)], area_id,
                            supp.visible_ratios[0], supp.visible_ratios[1],
                            supp.visible_ratios[2], supp.visible_ratios[3],
                            supp.cut_off_ratios[0], supp.cut_off_ratios[1],
                            supp.cut_off_ratios[2], supp.cut_off_ratios[3]);


                    cv::rectangle(
                            cv_img,
                            cv::Point(static_cast<int>(box.xmin), static_cast<int>(box.ymin)),
                            cv::Point(static_cast<int>(box.xmax), static_cast<int>(box.ymax)),
                            kSubType2ColarMap.at(obj->sub_type), 2);
                    float xmid = (box.xmin + box.xmax) / 2;

                    std::stringstream text;
                    auto &name = base::kSubType2NameMap.at(obj->sub_type);
                    text << name[0] << name[1] << name[2] << "-" << obj_id++;
                    cv::putText(
                            cv_img, text.str(),
                            cv::Point(static_cast<int>(box.xmin), static_cast<int>(box.ymin)),
                            cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 0, 0), 2);

                }
                cv::imwrite(vis_path.c_str(), cv_img);

                cv::imshow("result : ", cv_img);
                cv::waitKey(0);

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