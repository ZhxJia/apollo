//
// Created by jachin on 2020/3/9.
//

#include "yolo_net.h"
#include "cyber/common/file.h"


namespace apollo {
    namespace perception {
        namespace camera {

            YoloObstacleDetector::YoloObstacleDetector() {}

            void YoloObstacleDetector::LoadInputShape(
                    const yolo::ModelParam &model_param) {
                float offset_ratio = model_param.offset_ratio();
                float cropped_ratio = model_param.cropped_ratio();
                int resized_width = model_param.resized_width();
                int aligned_pixel = model_param.aligned_pixel();
                // inference input shape
                int image_height = 1080;
                int image_width = 1920;

                offset_y_ =
                        static_cast<int>(offset_ratio * static_cast<float>(image_height) + .5f);
                float roi_ratio = cropped_ratio * static_cast<float>(image_height) /
                                  static_cast<float>(image_width);
                width_ = static_cast<int>(resized_width + aligned_pixel / 2) / aligned_pixel *
                         aligned_pixel;  // TO DO : Suspicious code
                height_ = static_cast<int>(static_cast<float>(width_) * roi_ratio +
                                           static_cast<float>(aligned_pixel) / 2.0f) /
                          aligned_pixel * aligned_pixel;  // TO DO : Suspicious code

                AINFO << "image_height=" << image_height << ", "
                      << "image_width=" << image_width << ", "
                      << "roi_ratio=" << roi_ratio;
                AINFO << "offset_y=" << offset_y_ << ", height=" << height_
                      << ", width=" << width_;

            }

            void YoloObstacleDetector::LoadParam(const yolo::YoloParam &yolo_param) {
                const auto &model_param = yolo_param.model_param();


            }
            bool YoloObstacleDetector::Init() {

                const std::string proto_file = "/home/jachin/space/apollo_hit/test/yolo_detector_test/data/yolo/3d-r4-half/deploy.pt";
                const std::string weight_file = "/home/jachin/space/apollo_hit/test/yolo_detector_test/data/yolo/3d-r4-half/deploy.model";
                const std::string config_file = "/home/jachin/space/apollo_hit/test/yolo_detector_test/data/yolo/3d-r4-half-config.pt";

                CHECK((cyber::common::GetProtoFromFile(config_file, &yolo_param_)));
                const auto &model_param = yolo_param_.model_param();

                LoadInputShape(model_param);

            }


            bool YoloObstacleDetector::Load() {
                //set cpu running software
                Caffe::set_mode(Caffe::CPU);

                //load net file , caffe::TEST 用于测试时使用
                net_.reset(new caffe::Net<float>(proto_file_, caffe::TEST));
                if (net_ == nullptr)
                    return false;
                //load net train file caffemodel
                net_->CopyTrainedLayersFrom(weight_file_);


                return true;

            }


            bool YoloObstacleDetector::InitNet(const yolo::YoloParam &yolo_param) {

                std::vector<std::string> input_names;
                std::vector<std::string> output_names;
                //init net
                auto const &net_param = yolo_param.net_param();
                input_names.push_back(net_param.input_blob()); //jac!!20/1/10: blob 是caffe 作为数据传输的媒介
                output_names.push_back(net_param.det1_loc_blob());
                output_names.push_back(net_param.det1_obj_blob());
                output_names.push_back(net_param.det1_cls_blob());
                output_names.push_back(net_param.det1_ori_conf_blob());
                output_names.push_back(net_param.det1_ori_blob());
                output_names.push_back(net_param.det1_dim_blob());
                output_names.push_back(net_param.det2_loc_blob());
                output_names.push_back(net_param.det2_obj_blob());
                output_names.push_back(net_param.det2_cls_blob());
                output_names.push_back(net_param.det2_ori_conf_blob());
                output_names.push_back(net_param.det2_ori_blob());
                output_names.push_back(net_param.det2_dim_blob());
                output_names.push_back(net_param.det3_loc_blob());
                output_names.push_back(net_param.det3_obj_blob());
                output_names.push_back(net_param.det3_cls_blob());
                output_names.push_back(net_param.det3_ori_conf_blob());
                output_names.push_back(net_param.det3_ori_blob());
                output_names.push_back(net_param.det3_dim_blob());
                output_names.push_back(net_param.lof_blob());
                output_names.push_back(net_param.lor_blob());
                output_names.push_back(net_param.brvis_blob());
                output_names.push_back(net_param.brswt_blob());
                output_names.push_back(net_param.ltvis_blob());
                output_names.push_back(net_param.ltswt_blob());
                output_names.push_back(net_param.rtvis_blob());
                output_names.push_back(net_param.rtswt_blob());
                output_names.push_back(net_param.feat_blob());
                output_names.push_back(net_param.area_id_blob());
                output_names.push_back(net_param.visible_ratio_blob());
                output_names.push_back(net_param.cut_off_ratio_blob());

                //Infer
                inference_.reset(inference::CreateInferenceByName("CaffeNet", proto_file_,
                                                                  weight_file_, output_names,
                                                                  input_names));

                if (nullptr == inference_.get()) {
                    return false;
                }
                std::vector<int> shape = {1, height_, width_, 3};
                std::map<std::string, std::vector<int>> shape_map{
                        {net_param.input_blob(), shape}};

                //load net param
                if (!inference_->Init(shape_map)) {
                    return false;
                }
                //net forward
                inference_->Infer();
                return true;
            }

            void YoloObstacleDetector::InitYoloBlob(const yolo::NetworkParam &net_param) {

                auto obj_blob_scale1 = inference_->get_blob(net_param.det1_obj_blob());
                auto obj_blob_scale2 = inference_->get_blob(net_param.det2_obj_blob());
                auto obj_blob_scale3 = inference_->get_blob(net_param.det3_obj_blob());
                int output_height_scale1 = obj_blob_scale1->shape(1);
                int output_width_scale1 = obj_blob_scale1->shape(2);
                int obj_size = output_height_scale1 * output_width_scale1 *
                               static_cast<int>(anchors_.size()) / anchorSizeFactor; //2

                if (obj_blob_scale2) {
                    int output_height_scale2 = obj_blob_scale2->shape(1);
                    int output_width_scale2 = obj_blob_scale2->shape(2);
                    int output_height_scale3 = obj_blob_scale3->shape(1);
                    int output_width_scale3 = obj_blob_scale3->shape(2);
                    obj_size = (output_height_scale1 * output_width_scale1 +
                                output_height_scale2 * output_width_scale2 +
                                output_height_scale3 * output_width_scale3) *
                               static_cast<int>(anchors_.size()) / anchorSizeFactor / numScales;
                }
                yolo_blobs_.res_box_blob.reset(
                        new caffe::Blob<float>(1, 1, obj_size, kBoxBlockSize)); //32
                yolo_blobs_.res_cls_blob.reset(new caffe::Blob<float>(
                        1, 1, static_cast<int>(types_.size() + 1), obj_size));
                yolo_blobs_.res_cls_blob->cpu_data();


            }
        }//camera
    }//perception
}//apollo
