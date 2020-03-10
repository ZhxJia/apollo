//
// Created by jachin on 2020/3/9.
//

#include "yolo_net.h"
#include "cyber/common/file.h"
#include "common/util.h"

namespace apollo {
    namespace perception {
        namespace camera {

            YoloObstacleDetector::YoloObstacleDetector() {}

            void YoloObstacleDetector::get_objects_cpu(const YoloBlobs &yolo_blobs,
                                                       const std::vector<base::ObjectSubType> &types,
                                                       const NMSParam &nms, const yolo::ModelParam &model_param,
                                                       float light_vis_conf_threshold, float light_swt_conf_threshold,
                                                       caffe::Blob<float> *overlapped, caffe::Blob<float> *idx_sm,
                                                       std::vector<base::ObjectPtr> *objects) {
                bool multi_scale = false;
                if (yolo_blobs.det2_obj_blob) {
                    multi_scale = true;
                }
                int num_classes = types.size();
                int batch = yolo_blobs.det1_obj_blob->shape(0);
                int num_anchor = yolo_blobs.anchor_blob->shape(2);
                int num_anchor_per_scale = num_anchor;
                if (multi_scale) {
                    num_anchor_per_scale /= numScales;
                }
                CHECK_EQ(batch, 1) << "batch size should be 1!";
                std::vector<int> height_vec, width_vec, num_candidates_vec;
                height_vec.push_back(yolo_blobs.det1_obj_blob->shape(1));
                width_vec.push_back(yolo_blobs.det1_obj_blob->shape(2));
                if (multi_scale) {
                    height_vec.push_back(yolo_blobs.det2_obj_blob->shape(1));
                    height_vec.push_back(yolo_blobs.det3_obj_blob->shape(1));
                    width_vec.push_back(yolo_blobs.det2_obj_blob->shape(2));
                    width_vec.push_back(yolo_blobs.det3_obj_blob->shape(2));
                }
                for (size_t i = 0; i < height_vec.size(); i++) {
                    num_candidates_vec.push_back(
                            height_vec[i] * width_vec[i] * num_anchor_per_scale);
                }
                const float *loc_data_vec[3] = {yolo_blobs.det1_loc_blob->cpu_data(),
                                                yolo_blobs.det2_loc_blob ? yolo_blobs.det2_loc_blob->cpu_data()
                                                                         : nullptr,
                                                yolo_blobs.det3_loc_blob ? yolo_blobs.det3_loc_blob->cpu_data()
                                                                         : nullptr};
                const float *obj_data_vec[3] = {yolo_blobs.det1_obj_blob->cpu_data(),
                                                yolo_blobs.det2_obj_blob ? yolo_blobs.det2_obj_blob->cpu_data()
                                                                         : nullptr,
                                                yolo_blobs.det3_obj_blob ? yolo_blobs.det3_obj_blob->cpu_data()
                                                                         : nullptr};
                const float *cls_data_vec[3] = {yolo_blobs.det1_cls_blob->cpu_data(),
                                                yolo_blobs.det2_cls_blob ? yolo_blobs.det2_cls_blob->cpu_data()
                                                                         : nullptr,
                                                yolo_blobs.det3_cls_blob ? yolo_blobs.det3_cls_blob->cpu_data()
                                                                         : nullptr};
                const float *ori_data_vec[3] = {get_cpu_data(model_param.with_box3d(),
                                                             *yolo_blobs.det1_ori_blob),
                                                multi_scale ? get_cpu_data(model_param.with_box3d(),
                                                                           *yolo_blobs.det2_ori_blob) : nullptr,
                                                multi_scale ? get_cpu_data(model_param.with_box3d(),
                                                                           *yolo_blobs.det3_ori_blob) : nullptr};
                const float *dim_data_vec[3] = {get_cpu_data(model_param.with_box3d(),
                                                             *yolo_blobs.det1_dim_blob),
                                                multi_scale ? get_cpu_data(model_param.with_box3d(),
                                                                           *yolo_blobs.det2_dim_blob) : nullptr,
                                                multi_scale ? get_cpu_data(model_param.with_box3d(),
                                                                           *yolo_blobs.det3_dim_blob) : nullptr};
                AINFO << "cls_data_vec: " << yolo_blobs.det1_cls_blob->cpu_data()
                      << " ori_data_vec: " << obj_data_vec[0];
                //TODO[KaWai]: add 3 scale frbox data and light data.
                const float *lof_data = get_cpu_data(
                        model_param.with_frbox(), *yolo_blobs.lof_blob);
                const float *lor_data = get_cpu_data(
                        model_param.with_frbox(), *yolo_blobs.lor_blob);

                const float *area_id_data = get_cpu_data(
                        model_param.num_areas() > 0, *yolo_blobs.area_id_blob);
                const float *visible_ratio_data = get_cpu_data(
                        model_param.with_ratios(), *yolo_blobs.visible_ratio_blob);
                const float *cut_off_ratio_data = get_cpu_data(
                        model_param.with_ratios(), *yolo_blobs.cut_off_ratio_blob);

                const auto &with_lights = model_param.with_lights();
                const float *brvis_data = get_cpu_data(with_lights, *yolo_blobs.brvis_blob);
                const float *brswt_data = get_cpu_data(with_lights, *yolo_blobs.brswt_blob);
                const float *ltvis_data = get_cpu_data(with_lights, *yolo_blobs.ltvis_blob);
                const float *ltswt_data = get_cpu_data(with_lights, *yolo_blobs.ltswt_blob);
                const float *rtvis_data = get_cpu_data(with_lights, *yolo_blobs.rtvis_blob);
                const float *rtswt_data = get_cpu_data(with_lights, *yolo_blobs.rtswt_blob);



            }

            void YoloObstacleDetector::WrapInputLayer(std::vector<cv::Mat> *input_channels) {
                auto input_blob = inference_->get_blob(yolo_param_.net_param().input_blob());
                int width = input_blob->shape(2);
                int height = input_blob->shape(1);
                float *input_data = input_blob->mutable_cpu_data();

                for (int i = 0; i < input_blob->shape(3); ++i) {
                    cv::Mat channel(height, width, CV_32FC1, input_data);
                    input_channels->push_back(channel);
                    input_data += width * height;
                }

            }

            void YoloObstacleDetector::Preprocess(const cv::Mat &img, std::vector<cv::Mat> *input_channels) {

                cv::Mat sample_resized;

                if (img.size() != input_geometry_) {
                    cv::resize(img, sample_resized, input_geometry_);
                } else {
                    sample_resized = img;
                }
//                AINFO << "size:" << width_;
//                cv::Mat sample_sign;
//                sample_resized.convertTo(sample_sign,CV_32SC3);
//                cv::Mat mean_ = cv::Mat(input_geometry_,CV_32SC3,cv::Scalar(95,99,96));
//
//                cv::Mat sample_normalized;
//                cv::subtract(sample_sign,mean_,sample_normalized);
//
                cv::Mat sample_float;
//                sample_normalized.convertTo(sample_float,CV_32FC3);
                sample_resized.convertTo(sample_float, CV_32FC3);
                cv::split(sample_float, *input_channels);
//                cv::imshow("s",sample_float);
//                cv::waitKey(0);


                auto input_blob = inference_->get_blob(yolo_param_.net_param().input_blob());
//                AINFO << "input channels：" << reinterpret_cast<float*>(input_channels->at(0).data) << "  blob:" <<input_blob->cpu_data();
                CHECK(reinterpret_cast<float *>(input_channels->at(0).data)
                      == input_blob->cpu_data())
                << "Input channels are not wrapping the input layer of the network.";
            }

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
                confidence_threshold_ = model_param.confidence_threshold();
                light_vis_conf_threshold_ = model_param.light_vis_conf_threshold();
                light_swt_conf_threshold_ = model_param.light_swt_conf_threshold();
                min_dims_.min_2d_height = model_param.min_2d_height();
                min_dims_.min_3d_height = model_param.min_3d_height();
                min_dims_.min_3d_width = model_param.min_3d_width();
                min_dims_.min_3d_length = model_param.min_3d_length();
                ori_cycle_ = model_param.ori_cycle();

                border_ratio_ = model_param.border_ratio();

                //init NMS
                auto const &nms_param = yolo_param.nms_param();
                nms_.sigma = nms_param.sigma();
                nms_.type = nms_param.type();
                nms_.threshold = nms_param.threshold();
                nms_.inter_cls_nms_thresh = nms_param.inter_cls_nms_thresh();
                nms_.inter_cls_conf_thresh = nms_param.inter_cls_conf_thresh();

            }

            bool YoloObstacleDetector::Init() {

                const std::string proto_file = "/home/jachin/space/apollo_hit/test/yolo_detector_test/data/yolo/3d-r4-half/deploy.pt";
                const std::string weight_file = "/home/jachin/space/apollo_hit/test/yolo_detector_test/data/yolo/3d-r4-half/deploy.model";
                const std::string config_file = "/home/jachin/space/apollo_hit/test/yolo_detector_test/data/yolo/3d-r4-half-config.pt";
                const std::string anchors_file = "/home/jachin/space/apollo_hit/test/yolo_detector_test/data/yolo/3d-r4-half/anchors.txt";
                const std::string types_file = "/home/jachin/space/apollo_hit/test/yolo_detector_test/data/yolo/3d-r4-half/types.txt";
                const std::string expand_file = "/home/jachin/space/apollo_hit/test/yolo_detector_test/data/yolo/3d-r4-half/expand.txt";

                CHECK((cyber::common::GetProtoFromFile(config_file, &yolo_param_)));
                const auto &model_param = yolo_param_.model_param();
                proto_file_ = proto_file;
                weight_file_ = weight_file;

                LoadInputShape(model_param);
                LoadParam(yolo_param_);
                min_dims_.min_2d_height /= static_cast<float>(height_);

                if (!LoadAnchors(anchors_file, &anchors_)) {
                    return false;
                }
                if (!LoadTypes(types_file, &types_)) {
                    return false;
                }
                if (!LoadExpand(expand_file, &expands_)) {
                    return false;
                }
                CHECK(expands_.size() == types_.size());
                if (!InitNet(yolo_param_)) {
                    return false;
                }
                InitYoloBlob(yolo_param_.net_param());


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

            bool YoloObstacleDetector::Detect(const cv::Mat &img) {
                if (!img.data) {
                    AERROR << "imgdata is empty";
                }

                std::vector<cv::Mat> input_channels;

                auto input_blob = inference_->get_blob(yolo_param_.net_param().input_blob());

                AINFO << "input_blob_shape=" << "[" << input_blob->shape(0) << "," << input_blob->shape(1)
                      << "," << input_blob->shape(2) << "," << input_blob->shape(3) << "]";

                WrapInputLayer(&input_channels);

                Preprocess(img, &input_channels);

                ///detection part
                inference_->Infer();

                get_objects_cpu(yolo_blobs_, types_, nms_, yolo_param_.model_param(),
                                light_vis_conf_threshold_, light_swt_conf_threshold_,
                                overlapped_.get(), idx_sm_.get(), &(detected_objects_));


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

                overlapped_.reset(new caffe::Blob<float>(std::vector<int>{obj_k_, obj_k_}));
                //overlapped_->cpu_data();
                idx_sm_.reset(new caffe::Blob<float>(std::vector<int>{obj_k_}));

                yolo_blobs_.anchor_blob.reset(
                        new caffe::Blob<float>(1, 1, static_cast<int>(anchors_.size() / 2), 2));
                yolo_blobs_.expand_blob.reset(
                        new caffe::Blob<float>(1, 1, 1, static_cast<int>(expands_.size())));
                auto expand_cpu_data = yolo_blobs_.expand_blob->mutable_cpu_data();
                memcpy(expand_cpu_data, expands_.data(), expands_.size() * sizeof(float));
                auto anchor_cpu_data = yolo_blobs_.anchor_blob->mutable_cpu_data();
                memcpy(anchor_cpu_data, anchors_.data(), anchors_.size() * sizeof(float));

                yolo_blobs_.det1_loc_blob =
                        inference_->get_blob(yolo_param_.net_param().det1_loc_blob());
                yolo_blobs_.det1_obj_blob =
                        inference_->get_blob(yolo_param_.net_param().det1_obj_blob());
                yolo_blobs_.det1_cls_blob =
                        inference_->get_blob(yolo_param_.net_param().det1_cls_blob());
                yolo_blobs_.det1_ori_conf_blob =
                        inference_->get_blob(yolo_param_.net_param().det1_ori_conf_blob());
                yolo_blobs_.det1_ori_blob =
                        inference_->get_blob(yolo_param_.net_param().det1_ori_blob());
                yolo_blobs_.det1_dim_blob =
                        inference_->get_blob(yolo_param_.net_param().det1_dim_blob());
                yolo_blobs_.det2_loc_blob =
                        inference_->get_blob(yolo_param_.net_param().det2_loc_blob());
                yolo_blobs_.det2_obj_blob =
                        inference_->get_blob(yolo_param_.net_param().det2_obj_blob());
                yolo_blobs_.det2_cls_blob =
                        inference_->get_blob(yolo_param_.net_param().det2_cls_blob());
                yolo_blobs_.det2_ori_conf_blob =
                        inference_->get_blob(yolo_param_.net_param().det2_ori_conf_blob());
                yolo_blobs_.det2_ori_blob =
                        inference_->get_blob(yolo_param_.net_param().det2_ori_blob());
                yolo_blobs_.det2_dim_blob =
                        inference_->get_blob(yolo_param_.net_param().det2_dim_blob());
                yolo_blobs_.det3_loc_blob =
                        inference_->get_blob(yolo_param_.net_param().det3_loc_blob());
                yolo_blobs_.det3_obj_blob =
                        inference_->get_blob(yolo_param_.net_param().det3_obj_blob());
                yolo_blobs_.det3_cls_blob =
                        inference_->get_blob(yolo_param_.net_param().det3_cls_blob());
                yolo_blobs_.det3_ori_conf_blob =
                        inference_->get_blob(yolo_param_.net_param().det3_ori_conf_blob());
                yolo_blobs_.det3_ori_blob =
                        inference_->get_blob(yolo_param_.net_param().det3_ori_blob());
                yolo_blobs_.det3_dim_blob =
                        inference_->get_blob(yolo_param_.net_param().det3_dim_blob());

                yolo_blobs_.lof_blob =
                        inference_->get_blob(yolo_param_.net_param().lof_blob());
                yolo_blobs_.lor_blob =
                        inference_->get_blob(yolo_param_.net_param().lor_blob());

                yolo_blobs_.brvis_blob =
                        inference_->get_blob(yolo_param_.net_param().brvis_blob());
                yolo_blobs_.brswt_blob =
                        inference_->get_blob(yolo_param_.net_param().brswt_blob());
                yolo_blobs_.ltvis_blob =
                        inference_->get_blob(yolo_param_.net_param().ltvis_blob());
                yolo_blobs_.ltswt_blob =
                        inference_->get_blob(yolo_param_.net_param().ltswt_blob());
                yolo_blobs_.rtvis_blob =
                        inference_->get_blob(yolo_param_.net_param().rtvis_blob());
                yolo_blobs_.rtswt_blob =
                        inference_->get_blob(yolo_param_.net_param().rtswt_blob());

                yolo_blobs_.area_id_blob =
                        inference_->get_blob(yolo_param_.net_param().area_id_blob());
                yolo_blobs_.visible_ratio_blob =
                        inference_->get_blob(yolo_param_.net_param().visible_ratio_blob());
                yolo_blobs_.cut_off_ratio_blob =
                        inference_->get_blob(yolo_param_.net_param().cut_off_ratio_blob());


            }
        }//camera
    }//perception
}//apollo
