/******************************************************************************
 * Copyright 2018 The Apollo Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/

#include "region_output.h"
#include <memory>
#include "cyber/common/log.h"
#include <functional>
#include "boost/iterator/counting_iterator.hpp"

namespace apollo {
    namespace perception {
        namespace camera {

            typedef pair<int, float> PAIR;

            bool _cmp_by_value(const PAIR &lhs, const PAIR &rhs) {
                return lhs.second < rhs.second;
            }

            struct CmpByValue {
                bool operator()(const PAIR &lhs, const PAIR &rhs) {
                    return lhs.second < rhs.second;
                }
            };

            template<typename K, typename V>
            void sort_by_value(std::vector<K> &key, std::vector<V> &value) {
                CHECK(key.size() == value.size());
                std::map<K, V> k_v_map;
                for (int i = 0; i < key.size(); ++i) {
                    k_v_map.insert(make_pair(key[i], value[i]));
                }
                //将map中元素转存到vector中
                std::vector<PAIR> key_val_vec(k_v_map.begin(), k_v_map.end());
                std::sort(key_val_vec.begin(), key_val_vec.end(), CmpByValue());

                for (int iter = 0; iter < key.size(); ++iter) {
                    key[iter] = key_val_vec[iter].first;
                    value[iter] = key_val_vec[iter].second;
                }

            }

            float bbox_size_cpu(const float *bbox,
                                const bool normalized) {
                if (bbox[2] <= bbox[0] || bbox[3] <= bbox[1]) {
                    // If bbox is invalid (e.g. xmax < xmin or ymax < ymin), return 0.
                    return 0.f; // NOLINT
                } else {
                    const float width = bbox[2] - bbox[0];
                    const float height = bbox[3] - bbox[1];
                    if (normalized) {
                        return width * height;
                    } else {
                        // If bbox is not within range [0, 1].
                        return (width + 1) * (height + 1);
                    }
                }
            }

            float jaccard_overlap_cpu(const float *bbox1,
                                      const float *bbox2) {
                if (bbox2[0] > bbox1[2] || bbox2[2] < bbox1[0] ||
                    bbox2[1] > bbox1[3] || bbox2[3] < bbox1[1]) {
                    return float(0.); // NOLINT
                } else {
                    const float inter_xmin = std::max(bbox1[0], bbox2[0]);
                    const float inter_ymin = std::max(bbox1[1], bbox2[1]);
                    const float inter_xmax = std::min(bbox1[2], bbox2[2]);
                    const float inter_ymax = std::min(bbox1[3], bbox2[3]);

                    const float inter_width = inter_xmax - inter_xmin;
                    const float inter_height = inter_ymax - inter_ymin;
                    const float inter_size = inter_width * inter_height;

                    const float bbox1_size = bbox_size_cpu(bbox1, true);
                    const float bbox2_size = bbox_size_cpu(bbox2, true);

                    return inter_size / (bbox1_size + bbox2_size - inter_size);
                }
            }

            void get_intersect_bbox(const NormalizedBBox &bbox1,
                                    const NormalizedBBox &bbox2,
                                    NormalizedBBox *intersect_bbox) {
                if (bbox2.xmin > bbox1.xmax || bbox2.xmax < bbox1.xmin ||
                    bbox2.ymin > bbox1.ymax || bbox2.ymax < bbox1.ymin) {
                    // Return [0, 0, 0, 0] if there is no intersection.
                    intersect_bbox->xmin = 0;
                    intersect_bbox->ymin = 0;
                    intersect_bbox->xmax = 0;
                    intersect_bbox->ymax = 0;
                } else {
                    intersect_bbox->xmin = std::max(bbox1.xmin, bbox2.xmin);
                    intersect_bbox->ymin = std::max(bbox1.ymin, bbox2.ymin);
                    intersect_bbox->xmax = std::min(bbox1.xmax, bbox2.xmax);
                    intersect_bbox->ymax = std::min(bbox1.ymax, bbox2.ymax);
                }
            }

            float get_bbox_size(const NormalizedBBox &bbox) {
                if (bbox.xmax < bbox.xmin || bbox.ymax < bbox.ymin) {
                    // If bbox is invalid (e.g. xmax < xmin or ymax < ymin), return 0.
                    return 0;
                } else {
                    if (bbox.size >= 0) {
                        return bbox.size;
                    } else {
                        float width = bbox.xmax - bbox.xmin;
                        float height = bbox.ymax - bbox.ymin;
                        return width * height;
                    }
                }
            }

            float get_jaccard_overlap(const NormalizedBBox &bbox1,
                                      const NormalizedBBox &bbox2) {
                NormalizedBBox intersect_bbox;
                get_intersect_bbox(bbox1, bbox2, &intersect_bbox);
                float intersect_width = 0.f;
                float intersect_height = 0.f;
                intersect_width = intersect_bbox.xmax - intersect_bbox.xmin;
                intersect_height = intersect_bbox.ymax - intersect_bbox.ymin;

                if (intersect_width > 0 && intersect_height > 0) {
                    float intersect_size = intersect_width * intersect_height;
                    float bbox1_size = get_bbox_size(bbox1);
                    float bbox2_size = get_bbox_size(bbox2);
                    return intersect_size / (bbox1_size + bbox2_size - intersect_size);
                } else {
                    return 0.;
                }
            }

            void get_max_score_index(const std::vector<float> &scores,
                                     const float threshold, const int top_k,
                                     std::vector<std::pair<float, int> > *score_index_vec) {
                // Generate index score pairs.
                for (int i = 0; i < static_cast<int>(scores.size()); ++i) {
                    if (scores[i] > threshold) {
                        score_index_vec->push_back(std::make_pair(scores[i], i));
                    }
                }

                // Sort the score pair according to the scores in descending order
                std::stable_sort(score_index_vec->begin(), score_index_vec->end(),
                                 sort_score_pair_descend<int>);

                // Keep top_k scores if needed.
                if (top_k > -1 && top_k < static_cast<int>(score_index_vec->size())) {
                    score_index_vec->resize(top_k);
                }
            }

            void apply_softnms_fast(const std::vector<NormalizedBBox> &bboxes,
                                    std::vector<float> *scores, const float score_threshold,
                                    const float nms_threshold, const int top_k,
                                    std::vector<int> *indices, bool is_linear,
                                    const float sigma) {
                // Sanity check.
                CHECK_EQ(bboxes.size(), scores->size())
                    << "bboxes and scores have different size.";

                // Get top_k scores (with corresponding indices).
                std::vector<std::pair<float, int> > score_index_vec;
                get_max_score_index(*scores, score_threshold, top_k, &score_index_vec);

                // Do nms.
                indices->clear();
                while (!score_index_vec.empty()) {
                    auto best_it =
                            max_element(std::begin(score_index_vec), std::end(score_index_vec));
                    const int best_idx = (*best_it).second;
                    score_index_vec.erase(best_it);
                    const NormalizedBBox &best_bbox = bboxes[best_idx];
                    indices->push_back(best_idx);
                    for (std::vector<std::pair<float, int> >::iterator it =
                            score_index_vec.begin();
                         it != score_index_vec.end();) {
                        int cur_idx = it->second;
                        const NormalizedBBox &cur_bbox = bboxes[cur_idx];

                        float cur_overlap = 0.;
                        cur_overlap = get_jaccard_overlap(best_bbox, cur_bbox);
                        if (is_linear) {
                            (*scores)[cur_idx] *= static_cast<float>((1.0 - cur_overlap));
                        } else {
                            (*scores)[cur_idx] *=
                                    static_cast<float>(exp(-1.0 * pow(cur_overlap, 2) / sigma));
                        }
                        ++it;
                    }
                }
            }

            void apply_boxvoting_fast(std::vector<NormalizedBBox> *bboxes,
                                      std::vector<float> *scores,
                                      const float conf_threshold, const float nms_threshold,
                                      const float sigma, std::vector<int> *indices) {
                if (bboxes->size() == 0) {
                    return;
                }
                indices->clear();
                for (int i = 0; i < static_cast<int>(bboxes->size()); i++) {
                    (*bboxes)[i].mask = false;
                    if ((*scores)[i] > conf_threshold) {
                        indices->push_back(i);
                    }
                }
                for (int count = 0; count < static_cast<int>(indices->size()); count++) {
                    int max_box_idx = 0;

                    for (int i = 1; i < static_cast<int>(indices->size()); i++) {
                        int idx = indices->at(i);
                        if ((*bboxes)[idx].mask) {
                            continue;
                        }
                        if ((*scores)[idx] > (*scores)[max_box_idx]) {
                            max_box_idx = idx;
                        }
                    }

                    NormalizedBBox &best_bbox = (*bboxes)[max_box_idx];
                    best_bbox.score = (*scores)[max_box_idx];
                    best_bbox.mask = true;
                    float s_vt = (*scores)[max_box_idx];
                    float x1_vt = best_bbox.xmin * s_vt;
                    float x2_vt = best_bbox.xmax * s_vt;
                    float y1_vt = best_bbox.ymin * s_vt;
                    float y2_vt = best_bbox.ymax * s_vt;
                    for (int i = 0; i < static_cast<int>(indices->size()); i++) {
                        int sub_it = indices->at(i);
                        if ((*bboxes)[sub_it].mask) {
                            continue;
                        }
                        float cur_overlap = 0.;
                        cur_overlap = get_jaccard_overlap(best_bbox, (*bboxes)[sub_it]);
                        if (sigma == 0) {
                            (*bboxes)[sub_it].mask = true;
                        } else {
                            (*scores)[sub_it] *=
                                    static_cast<float>(exp(-1.0 * pow(cur_overlap, 2) / sigma));
                        }
                        (*bboxes)[sub_it].score = (*scores)[sub_it];

                        // Remove it if necessary
                        if (cur_overlap > nms_threshold) {
                            float s_vt_cur = (*bboxes)[sub_it].score;
                            s_vt += s_vt_cur;
                            x1_vt += (*bboxes)[sub_it].xmin * s_vt_cur;
                            x2_vt += (*bboxes)[sub_it].xmax * s_vt_cur;
                            y1_vt += (*bboxes)[sub_it].ymin * s_vt_cur;
                            y2_vt += (*bboxes)[sub_it].ymax * s_vt_cur;
                        }
                    }
                    if (s_vt > 0.0001) {
                        (*bboxes)[max_box_idx].xmin = x1_vt / s_vt;
                        (*bboxes)[max_box_idx].xmax = x2_vt / s_vt;
                        (*bboxes)[max_box_idx].ymin = y1_vt / s_vt;
                        (*bboxes)[max_box_idx].ymax = y2_vt / s_vt;
                    }
                }
            }

            void apply_nms_fast(const std::vector<NormalizedBBox> &bboxes,
                                const std::vector<float> &scores,
                                const float score_threshold, const float nms_threshold,
                                const float eta, const int top_k,
                                std::vector<int> *indices) {
                // Sanity check.
                CHECK_EQ(bboxes.size(), scores.size())
                    << "bboxes and scores have different size.";

                // Get top_k scores (with corresponding indices).
                std::vector<std::pair<float, int> > score_index_vec;
                get_max_score_index(scores, score_threshold, top_k, &score_index_vec);

                // Do nms.
                float adaptive_threshold = nms_threshold;
                indices->clear();
                while (!score_index_vec.empty()) {
                    const int idx = score_index_vec.front().second;
                    bool keep = true;
                    for (int k = 0; k < static_cast<int>(indices->size()); ++k) {
                        if (keep) {
                            const int kept_idx = (*indices)[k];
                            float overlap = get_jaccard_overlap(bboxes[idx], bboxes[kept_idx]);
                            keep = overlap <= adaptive_threshold;
                        } else {
                            break;
                        }
                    }
                    if (keep) {
                        indices->push_back(idx);
                    }
                    score_index_vec.erase(score_index_vec.begin());
                    if (keep && eta < 1 && adaptive_threshold > 0.5) {
                        adaptive_threshold *= eta;
                    }
                }
            }

            void filter_bbox(const MinDims &min_dims,
                             std::vector<base::ObjectPtr> *objects) {
                int valid_obj_idx = 0;
                int total_obj_idx = 0;
                while (total_obj_idx < static_cast<int>(objects->size())) {
                    const auto &obj = (*objects)[total_obj_idx];
                    if ((obj->camera_supplement.box.ymax - obj->camera_supplement.box.ymin) >=
                        min_dims.min_2d_height &&
                        (min_dims.min_3d_height <= 0 ||
                         obj->size[2] >= min_dims.min_3d_height) &&
                        (min_dims.min_3d_width <= 0 || obj->size[1] >= min_dims.min_3d_width) &&
                        (min_dims.min_3d_length <= 0 ||
                         obj->size[0] >= min_dims.min_3d_length)) {
                        (*objects)[valid_obj_idx] = (*objects)[total_obj_idx];
                        ++valid_obj_idx;
                    }
                    ++total_obj_idx;
                }
                AINFO << valid_obj_idx << " of " << total_obj_idx << " obstacles kept";
                objects->resize(valid_obj_idx);
                AINFO << "Number of detected obstacles: " << objects->size();
            }

            void recover_bbox(int roi_w, int roi_h, int offset_y,
                              std::vector<base::ObjectPtr> *objects) {
                for (auto &obj : *objects) {
                    float xmin = obj->camera_supplement.box.xmin;
                    float ymin = obj->camera_supplement.box.ymin;
                    float xmax = obj->camera_supplement.box.xmax;
                    float ymax = obj->camera_supplement.box.ymax;
                    int x = static_cast<int>(xmin * static_cast<float>(roi_w));
                    int w = static_cast<int>((xmax - xmin) * static_cast<float>(roi_w));
                    int y = static_cast<int>(ymin * static_cast<float>(roi_h)) + offset_y;
                    int h = static_cast<int>((ymax - ymin) * static_cast<float>(roi_h));
                    base::RectF rect_det(static_cast<float>(x), static_cast<float>(y),
                                         static_cast<float>(w), static_cast<float>(h));
                    base::RectF rect_img(0, 0, static_cast<float>(roi_w),
                                         static_cast<float>(roi_h + offset_y));
                    base::RectF rect = rect_det & rect_img;
                    obj->camera_supplement.box = rect;

                    double eps = 1e-2;

                    // Truncation assignment based on bbox positions
                    if ((ymin < eps) || (ymax >= 1.0 - eps)) {
                        obj->camera_supplement.truncated_vertical = 0.5;
                    } else {
                        obj->camera_supplement.truncated_vertical = 0.0;
                    }
                    if ((xmin < eps) || (xmax >= 1.0 - eps)) {
                        obj->camera_supplement.truncated_horizontal = 0.5;
                    } else {
                        obj->camera_supplement.truncated_horizontal = 0.0;
                    }

                    obj->camera_supplement.front_box.xmin *= static_cast<float>(roi_w);
                    obj->camera_supplement.front_box.ymin *= static_cast<float>(roi_h);
                    obj->camera_supplement.front_box.xmax *= static_cast<float>(roi_w);
                    obj->camera_supplement.front_box.ymax *= static_cast<float>(roi_h);

                    obj->camera_supplement.back_box.xmin *= static_cast<float>(roi_w);
                    obj->camera_supplement.back_box.ymin *= static_cast<float>(roi_h);
                    obj->camera_supplement.back_box.xmax *= static_cast<float>(roi_w);
                    obj->camera_supplement.back_box.ymax *= static_cast<float>(roi_h);

                    obj->camera_supplement.front_box.ymin += static_cast<float>(offset_y);
                    obj->camera_supplement.front_box.ymax += static_cast<float>(offset_y);
                    obj->camera_supplement.back_box.ymin += static_cast<float>(offset_y);
                    obj->camera_supplement.back_box.ymax += static_cast<float>(offset_y);
                }
            }

            void fill_base(base::ObjectPtr obj, const float *bbox) {
                obj->camera_supplement.box.xmin = bbox[0];
                obj->camera_supplement.box.ymin = bbox[1];
                obj->camera_supplement.box.xmax = bbox[2];
                obj->camera_supplement.box.ymax = bbox[3];
            }

            void fill_bbox3d(bool with_box3d, base::ObjectPtr obj, const float *bbox) {
                if (with_box3d) {
                    obj->camera_supplement.alpha = bbox[0];
                    obj->size[2] = bbox[1];
                    obj->size[1] = bbox[2];
                    obj->size[0] = bbox[3];
                }
            }

            void fill_frbox(bool with_frbox, base::ObjectPtr obj, const float *bbox) {
                if (with_frbox) {
                    obj->camera_supplement.front_box.xmin = bbox[0];
                    obj->camera_supplement.front_box.ymin = bbox[1];
                    obj->camera_supplement.front_box.xmax = bbox[2];
                    obj->camera_supplement.front_box.ymax = bbox[3];

                    obj->camera_supplement.back_box.xmin = bbox[4];
                    obj->camera_supplement.back_box.ymin = bbox[5];
                    obj->camera_supplement.back_box.xmax = bbox[6];
                    obj->camera_supplement.back_box.ymax = bbox[7];
                }
            }

            void fill_lights(bool with_lights, base::ObjectPtr obj, const float *bbox) {
                if (with_lights) {
                    obj->car_light.brake_visible = bbox[0];
                    obj->car_light.brake_switch_on = bbox[1];
                    obj->car_light.left_turn_visible = bbox[2];
                    obj->car_light.left_turn_switch_on = bbox[3];
                    obj->car_light.right_turn_visible = bbox[4];
                    obj->car_light.right_turn_switch_on = bbox[5];
                }
            }

            void fill_ratios(bool with_ratios, base::ObjectPtr obj, const float *bbox) {
                if (with_ratios) {
                    // visible ratios of face a/b/c/d
                    obj->camera_supplement.visible_ratios[0] = bbox[0];
                    obj->camera_supplement.visible_ratios[1] = bbox[1];
                    obj->camera_supplement.visible_ratios[2] = bbox[2];
                    obj->camera_supplement.visible_ratios[3] = bbox[3];

                    // cut off on width and length (3D)
                    obj->camera_supplement.cut_off_ratios[0] = bbox[4];
                    obj->camera_supplement.cut_off_ratios[1] = bbox[5];
                    // cut off on left and right side (2D)
                    obj->camera_supplement.cut_off_ratios[2] = bbox[6];
                    obj->camera_supplement.cut_off_ratios[3] = bbox[7];
                }
            }

            void fill_area_id(bool with_flag, base::ObjectPtr obj, const float *data) {
                if (with_flag) {
                    obj->camera_supplement.area_id = static_cast<int>(data[0]);
                    // obj->camera_supplement.area_id_prob = data[1];
                }
            }

            int get_area_id(float visible_ratios[4]) {
                int area_id = 0;
                int max_face = 0;
                for (int i = 1; i < 4; i++) {
                    if (visible_ratios[i] > visible_ratios[max_face]) {
                        max_face = i;
                    }
                }
                int left_face = (max_face + 1) % 4;
                int right_face = (max_face + 3) % 4;
                const float eps = 1e-3f;
                float max_ratio = visible_ratios[max_face];
                float left_ratio = visible_ratios[left_face];
                float right_ratio = visible_ratios[right_face];
                memset(visible_ratios, 0, 4 * sizeof(visible_ratios[0]));
                if (left_ratio < eps && right_ratio < eps) {
                    area_id = (max_face * 2 + 1);
                    visible_ratios[max_face] = 1.f;
                } else if (left_ratio > right_ratio) {
                    area_id = (max_face * 2 + 2);
                    auto &&sum_ratio = left_ratio + max_ratio;
                    visible_ratios[max_face] = max_ratio / sum_ratio;
                    visible_ratios[left_face] = left_ratio / sum_ratio;
                } else {
                    area_id = (max_face * 2);
                    if (area_id == 0) {
                        area_id = 8;
                    }
                    auto &&sum_ratio = right_ratio + max_ratio;
                    visible_ratios[max_face] = max_ratio / sum_ratio;
                    visible_ratios[right_face] = right_ratio / sum_ratio;
                }
                return area_id;
            }

            void apply_nms(const bool *overlapped,
                           const int num,
                           std::vector<int> *indices) {
                std::vector<int> index_vec(boost::counting_iterator<int>(0),
                                           boost::counting_iterator<int>(num));
                // Do nms.
                indices->clear();
                while (index_vec.size() != 0) {
                    // Get the current highest score box.
                    int best_idx = index_vec.front();
                    indices->push_back(best_idx);
                    // Erase the best box.
                    index_vec.erase(index_vec.begin());

                    for (std::vector<int>::iterator it = index_vec.begin();
                         it != index_vec.end();) {
                        int cur_idx = *it;

                        // Remove it if necessary
                        if (overlapped[best_idx * num + cur_idx]) {
                            it = index_vec.erase(it);
                        } else {
                            ++it;
                        }
                    }
                }
            }
            const float *get_cpu_data(bool flag, const caffe::Blob<float> &blob) {
                return flag ? blob.cpu_data() : nullptr;
            }


            void get_object_kernel(
                    int n,
                    const float *loc_data,
                    const float *obj_data,
                    const float *cls_data,
                    const float *ori_data,
                    const float *dim_data,
                    const float *lof_data,
                    const float *lor_data,
                    const float *area_id_data,
                    const float *visible_ratio_data,
                    const float *cut_off_ratio_data,
                    const float *brvis_data,
                    const float *brswt_data,
                    const float *ltvis_data,
                    const float *ltswt_data,
                    const float *rtvis_data,
                    const float *rtswt_data,
                    const float *anchor_data,
                    const float *expand_data,
                    int width,
                    int height,
                    int num_anchors,
                    int num_classes,
                    float confidence_threshold,
                    float light_vis_conf_threshold,
                    float light_swt_conf_threshold,
                    bool with_box3d,
                    bool with_frbox,
                    bool with_lights,
                    bool with_ratios,
                    bool multi_scale,
                    int num_areas,
                    float *res_box_data,
                    float *res_cls_data,
                    int res_cls_offset,
                    int all_scales_num_candidates
            ) {
                //loop candidate box
                for (int i = 0; i < (n); ++i) {

                    int box_block = kBoxBlockSize;
                    int idx = i;
                    int c = idx % num_anchors;
                    idx = idx / num_anchors;
                    int w = idx % width;
                    idx = idx / width;
                    int h = idx;
                    int loc_index = (h * width + w) * num_anchors + c;//anchor index :0-51839
                    //AINFO << "loc_index = " << loc_index;
                    int offset_loc = loc_index * 4;
                    int offset_cls = loc_index * num_classes;
                    float scale = obj_data[loc_index];
                    float cx =
                            (w + sigmoid<float>(loc_data[offset_loc + 0])) / width; //c_x = sigmoid(t_x) + w  w,h为网格位置
                    float cy = (h + sigmoid<float>(loc_data[offset_loc + 1])) / height; //c_y = sigmoid(t_y) + h
                    float hw = exp(std::max(minExpPower, std::min(loc_data[offset_loc + 2], maxExpPower))) *
                               anchor_data[2 * c] / width * 0.5; //box_w = anchor_w * exp(t_w)
                    float hh = exp(std::max(minExpPower, std::min(loc_data[offset_loc + 3], maxExpPower))) *
                               anchor_data[2 * c + 1] / height * 0.5; //box_h = anchor_h * exp(t_h)
                    float max_prob = 0.f;
                    int max_index = 0;
                    for (int k = 0; k < num_classes; ++k) {
                        float prob = cls_data[offset_cls + k] * scale;
                        res_cls_data[k * all_scales_num_candidates + res_cls_offset + i] = prob;
                        if (prob > max_prob) {
                            max_prob = prob;
                            max_index = k;
                        }
                    } // find max_prob

                    res_cls_data[num_classes * all_scales_num_candidates + res_cls_offset + i] = max_prob;

                    auto &&dst_ptr = res_box_data + i * box_block; //右值引用，对象是临时的,要求=右边必须是右值
                    hw += expand_data[max_index]; //根据该box概率最大的类别的expand扩充hw
                    dst_ptr[0] = cx - hw;
                    dst_ptr[1] = cy - hh;
                    dst_ptr[2] = cx + hw;
                    dst_ptr[3] = cy + hh;

                    if (with_box3d) {
                        int offset_ori = loc_index * 2;
                        dst_ptr[4] = atan2(ori_data[offset_ori + 1], ori_data[offset_ori]);

                        int offset_dim = loc_index * 3;
                        if (multi_scale) {
                            offset_dim = loc_index * num_classes * 3 + max_index * 3;
                        }
                        dst_ptr[5] = dim_data[offset_dim + 0];
                        dst_ptr[6] = dim_data[offset_dim + 1];
                        dst_ptr[7] = dim_data[offset_dim + 2];

                    }

                    if (with_frbox) {
                        {
                            int offset_lof = loc_index * 4;
                            auto &&src_ptr = lof_data + offset_lof;
                            auto sb_x = src_ptr[0] * hw * 2 + cx;
                            auto sb_y = src_ptr[1] * hh * 2 + cy;
                            auto sb_hw = exp(src_ptr[2]) * hw;
                            auto sb_hh = exp(src_ptr[3]) * hh;

                            dst_ptr[8] = sb_x - sb_hw;
                            dst_ptr[9] = sb_y - sb_hh;
                            dst_ptr[10] = sb_x + sb_hw;
                            dst_ptr[11] = sb_y + sb_hh;
                        }
                        {
                            int offset_lor = loc_index * 4;
                            auto &&src_ptr = lor_data + offset_lor;
                            auto sb_x = src_ptr[0] * hw * 2 + cx;
                            auto sb_y = src_ptr[1] * hh * 2 + cy;
                            auto sb_hw = exp(src_ptr[2]) * hw;
                            auto sb_hh = exp(src_ptr[3]) * hh;
                            dst_ptr[12] = sb_x - sb_hw;
                            dst_ptr[13] = sb_y - sb_hh;
                            dst_ptr[14] = sb_x + sb_hw;
                            dst_ptr[15] = sb_y + sb_hh;

                        }

                    }

                    if (with_lights) {
                        dst_ptr[16] = sigmoid(brvis_data[loc_index]);
                        dst_ptr[17] = sigmoid(brswt_data[loc_index]);
                        dst_ptr[18] = sigmoid(ltvis_data[loc_index]);
                        dst_ptr[19] = sigmoid(ltswt_data[loc_index]);
                        dst_ptr[20] = sigmoid(rtvis_data[loc_index]);
                        dst_ptr[21] = sigmoid(rtswt_data[loc_index]);

                        dst_ptr[16] = dst_ptr[16] > light_vis_conf_threshold ? dst_ptr[16] : 0;
                        dst_ptr[18] = dst_ptr[18] > light_vis_conf_threshold ? dst_ptr[18] : 0;
                        dst_ptr[20] = dst_ptr[20] > light_vis_conf_threshold ? dst_ptr[20] : 0;

                        float swt_score = 0;
                        swt_score = dst_ptr[16] * dst_ptr[17];
                        dst_ptr[17] = swt_score > light_swt_conf_threshold ? swt_score : 0;

                        swt_score = dst_ptr[18] * dst_ptr[19];
                        dst_ptr[19] = swt_score > light_swt_conf_threshold ? swt_score : 0;

                        swt_score = dst_ptr[20] * dst_ptr[21];
                        dst_ptr[21] = swt_score > light_swt_conf_threshold ? swt_score : 0;
                    }

                    if (with_ratios) {
                        // 0~3: cos2, left, visa, visb
                        auto vis_pred = visible_ratio_data + loc_index * 4;
                        auto vis_ptr = dst_ptr + 22;//对应到vis在dst_ptr中的位置
                        vis_ptr[0] = vis_ptr[1] = vis_ptr[2] = vis_ptr[3] = 0;
                        const float hi_th = 0.75;
                        const float lo_th = 1.f - hi_th;
                        if (vis_pred[2] >= hi_th && vis_pred[3] >= hi_th) {         // 2 (1, 3)
                            vis_ptr[0] = vis_pred[0];
                            vis_ptr[1] = 1 - vis_pred[0];
                        } else if (vis_pred[2] <= lo_th && vis_pred[3] >= hi_th) {  // 4 (3, 5)
                            vis_ptr[2] = vis_pred[0];
                            vis_ptr[1] = 1 - vis_pred[0];
                        } else if (vis_pred[2] <= lo_th && vis_pred[3] <= lo_th) {  // 6 (5, 7)
                            vis_ptr[2] = vis_pred[0];
                            vis_ptr[3] = 1 - vis_pred[0];
                        } else if (vis_pred[2] >= hi_th && vis_pred[3] <= lo_th) {  // 8 (7, 1)
                            vis_ptr[0] = vis_pred[0];
                            vis_ptr[3] = 1 - vis_pred[0];
                        } else {
                            vis_ptr[2] = vis_pred[0];
                            if (vis_pred[1] > 0.5) {
                                vis_ptr[1] = 1 - vis_pred[0];
                            } else {
                                vis_ptr[3] = 1 - vis_pred[0];
                            }
                        }

                        int offset_cut = loc_index * 4;
                        dst_ptr[26] = cut_off_ratio_data[offset_cut + 0];
                        dst_ptr[27] = cut_off_ratio_data[offset_cut + 1];
                        dst_ptr[28] = cut_off_ratio_data[offset_cut + 2];
                        dst_ptr[29] = cut_off_ratio_data[offset_cut + 3];

                    }

                    if (num_areas > 0) {
                        int offset_area_id = loc_index * num_areas;
                        int max_area_id = 0;
                        for (int area_id = 1; area_id < num_areas; ++area_id) {
                            if (area_id_data[offset_area_id + area_id] >
                                area_id_data[offset_area_id + max_area_id]) {
                                max_area_id = area_id;
                            }
                        }
                        dst_ptr[30] = max_area_id + 1;
                        dst_ptr[31] = area_id_data[offset_area_id + max_area_id];
                    }


                }


            }


            void sort_by_key() {
                typedef pair<int, float> PAIR;
            }


            void compute_overlapped_by_idx_kernel(
                    const int nthreads,
                    const float *bbox_data,
                    const int bbox_step,
                    const float overlap_threshold,
                    const float *idx,
                    const int num_idx,
                    float *overlapped_data) {
                for (int index = 0; index < (nthreads); ++index) {
                    const int j = index % num_idx;
                    const int i = index / num_idx;

                    if (i == j) {
                        //Ignore same bbox.
                        continue;
                    }
                    //Compute overlap between i-th bbox and j-th bbox.
                    const int start_loc_i = idx[i] * bbox_step;
                    const int start_loc_j = idx[j] * bbox_step;
                    const float overlap = jaccard_overlap_cpu(bbox_data + start_loc_i,
                                                              bbox_data + start_loc_j);
                    overlapped_data[index] = overlap > overlap_threshold;

                }


            }

            void compute_overlapped_by_idx_cpu(const int nthreads,
                                               const float *bbox_data,
                                               const int bbox_step,
                                               const float overlap_threshold,
                                               const float *idx,
                                               const int num_idx,
                                               float *overlapped_data) {

                compute_overlapped_by_idx_kernel(nthreads, bbox_data, bbox_step, overlap_threshold, idx, num_idx,
                                                 overlapped_data);

            }

            void apply_nms_cpu(const float *bbox_data,
                               const float *conf_data,
                               const std::vector<int> &origin_indices,
                               const int bbox_step,
                               const float confidence_threshold,
                               const int top_k,
                               const float nms_threshold,
                               std::vector<int> *indices,
                               caffe::Blob<float> *overlapped,
                               caffe::Blob<float> *idx_sm
            ) {
                // Keep part of detections whose scores are higher than confidence threshold.
                std::vector<int> idx; //存储大于thresh的框的索引
                std::vector<float> confidences;//存储大于thresh的candidate的置信度
                for (auto i : origin_indices) {
                    if (conf_data[i] > confidence_threshold) {
                        idx.push_back(i);
                        confidences.push_back(conf_data[i]);
                    }
                }
                int num_remain = confidences.size();
                if (num_remain == 0) {
                    return;
                }
                for (int i = 0; i < num_remain; i++)
                    AINFO << "confidences: " << confidences[i] << "  anchor_idx: " << idx[i];
                sort_by_value<int, float>(idx, confidences);
//                for (int i = 0; i < num_remain; i++)
//                    AINFO << "_confidences: " << confidences[i] << "  _anchor_idx: " << idx[i];

                if (top_k > -1 && top_k < num_remain) {
                    num_remain = top_k; //最多取前top_k个
                }
                float *idx_data = (idx_sm->mutable_cpu_data());
                std::copy(idx.begin(), idx.begin() + num_remain, idx_data);

                overlapped->Reshape(std::vector<int>{num_remain, num_remain});
                float *overlapped_data = (overlapped->mutable_cpu_data());

                compute_overlapped_by_idx_cpu(overlapped->count(),
                                              bbox_data,
                                              bbox_step,
                                              nms_threshold,
                                              idx_sm->cpu_data(),
                                              num_remain,
                                              overlapped_data);
                //Do non-maximum suppression based on overlapped results.
                const bool *overlapped_results = (const bool*) overlapped->cpu_data();
                std::vector<int> selected_indices;

                apply_nms(overlapped_results,num_remain, & selected_indices);
                //Put back the selected information
                for(size_t i= 0;i<selected_indices.size();++i){
                    indices->push_back(idx[selected_indices[i]]);
                }

            }


        }  // namespace camera
    }  // namespace perception
}  // namespace apollo
