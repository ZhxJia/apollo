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
#include "modules/perception/camera/lib/obstacle/transformer/multicue/obj_mapper.h"

namespace apollo {
namespace perception {
namespace camera {

void ObjMapperParams::set_default() {
  nr_bins_z = 15;
  nr_bins_ry = 36;
  boundary_len = 20;
  max_nr_iter = 10;

  small_bbox_height = 50.0f;
  factor_small = 0.6f;
  learning_r = 0.7f;
  reproj_err = 4 * sqrtf(2.0f);
  rz_ratio = 0.1f;
  abnormal_h_veh = 0.8f;
  stable_w = 0.5f;
  occ_ratio = 0.9f;
  depth_min = 0.1f;
  dist_far = 100.0f;
  eps_mapper = 1e-5f;

  iou_suc = 0.5f;
  iou_high = 0.7f;
  angle_resolution_degree = 10.0f;
}

bool ObjMapper::SolveCenterFromNearestVerticalEdge(const float *bbox,
                                                   const float *hwl, float ry,
                                                   float *center,
                                                   float *center_2d) const {
  center[0] = center[1] = center[2] = 0.0f;
  float height_bbox = bbox[3] - bbox[1]; 
  float width_bbox = bbox[2] - bbox[0];
  CHECK(width_bbox > 0.0f && height_bbox > 0.0f);

  if (common::IRound(bbox[3]) >= height_ - 1) {
    height_bbox /= params_.occ_ratio; //0.9
  } //框超出图像　

  float f = (k_mat_[0] + k_mat_[4]) / 2;//相机的焦距(单位像素)
  float depth = f * hwl[0] * common::IRec(height_bbox);//f*H(米)/h(像素)　相似得到

  // compensate from the nearest vertical edge to center  从最近垂直边的高度计算深度　最后加上到物体中心的补偿
  const float PI = common::Constant<float>::PI();
  float theta_bbox = static_cast<float>(atan(hwl[1] * common::IRec(hwl[2])));//theta_box = arctan(W/L)
  float radius_bbox =
      common::ISqrt(common::ISqr(hwl[2] / 2) + common::ISqr(hwl[1] / 2)); //半径

  float abs_ry = fabsf(ry); // 0- pi  弧度
  float theta_z = std::min(abs_ry, PI - abs_ry) + theta_bbox; // 0 - pi/2 +theta_box 
  theta_z = std::min(theta_z, PI - theta_z); //0 - pi/2
  depth += static_cast<float>(fabs(radius_bbox * sin(theta_z)));

  // back-project to solve center
  center_2d[0] = (bbox[0] + bbox[2]) / 2;
  center_2d[1] = (bbox[1] + bbox[3]) / 2;
  if (hwl[1] > params_.stable_w) { //0.5
    GetCenter(bbox, depth, ry, hwl, center, center_2d); //根据深度　反映射得到　3D 点的坐标，深度是由尺寸估计得到
  } else {
    center[2] = depth;
    UpdateCenterViaBackProjectZ(bbox, hwl, center_2d, center);
  }

  return center[2] > params_.depth_min;//0.1
}

bool ObjMapper::Solve3dBboxGivenOneFullBboxDimensionOrientation(
    const float *bbox, const float *hwl, float *ry, float *center) {
  const float PI = common::Constant<float>::PI();
  const float PI_HALF = PI / 2;
  const float small_angle_diff =
      common::IDegreeToRadians(params_.angle_resolution_degree);//10.0 -> 0.174532
  float center_2d[2] = {0};
  bool success =
      SolveCenterFromNearestVerticalEdge(bbox, hwl, *ry, center, center_2d);　//由深度信息反投影得到3d bbox中心点center　计算距离大于0.1时，返回true　
  float min_x = static_cast<float>(params_.boundary_len); //20
  float min_y = static_cast<float>(params_.boundary_len);
  float max_x = static_cast<float>(width_ - params_.boundary_len);
  float max_y = static_cast<float>(height_ - params_.boundary_len);
  bool truncated = bbox[0] <= min_x || bbox[2] >= max_x || bbox[1] <= min_y ||
                   bbox[3] >= max_y; 
  float dist_rough = sqrtf(common::ISqr(center[0]) + common::ISqr(center[2]));
  bool ry_pred_is_not_reliable = dist_rough > params_.dist_far &&
                                 bbox[3] - bbox[1] < params_.small_bbox_height; //距离超过100或者2d box高度小于50，则旋转角度的预测不可信
  if (ry_pred_is_not_reliable || std::abs(*ry - PI_HALF) < small_angle_diff ||
      std::abs(*ry + PI_HALF) < small_angle_diff) { //旋转角度预测不可信，或者旋转角度在正负pi/2之间
    *ry = *ry > 0.0f ? PI_HALF : -PI_HALF; //直接赋值pi/2 就是迎面　或者前方直行的车辆
  }
  if (!truncated) {
    PostRefineOrientation(bbox, hwl, center, ry);//根据3d角点投影的bbox 与检测的bbox的交并比分数，确定最终的转角ry
    success =
        SolveCenterFromNearestVerticalEdge(bbox, hwl, *ry, center, center_2d); //由于上面函数修正了ry ,因此重新计算3d中心点
    PostRefineZ(bbox, hwl, center_2d, *ry, center);//后处理，精细化深度z
  } else {
    FillRyScoreSingleBin(*ry);
  }
  return success &&
         GetProjectionScore(*ry, bbox, hwl, center, true) > params_.iou_suc; //最终所有的参数都精细化调整完，再次判断是否满足iou_suc阈值
}

bool ObjMapper::Solve3dBbox(const ObjMapperOptions &options, float center[3],
                            float hwl[3], float *ry) {
  // set default value for variance
  set_default_variance();
  float var_yaw = 0.0f;
  float var_z = 0.0f;

  // get input from options
  memcpy(hwl, options.hwl, sizeof(float) * 3);
  float bbox[4] = {0};
  memcpy(bbox, options.bbox, sizeof(float) * 4);
  *ry = options.ry;
  bool check_dimension = options.check_dimension;//检查尺寸
  int type_min_vol_index = options.type_min_vol_index;//该类型的最小体积(开头)在veh_hwl_中的索引 

  // check input hwl insanity 对于车辆检查尺寸
  if (options.is_veh && check_dimension) {
    assert(type_min_vol_index >= 0);
    const std::vector<float> &kVehHwl = object_template_manager_->VehHwl(); //得到全部的模板
    const float *tmplt_with_min_vol = &kVehHwl[type_min_vol_index]; //得到该车辆目标类型的模板尺寸所对应的起始索引
    float min_tmplt_vol =
        tmplt_with_min_vol[0] * tmplt_with_min_vol[1] * tmplt_with_min_vol[2];//得到最小体积
    float shrink_ratio_vol = common::ISqr(sqrtf(params_.iou_high));//0.7 体积缩小率
    shrink_ratio_vol *= shrink_ratio_vol;//0.49
    // float shrink_ratio_vol = sqrt(params_.iou_high);
    if (hwl[0] < params_.abnormal_h_veh ||
        hwl[0] * hwl[1] * hwl[2] < min_tmplt_vol * shrink_ratio_vol) {
      memcpy(hwl, tmplt_with_min_vol, sizeof(float) * 3); //限制检测出来物体最小尺寸
    } else {
      float hwl_tmplt[3] = {hwl[0], hwl[1], hwl[2]};
      int tmplt_index = -1; //得分最高的模板尺寸　对应与veh_hwl_的索引
      float score = object_template_manager_->VehObjHwlBySearchTemplates(
          hwl_tmplt, &tmplt_index);　//通过匹配检测出来车辆的尺寸(包括通过翻转)与模板尺寸，将得到分数最高的模板尺寸赋值于hwl_tmplt
      float thres_min_score = shrink_ratio_vol;

      const int kNrDimPerTmplt = object_template_manager_->NrDimPerTmplt();
      bool search_success = score > thres_min_score;  
      bool is_same_type = (type_min_vol_index / kNrDimPerTmplt) == tmplt_index;//判断得分最高的模板对应的物体类型是否与检测到的物体类型相同
      const std::map<TemplateIndex, int> &kLookUpTableMinVolumeIndex =
          object_template_manager_->LookUpTableMinVolumeIndex(); //返回这几种车辆在veh_hwl_中的起始位置
      bool is_car_pred =
          type_min_vol_index ==
          kLookUpTableMinVolumeIndex.at(TemplateIndex::CAR_MIN_VOLUME_INDEX); //是不是car

      bool hwl_is_reliable = search_success && is_same_type;　//若是同一类型，同时最高匹配得分高于阈值　则认为结果是可依赖的
      if (hwl_is_reliable) {
        memcpy(hwl, hwl_tmplt, sizeof(float) * 3); //直接将模板中的车辆尺寸赋值给hwl，这意味着神经网络的检测结果仅仅是用于判断和查表的?
      } else if (is_car_pred) {                    //若结果不可依赖，但是预测模型是car，获取中等尺寸的模板，将此值作为最终car的hwl
        const float *tmplt_with_median_vol =
            tmplt_with_min_vol + kNrDimPerTmplt;
        memcpy(hwl, tmplt_with_median_vol, sizeof(float) * 3);
      }
    }
  }

  // call 3d solver
  bool success =
      Solve3dBboxGivenOneFullBboxDimensionOrientation(bbox, hwl, ry, center); //经过计算以及一些后处理精细化补偿得到最终满足要求的hwl,ry,center

  // calculate variance for yaw & z
  float yaw_score_mean =
      common::IMean(ry_score_.data(), static_cast<int>(ry_score_.size())); //分数均值
  float yaw_score_sdv = common::ISdv(ry_score_.data(), yaw_score_mean,
                                     static_cast<int>(ry_score_.size()));  //分数的标准差
  var_yaw = common::ISqrt(common::IRec(yaw_score_sdv + params_.eps_mapper)); //　１/标准差　数据越平均　该值越大

  float z = center[2];
  float rz = z * params_.rz_ratio; // 0.1*z为z的变化率
  float nr_bins_z = static_cast<float>(params_.nr_bins_z);//15.0
  std::vector<float> buffer(static_cast<size_t>(2 * nr_bins_z), 0);
  float *score_z = buffer.data();
  float dz = 2 * rz / nr_bins_z;　//z步进（２*rz/15）
  float z_start = std::max(z - rz, params_.depth_min);
  float z_end = z + rz;
  int count_z_test = 0;
  for (float z_test = z_start; z_test <= z_end; z_test += dz) {
    float center_test[3] = {center[0], center[1], center[2]};
    float sf = z_test * common::IRec(center_test[2]);//测试深度z_test与原深度z的比值
    common::IScale3(center_test, sf);//将center_test 中的ｘ,y按照z等比例缩放.center_test[2]=z_test
    float score_test = GetProjectionScore(*ry, bbox, hwl, center_test);//同样计算计算交并比分数 最终得到不同中心的位置的分数
    score_z[count_z_test++] = score_test;
  }
  float z_score_mean = common::IMean(score_z, count_z_test); //得到均值
  float z_score_sdv = common::ISdv(score_z, z_score_mean, count_z_test);　//标准差
  var_z = common::ISqr(common::IRec(z_score_sdv + params_.eps_mapper));

  // fill the position_uncertainty_ and orientation_variance_
  orientation_variance_(0) = var_yaw;
  float bbox_cx = (bbox[0] + bbox[2]) / 2;
  float focal = (k_mat_[0] + k_mat_[4]) / 2; //焦距
  float sf_z_to_x = fabsf(bbox_cx - k_mat_[2]) * common::IRec(focal); //得2d bbox的中心坐标反投影 估计X的不确定性
  float var_x = var_z * common::ISqr(sf_z_to_x);//?????
  float var_xz = sf_z_to_x * var_z;//z估计ｘ的不确定性
  position_uncertainty_(0, 0) = var_x;
  position_uncertainty_(2, 2) = var_z;
  position_uncertainty_(0, 2) = position_uncertainty_(2, 0) = var_xz;
  return success;
}

void ObjMapper::PostRefineOrientation(const float *bbox, const float *hwl,
                                      const float *center, float *ry) {
  const int kNrBinsRy = static_cast<int>(ry_score_.size()); //36
  const float PI = common::Constant<float>::PI();
  const float PI_HALF = PI * 0.5f;
  const float D_RY = 2 * PI / static_cast<float>(kNrBinsRy); //2*pi/36 分为36份 每份10度

  float ry_test = -PI;
  float ry_best = -PI;
  float score_best = 0.0f;
  float score_cur = GetProjectionScore(*ry, bbox, hwl, center, true); //获取3d角点投影的2d bbox与检测得到的bbox的交并比
  int count_bin = 0;
  while (ry_test < PI - params_.eps_mapper) { //1e-5
    if (CalAngleDiff(ry_test, *ry) > PI_HALF) { 
      ry_test += D_RY;
      ry_score_[count_bin++ % kNrBinsRy] = 0.0f;
      continue;
    }//只计算ry_test与检测出的角度ry之差小于pi/2的角度范围

    float score_test = GetProjectionScore(ry_test, bbox, hwl, center, true); //由ry_test投影得到的２d bbox 与检测bbox的交并比
    if (score_test > score_best) {
      score_best = score_test;
      ry_best = ry_test;
    }　//匹配分数最高的角度
    ry_test += D_RY;
    ry_score_[count_bin++ % kNrBinsRy] = score_test;//记录这36分隔的角度对应匹配的分数
  }
  common::IUnitize(ry_score_.data(), kNrBinsRy); //将这３６个分数单位化
  if (score_best > params_.iou_high && score_cur < params_.iou_suc) {
    *ry = ry_best;
  }//iou_high 0.7  iou_suc =0.5 　若检测得到的角度的分数小于一定阈值　对角度进行修正

  float bbox_res[4] = {0};
  float score_final =
      GetProjectionScore(*ry, bbox, hwl, center, true, bbox_res);//根据最终角度　得到的分数
  if (score_final > params_.iou_high) {
    return;
  } else if (bbox[2] - bbox[0] <
             (bbox_res[2] - bbox_res[0]) * params_.factor_small) { //检测bbox的宽比投影得到的bbox的宽*0.6小
    *ry = *ry > 0 ? PI_HALF : -PI_HALF;
    FillRyScoreSingleBin(*ry); //将ry 对应与ry_score_的索引的分数置为１
  } else if ((bbox[2] - bbox[0]) * params_.factor_small >
             bbox_res[2] - bbox_res[0]) {
    *ry = 0.0f;
    FillRyScoreSingleBin(*ry);
  }
}

void ObjMapper::GetCenter(const float *bbox, const float &z_ref,
                          const float &ry, const float *hwl, float *center,
                          float *x) const {
  float x_target[2] = {(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2};//检测得到的2d框中心
  const float kMinCost = params_.reproj_err;//4*sqrt(2)
  const float EPS_COST_DELTA = params_.eps_mapper;//1e-5
  const float LR = params_.learning_r;//0.7
  const int MAX_ITERATION = params_.max_nr_iter;//10

  float cost_pre = 2.0f * static_cast<float>(width_);
  float cost_delta = 0.0f;
  float center_test[3] = {0};
  float rot[9] = {0};
  GenRotMatrix(ry, rot);//生成旋转矩阵
  int iter = 1;
  bool stop = false;
  float h = hwl[0];
  float w = hwl[1];
  float l = hwl[2];
  float x_corners[8] = {0};
  float y_corners[8] = {0};
  float z_corners[8] = {0};
  GenCorners(h, w, l, x_corners, y_corners, z_corners);//生成各个顶点相对3d box中心坐标

  float x_max_flt = static_cast<float>(width_ - 1);
  float y_max_flt = static_cast<float>(height_ - 1);

  // std::cout << "start to iteratively search the center..." << std::endl;
  while (!stop) {
    common::IBackprojectCanonical(x, k_mat_, z_ref, center_test);　//x :2d框中心根据深度信息反投影到　3d位置center_test
    center_test[1] += hwl[0] / 2;
    float x_min = FLT_MAX;
    float x_max = -FLT_MAX;
    float y_min = FLT_MAX;
    float y_max = -FLT_MAX;
    float x_proj[3] = {0};

    for (int i = 0; i < 8; ++i) {
      // bbox from x_proj
      float x_box[3] = {x_corners[i], y_corners[i], z_corners[i]}; 
      common::IProjectThroughKRt(k_mat_, rot, center_test, x_box, x_proj);//将车辆坐标系下的3d角点坐标x_box投影到图像中得到x_proj
      x_proj[0] *= common::IRec(x_proj[2]);
      x_proj[1] *= common::IRec(x_proj[2]);　//转为齐次坐标
      x_min = std::min(x_min, x_proj[0]);　　//迭代获取８个角点投影回来的二维坐标的最小值和最大值
      x_max = std::max(x_max, x_proj[0]);
      y_min = std::min(y_min, x_proj[1]);
      y_max = std::max(y_max, x_proj[1]);

      // truncation processing
      x_min = std::min(std::max(x_min, 0.0f), x_max_flt);　//截断处理
      x_max = std::min(std::max(x_max, 0.0f), x_max_flt);
      y_min = std::min(std::max(y_min, 0.0f), y_max_flt);
      y_max = std::min(std::max(y_max, 0.0f), y_max_flt);
    }
    float x_cur[2] = {(x_min + x_max) / 2, (y_min + y_max) / 2};　//获取此次迭代　3dbbox角点投影到图像所得的　中心点
    float cost_cur = common::ISqrt(common::ISqr(x_cur[0] - x_target[0]) +
                                   common::ISqr(x_cur[1] - x_target[1]));//计算欧氏距离

    if (cost_cur >= cost_pre) {　//cost_pre 是上一次迭代的cost 当cost不在下降时
      stop = true;
    } else {
      memcpy(center, center_test, sizeof(float) * 3); //center_test 是２d检测框根据深度反投影得到的3d box中心 ，由此中心计算得到　3d box在图像下的2d框　，由此2d框与实际检测的2d框计算欧式距离
      cost_delta = (cost_pre - cost_cur) / cost_pre;　//相对变化
      cost_pre = cost_cur;
      x[0] += (x_target[0] - x_cur[0]) * LR; //不断调整　2d框的中心
      x[1] += (x_target[1] - x_cur[1]) * LR;
      ++iter;
      stop = iter >= MAX_ITERATION || cost_delta < EPS_COST_DELTA ||
             cost_pre < kMinCost;
    } 
  }
}

}  // namespace camera
}  // namespace perception
}  // namespace apollo
