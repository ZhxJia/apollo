/******************************************************************************
 * Copyright 2018 The Apollo Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/
#pragma once

#include <map>
#include <string>
#include <vector>

#include "modules/perception/fusion/common/dst_evidence.h"
#include "modules/perception/fusion/lib/interface/base_existance_fusion.h"

namespace apollo {
namespace perception {
namespace fusion {

struct ToicDstMaps {
  // for (N)TOIC: (not)target of interest in camera judgement
  enum { TOIC = (1 << 0), NTOIC = (1 << 1), TOICUNKOWN = (TOIC | NTOIC) };
  std::vector<uint64_t> fod_subsets_ = {TOIC, NTOIC, TOICUNKOWN};
  std::vector<std::string> subset_names_ = {"TOIC", "NTOIC", "TOICUNKOWN"};
};

struct ExistanceDstMaps {
  enum { EXIST = (1 << 0), NEXIST = (1 << 1), EXISTUNKOWN = (EXIST | NEXIST) }; //1,2,3
  std::vector<uint64_t> fod_subsets_ = {EXIST, NEXIST, EXISTUNKOWN};
  std::vector<std::string> subset_names_ = {"EXIST", "NEXIST", "EXISTUNKOWN"};
};

struct DstExistanceFusionOptions {
  std::map<std::string, double> camera_max_valid_dist_ = {
      {"camera_smartereye", 110},
      {"camera_front_obstacle", 110},
      {"camera_front_narrow", 150},
      {"front_6mm", 110},
  };
  double track_object_max_match_distance_ = 4.0;
};

class DstExistanceFusion : public BaseExistanceFusion {
 public:
  explicit DstExistanceFusion(TrackPtr track);
  ~DstExistanceFusion() {}

  // @brief: add dst application
  static bool Init();

  // @brief: update track state with measurement
  // @param [in]: measurement
  // @param [in]: target_timestamp
  void UpdateWithMeasurement(const SensorObjectPtr measurement,
                             double target_timestamp,
                             double match_dist) override;

  void UpdateWithoutMeasurement(const std::string &sensor_id,
                                double measurement_timestamp,
                                double target_timestamp,
                                double min_match_dist) override;

  std::string Name() const;
  double GetToicScore() const { return toic_score_; }
  double GetExistanceProbability() const;

 private:
  void UpdateToicWithCameraMeasurement(const SensorObjectPtr &camera_obj,
                                       double match_dist);
  void UpdateToicWithoutCameraMeasurement(const std::string &sensor_id,
                                          double measurement_timestamp,
                                          double match_dist);

  double ComputeDistDecay(base::ObjectConstPtr obj,
                          const std::string &sensor_id, double timestamp);
  double ComputeFeatureInfluence(const SensorObjectPtr measurement);
  double GetExistReliability(const SensorObjectPtr measurement);
  double GetUnexistReliability(const std::string &sensor_id);
  double GetToicProbability() const;

  // Update state
  void UpdateExistanceState();

 private:
  double existance_score_ = 0.0;
  Dst fused_toic_;
  Dst fused_existance_;
  double toic_score_ = 0.0;

 private:
  static const char *name_;
  static const char *toic_name_;
  static ExistanceDstMaps existance_dst_maps_;
  static ToicDstMaps toic_dst_maps_;
  static DstExistanceFusionOptions options_;
};

}  // namespace fusion
}  // namespace perception
}  // namespace apollo
