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
#include "modules/perception/fusion/lib/data_association/hm_data_association/hm_tracks_objects_match.h"

#include <map>
#include <utility>

#include "modules/perception/common/graph/secure_matrix.h"

namespace apollo {
namespace perception {
namespace fusion {

double HMTrackersObjectsAssociation::s_match_distance_thresh_ = 4.0;
double HMTrackersObjectsAssociation::s_match_distance_bound_ = 100.0;
/* this is a slack threshold for camera 2 lidar/radar association.
 * consider the ave 2d-to-3d error is 7%, 30m is 15% of 200m, which
 * is 2 times of ave error around 200m. */
double HMTrackersObjectsAssociation::s_association_center_dist_threshold_ =
    30.0;
//@brief:从vec中提取subset_inds对应的值形成新的sub_vec
template <typename T>
void extract_vector(const std::vector<T>& vec,
                    const std::vector<size_t>& subset_inds,
                    std::vector<T>* sub_vec) {
  sub_vec->reserve(subset_inds.size());
  sub_vec->clear();
  for (auto subset_ind : subset_inds) {
    sub_vec->push_back(vec[subset_ind]);
  }
}

bool HMTrackersObjectsAssociation::Associate(
    const AssociationOptions& options, SensorFramePtr sensor_measurements,
    ScenePtr scene, AssociationResult* association_result) {
  const std::vector<SensorObjectPtr>& sensor_objects =
      sensor_measurements->GetForegroundObjects();
  const std::vector<TrackPtr>& fusion_tracks = scene->GetForegroundTracks(); //初次运行为空
  std::vector<std::vector<double>> association_mat;

  if (fusion_tracks.empty() || sensor_objects.empty()) {
    association_result->unassigned_tracks.resize(fusion_tracks.size());
    association_result->unassigned_measurements.resize(sensor_objects.size());
    std::iota(association_result->unassigned_tracks.begin(),
              association_result->unassigned_tracks.end(), 0);
    std::iota(association_result->unassigned_measurements.begin(),
              association_result->unassigned_measurements.end(), 0);
    return true;
  }
  std::string measurement_sensor_id = sensor_objects[0]->GetSensorId(); //获取该物体对应的传感器
  double measurement_timestamp = sensor_objects[0]->GetTimestamp(); //获取该物体被检测到的时间
  track_object_distance_.ResetProjectionCache(measurement_sensor_id,
                                              measurement_timestamp); //以sensor_id,timestamp为key,重置ProjectionCache
  bool do_nothing = (sensor_objects[0]->GetSensorId() == "radar_front");
  IdAssign(fusion_tracks, sensor_objects, &association_result->assignments,
           &association_result->unassigned_tracks,
           &association_result->unassigned_measurements, do_nothing, false); //相同sensor_id的匹配按照track_id相同即可进行匹配

  Eigen::Affine3d pose;
  sensor_measurements->GetPose(&pose); //sensor2world_pose
  Eigen::Vector3d ref_point = pose.translation(); //sensor2world 位移

  ADEBUG << "association_measurement_timestamp@" << measurement_timestamp;
  ComputeAssociationDistanceMat(fusion_tracks, sensor_objects, ref_point,
                                association_result->unassigned_tracks,
                                association_result->unassigned_measurements,
                                &association_mat); //不同sensor_id的匹配就需要通过特定的度量方式

  int num_track = static_cast<int>(fusion_tracks.size());
  int num_measurement = static_cast<int>(sensor_objects.size());
  association_result->track2measurements_dist.assign(num_track, 0);
  association_result->measurement2track_dist.assign(num_measurement, 0);
  std::vector<int> track_ind_g2l;
  track_ind_g2l.resize(num_track, -1);//填充值和索引
  for (size_t i = 0; i < association_result->unassigned_tracks.size(); i++) {
    track_ind_g2l[association_result->unassigned_tracks[i]] =
        static_cast<int>(i);//将track_ind_g2l中对应unassigned_tracks的id索引 从0开始设置 ，其余已经匹配的设置为-1
  }
  std::vector<int> measurement_ind_g2l; //global to local
  measurement_ind_g2l.resize(num_measurement, -1);
  std::vector<size_t> measurement_ind_l2g =
      association_result->unassigned_measurements;
  for (size_t i = 0; i < association_result->unassigned_measurements.size();
       i++) {
    measurement_ind_g2l[association_result->unassigned_measurements[i]] =
        static_cast<int>(i);//将measurement_ind_g2l中对应unassigned_measurements的id索引 从0开始设置 ，其余已经匹配的设置为-1
  }
  std::vector<size_t> track_ind_l2g = association_result->unassigned_tracks;

  if (association_result->unassigned_tracks.empty() ||
      association_result->unassigned_measurements.empty()) {
    return true;
  }

  bool state = MinimizeAssignment(
      association_mat, track_ind_l2g, measurement_ind_l2g,
      &association_result->assignments, &association_result->unassigned_tracks,
      &association_result->unassigned_measurements); //分配

  // start do post assign
  std::vector<TrackMeasurmentPair> post_assignments;
  PostIdAssign(fusion_tracks, sensor_objects,
               association_result->unassigned_tracks,
               association_result->unassigned_measurements, &post_assignments);
  association_result->assignments.insert(association_result->assignments.end(),
                                         post_assignments.begin(),
                                         post_assignments.end());

  GenerateUnassignedData(fusion_tracks.size(), sensor_objects.size(),
                         association_result->assignments,
                         &association_result->unassigned_tracks,
                         &association_result->unassigned_measurements);

  ComputeDistance(fusion_tracks, sensor_objects,
                  association_result->unassigned_tracks, track_ind_g2l,
                  measurement_ind_g2l, measurement_ind_l2g, association_mat,
                  association_result);

  AINFO << "association: measurement_num = " << sensor_objects.size()
        << ", track_num = " << fusion_tracks.size()
        << ", assignments = " << association_result->assignments.size()
        << ", unassigned_tracks = "
        << association_result->unassigned_tracks.size()
        << ", unassigned_measuremnets = "
        << association_result->unassigned_measurements.size();

  return state;
}
void HMTrackersObjectsAssociation::PostIdAssign(
    const std::vector<TrackPtr>& fusion_tracks,
    const std::vector<SensorObjectPtr>& sensor_objects,
    const std::vector<size_t>& unassigned_fusion_tracks,
    const std::vector<size_t>& unassigned_sensor_objects,
    std::vector<TrackMeasurmentPair>* post_assignments) {
  std::vector<size_t> valid_unassigned_tracks;
  valid_unassigned_tracks.reserve(unassigned_fusion_tracks.size());
  // only camera track
  auto is_valid_track = [](const TrackPtr& fusion_track) {
    SensorObjectConstPtr camera_obj = fusion_track->GetLatestCameraObject();
    return camera_obj != nullptr &&
           fusion_track->GetLatestLidarObject() == nullptr;
    // && fusion_track->GetLatestRadarObject() == nullptr;
  }; //判断是否是有效的未分配track,即该track是否有对应的camera_object
  for (auto unassigned_track_id : unassigned_fusion_tracks) {
    if (is_valid_track(fusion_tracks[unassigned_track_id])) {
      valid_unassigned_tracks.push_back(unassigned_track_id);
    }
  }
  std::vector<TrackPtr> sub_tracks;
  std::vector<SensorObjectPtr> sub_objects;
  extract_vector(fusion_tracks, valid_unassigned_tracks, &sub_tracks); //根据未分配的有效的track id从原fusion_tracks中提取得到sub_tracks
  extract_vector(sensor_objects, unassigned_sensor_objects, &sub_objects);
  std::vector<size_t> tmp1, tmp2;
  IdAssign(sub_tracks, sub_objects, post_assignments, &tmp1, &tmp2, false,
           true);
  for (auto& post_assignment : *post_assignments) {
    post_assignment.first = valid_unassigned_tracks[post_assignment.first];
    post_assignment.second = unassigned_sensor_objects[post_assignment.second];
  }
}

bool HMTrackersObjectsAssociation::MinimizeAssignment(
    const std::vector<std::vector<double>>& association_mat,
    const std::vector<size_t>& track_ind_l2g,
    const std::vector<size_t>& measurement_ind_l2g,
    std::vector<TrackMeasurmentPair>* assignments,
    std::vector<size_t>* unassigned_tracks,
    std::vector<size_t>* unassigned_measurements) {
  common::GatedHungarianMatcher<float>::OptimizeFlag opt_flag =
      common::GatedHungarianMatcher<float>::OptimizeFlag::OPTMIN; //最小化距离
  common::SecureMat<float>* global_costs = optimizer_.mutable_global_costs(); //GatedHungarianMatcher
  int rows = static_cast<int>(unassigned_tracks->size());
  int cols = static_cast<int>(unassigned_measurements->size());

  global_costs->Resize(rows, cols);
  for (int r_i = 0; r_i < rows; r_i++) {
    for (int c_i = 0; c_i < cols; c_i++) {
      (*global_costs)(r_i, c_i) = static_cast<float>(association_mat[r_i][c_i]); //(unassigned_tracks,unassigned_measurements)
    }
  } //将optimizer的global_cost赋值为计算的距离矩阵
  std::vector<TrackMeasurmentPair> local_assignments;
  std::vector<size_t> local_unassigned_tracks;
  std::vector<size_t> local_unassigned_measurements;
  optimizer_.Match(static_cast<float>(s_match_distance_thresh_), // thresh 4.0
                   static_cast<float>(s_match_distance_bound_), opt_flag, //bound 100.0
                   &local_assignments, &local_unassigned_tracks,
                   &local_unassigned_measurements);
  for (auto assign : local_assignments) {
    assignments->push_back(std::make_pair(track_ind_l2g[assign.first],
                                          measurement_ind_l2g[assign.second])); //将从0开始的id转换为全局的measurement_id或track_id
  }
  unassigned_tracks->clear();
  unassigned_measurements->clear();
  for (auto un_track : local_unassigned_tracks) {
    unassigned_tracks->push_back(track_ind_l2g[un_track]);
  }
  for (auto un_mea : local_unassigned_measurements) {
    unassigned_measurements->push_back(measurement_ind_l2g[un_mea]);
  }
  return true;
}

void HMTrackersObjectsAssociation::ComputeDistance(
    const std::vector<TrackPtr>& fusion_tracks,
    const std::vector<SensorObjectPtr>& sensor_objects,
    const std::vector<size_t>& unassigned_fusion_tracks,
    const std::vector<int>& track_ind_g2l,
    const std::vector<int>& measurement_ind_g2l,
    const std::vector<size_t>& measurement_ind_l2g,
    const std::vector<std::vector<double>>& association_mat,
    AssociationResult* association_result) {
  for (size_t i = 0; i < association_result->assignments.size(); i++) {
    int track_ind = static_cast<int>(association_result->assignments[i].first);
    int measurement_ind =
        static_cast<int>(association_result->assignments[i].second);
    int track_ind_loc = track_ind_g2l[track_ind]; 
    int measurement_ind_loc = measurement_ind_g2l[measurement_ind];
    if (track_ind_loc >= 0 && measurement_ind_loc >= 0) {
      association_result->track2measurements_dist[track_ind] =
          association_mat[track_ind_loc][measurement_ind_loc];
      association_result->measurement2track_dist[measurement_ind] =
          association_mat[track_ind_loc][measurement_ind_loc];
    }
  }
  for (size_t i = 0; i < association_result->unassigned_tracks.size(); i++) {
    int track_ind = static_cast<int>(unassigned_fusion_tracks[i]);
    int track_ind_loc = track_ind_g2l[track_ind];
    association_result->track2measurements_dist[track_ind] =
        association_mat[track_ind_loc][0];
    int min_m_loc = 0;
    for (size_t j = 1; j < association_mat[track_ind_loc].size(); j++) {
      if (association_result->track2measurements_dist[track_ind] >
          association_mat[track_ind_loc][j]) {
        association_result->track2measurements_dist[track_ind] =
            association_mat[track_ind_loc][j];
        min_m_loc = static_cast<int>(j);
      } //找距离的最小值,记录对应的局部索引
    }
    int min_m_ind = static_cast<int>(measurement_ind_l2g[min_m_loc]);
    const SensorObjectPtr& min_sensor_object = sensor_objects[min_m_ind]; //找到该最短距离对应的sensor_object
    const TrackPtr& fusion_track = fusion_tracks[track_ind];
    SensorObjectConstPtr lidar_object = fusion_track->GetLatestLidarObject();
    SensorObjectConstPtr radar_object = fusion_track->GetLatestRadarObject();
    if (IsCamera(min_sensor_object)) { //如果该测量值是相机检测得到的
      // TODO(linjian) not reasonable,
      // just for return dist score, the dist score is
      // a similarity probability [0, 1] 1 is the best
      association_result->track2measurements_dist[track_ind] = 0.0;
      for (size_t j = 0; j < association_mat[track_ind_loc].size(); ++j) {
        double dist_score = 0.0;
        if (lidar_object != nullptr) {
          dist_score = track_object_distance_.ComputeLidarCameraSimilarity(
              lidar_object, sensor_objects[measurement_ind_l2g[j]],
              IsLidar(sensor_objects[measurement_ind_l2g[j]]));
        }
        if (radar_object != nullptr) {
          dist_score = track_object_distance_.ComputeRadarCameraSimilarity(
              radar_object, sensor_objects[measurement_ind_l2g[j]]);
        }
        association_result->track2measurements_dist[track_ind] = std::max(
            association_result->track2measurements_dist[track_ind], dist_score);
      }
    }
  }
  for (size_t i = 0; i < association_result->unassigned_measurements.size();
       i++) {
    int m_ind =
        static_cast<int>(association_result->unassigned_measurements[i]);
    int m_ind_loc = measurement_ind_g2l[m_ind];
    association_result->measurement2track_dist[m_ind] =
        association_mat[0][m_ind_loc];
    for (size_t j = 1; j < association_mat.size(); j++) {
      if (association_result->measurement2track_dist[m_ind] >
          association_mat[j][m_ind_loc]) {
        association_result->measurement2track_dist[m_ind] =
            association_mat[j][m_ind_loc];
      }
    }
  }
}

void HMTrackersObjectsAssociation::ComputeAssociationDistanceMat(
    const std::vector<TrackPtr>& fusion_tracks,
    const std::vector<SensorObjectPtr>& sensor_objects,
    const Eigen::Vector3d& ref_point,
    const std::vector<size_t>& unassigned_tracks,
    const std::vector<size_t>& unassigned_measurements,
    std::vector<std::vector<double>>* association_mat) {
  // if (sensor_objects.empty()) return;
  TrackObjectDistanceOptions opt;
  // TODO(linjian) ref_point
  Eigen::Vector3d tmp = Eigen::Vector3d::Zero();
  opt.ref_point = &tmp;
  association_mat->resize(unassigned_tracks.size());
  for (size_t i = 0; i < unassigned_tracks.size(); ++i) {
    int fusion_idx = static_cast<int>(unassigned_tracks[i]); //未被分配的tracks的索引(从0开始的,表示第几个)
    (*association_mat)[i].resize(unassigned_measurements.size());
    const TrackPtr& fusion_track = fusion_tracks[fusion_idx]; //unassigned fusion track
    for (size_t j = 0; j < unassigned_measurements.size(); ++j) {
      int sensor_idx = static_cast<int>(unassigned_measurements[j]);
      const SensorObjectPtr& sensor_object = sensor_objects[sensor_idx]; //unassinged sensor object
      double distance = s_match_distance_thresh_; //4.0
      double center_dist =
          (sensor_object->GetBaseObject()->center -
           fusion_track->GetFusedObject()->GetBaseObject()->center)
              .norm(); //中心点的距离
      if (center_dist < s_association_center_dist_threshold_) { //30m
        distance =
            track_object_distance_.Compute(fusion_track, sensor_object, opt);//计算sensor_object与fusion_track中各个传感器检测目标距离度量的最小值
      } else {
        ADEBUG << "center_distance " << center_dist
               << " exceeds slack threshold "
               << s_association_center_dist_threshold_
               << ", track_id: " << fusion_track->GetTrackId()
               << ", obs_id: " << sensor_object->GetBaseObject()->track_id;
      }
      (*association_mat)[i][j] = distance;
      ADEBUG << "track_id: " << fusion_track->GetTrackId()
             << ", obs_id: " << sensor_object->GetBaseObject()->track_id
             << ", distance: " << distance;
    }
  }
}
//@brief: track id 分配 
void HMTrackersObjectsAssociation::IdAssign(
    const std::vector<TrackPtr>& fusion_tracks,
    const std::vector<SensorObjectPtr>& sensor_objects,
    std::vector<TrackMeasurmentPair>* assignments,
    std::vector<size_t>* unassigned_fusion_tracks,
    std::vector<size_t>* unassigned_sensor_objects, bool do_nothing,
    bool post) {
  size_t num_track = fusion_tracks.size();
  size_t num_obj = sensor_objects.size();
  if (num_track == 0 || num_obj == 0 || do_nothing) {
    unassigned_fusion_tracks->resize(num_track);
    unassigned_sensor_objects->resize(num_obj);
    std::iota(unassigned_fusion_tracks->begin(),
              unassigned_fusion_tracks->end(), 0);
    std::iota(unassigned_sensor_objects->begin(),
              unassigned_sensor_objects->end(), 0);
    return;
  }
  const std::string sensor_id = sensor_objects[0]->GetSensorId(); //对应该object的检测传感器

  std::map<int, int> sensor_id_2_track_ind;
  for (size_t i = 0; i < num_track; i++) {
    SensorObjectConstPtr obj = fusion_tracks[i]->GetSensorObject(sensor_id); //相同传感器之间的track_id匹配按照对应的id相同即可匹配
    /* when camera system has sub-fusion of obstacle & narrow, they share
     * the same track-id sequence. thus, latest camera object is ok for
     * camera id assign and its information is more up to date. */
    if (sensor_id == "front_6mm" || sensor_id == "front_12mm") {
      obj = fusion_tracks[i]->GetLatestCameraObject(); //由于相机组件fusioncameracomponent已经将12mm和6mm的跟踪进行了融合它们共享track id,直接对它们的总体分配id即可
    }
    if (obj == nullptr) {
      continue;
    }
    sensor_id_2_track_ind[obj->GetBaseObject()->track_id] = static_cast<int>(i); //pair(fusion_track_id,num_track_id) num_track_id为对应帧的排序从0开始
  }  
  std::vector<bool> fusion_used(num_track, false);
  std::vector<bool> sensor_used(num_obj, false);
  for (size_t i = 0; i < num_obj; i++) {
    int track_id = sensor_objects[i]->GetBaseObject()->track_id; //当前检测物体对应的track_id 
    auto it = sensor_id_2_track_ind.find(track_id); //在fusion_tracks中查找有没有对应的track_id

    // In id_assign, we don't assign the narrow camera object
    // with the track which only have narrow camera object
    // In post id_assign, we do this.
    if (!post && (sensor_id == "front_6mm" || sensor_id == "front_12mm")) //post:False //若当前检测传感器是相机且post=False则 跳过
      continue; //即当post=False时，不进行相机检测目标的id分配

    if (it != sensor_id_2_track_ind.end()) {
      sensor_used[i] = true;
      fusion_used[it->second] = true;
      assignments->push_back(std::make_pair(it->second, i)); //pair(fusion_tracks_id,sensor_objects_id)
    }
  }
  for (size_t i = 0; i < fusion_used.size(); ++i) {
    if (!fusion_used[i]) {
      unassigned_fusion_tracks->push_back(i); //未被分配的fusion_tracks
    }
  }
  for (size_t i = 0; i < sensor_used.size(); ++i) {
    if (!sensor_used[i]) {
      unassigned_sensor_objects->push_back(i); //未被分配的sensor_objects
    }
  }
}

void HMTrackersObjectsAssociation::GenerateUnassignedData(
    size_t track_num, size_t objects_num,
    const std::vector<TrackMeasurmentPair>& assignments,
    std::vector<size_t>* unassigned_tracks,
    std::vector<size_t>* unassigned_objects) {
  std::vector<bool> track_flags(track_num, false);
  std::vector<bool> objects_flags(objects_num, false);
  for (auto assignment : assignments) {
    track_flags[assignment.first] = true;
    objects_flags[assignment.second] = true;
  }
  unassigned_tracks->clear(), unassigned_tracks->reserve(track_num);
  unassigned_objects->clear(), unassigned_objects->reserve(objects_num);
  for (size_t i = 0; i < track_num; ++i) {
    if (!track_flags[i]) {
      unassigned_tracks->push_back(i);
    }
  }
  for (size_t i = 0; i < objects_num; ++i) {
    if (!objects_flags[i]) {
      unassigned_objects->push_back(i);
    }
  }
}

}  // namespace fusion
}  // namespace perception
}  // namespace apollo
