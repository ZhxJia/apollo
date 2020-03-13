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
#pragma once

namespace apollo {
namespace perception {
namespace radar {

const double PI = 3.1415926535898;
const int MAX_RADAR_IDX = 2147483647;
const double CONTI_ARS_INTERVAL = 0.074;
const int ORIGIN_CONTI_MAX_ID_NUM = 100;
const double MIN_PROBEXIST = 0.08;

enum ContiObjectType {
  CONTI_POINT = 0,
  CONTI_CAR = 1,
  CONTI_TRUCK = 2,
  CONTI_PEDESTRIAN = 3,
  CONTI_MOTOCYCLE = 4,
  CONTI_BICYCLE = 5,
  CONTI_WIDE = 6,
  CONTI_TYPE_UNKNOWN = 7,
  CONTI_MAX_OBJECT_TYPE = 8
};

enum ContiMeasState {
  CONTI_DELETED = 0, //已删除object:在object id消失之前的最后一次传输周期中显示
  CONTI_NEW = 1,     //创建新的object:在创建此object id后的第一次传输周期中显示
  CONTI_MEASURED = 2, //测量，object的创建已通过实际测量确认
  CONTI_PREDICTED = 3, //预测，实际测量无法确认object创建
  CONTI_DELETED_FOR = 4,  //融合删除 与另一个object融合，再删除此object
  CONTI_NEW_FROM_MERGE = 5 //合并出新，与别的object融合产生此object
};

enum ContiDynProp {
  CONTI_MOVING = 0,
  CONTI_STATIONARY = 1,
  CONTI_ONCOMING = 2,
  CONTI_STATIONARY_CANDIDATE = 3,
  CONTI_DYNAMIC_UNKNOWN = 4,
  CONTI_CROSSING_STATIONARY = 5,
  CONTI_CROSSING_MOVING = 6,
  CONTI_STOPPED = 7
};

}  // namespace radar
}  // namespace perception
}  // namespace apollo
