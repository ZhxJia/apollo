/******************************************************************************
 * Copyright 2017 The Apollo Authors. All Rights Reserved.
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

#include "modules/drivers/camera/camera_component.h"

namespace apollo {
namespace drivers {
namespace camera {

bool CameraComponent::Init() {
  camera_config_ = std::make_shared<Config>();
  if (!apollo::cyber::common::GetProtoFromFile(config_file_path_,
                                               camera_config_.get())) { //由dag文件中的pb.txt初始化config.proto中的参数
    return false;
  }
  AINFO << "UsbCam config: " << camera_config_->DebugString();

  camera_device_.reset(new UsbCam());
  camera_device_->init(camera_config_);
  raw_image_.reset(new CameraImage);

  raw_image_->width = camera_config_->width();
  raw_image_->height = camera_config_->height();
  raw_image_->bytes_per_pixel = camera_config_->bytes_per_pixel();

  device_wait_ = camera_config_->device_wait_ms();
  spin_rate_ = static_cast<uint32_t>((1.0 / camera_config_->spin_rate()) * 1e6); //1/200*1e6 设备之间的轮转时间

  if (camera_config_->output_type() == YUYV) {
    raw_image_->image_size = raw_image_->width * raw_image_->height * 2; //一个像素两个字节
  } else if (camera_config_->output_type() == RGB) {
    raw_image_->image_size = raw_image_->width * raw_image_->height * 3; //一个像素三个字节 bingo
  }
  if (raw_image_->image_size > MAX_IMAGE_SIZE) {
    AERROR << "image size is too big ,must less than " << MAX_IMAGE_SIZE
           << " bytes.";
    return false;
  }
  raw_image_->is_new = 0;
  // free memory in this struct desturctor
  raw_image_->image =
      reinterpret_cast<char*>(calloc(raw_image_->image_size, sizeof(char))); //在内存动态分配区内分配长度为imagesize的连续空间
  if (raw_image_->image == nullptr) {
    AERROR << "system calloc memory error, size:" << raw_image_->image_size;
    return false;
  }

  for (int i = 0; i < buffer_size_; ++i) { //16
    auto pb_image = std::make_shared<Image>();
    pb_image->mutable_header()->set_frame_id(camera_config_->frame_id()); //eg. camera_front_6mm
    pb_image->set_width(raw_image_->width);
    pb_image->set_height(raw_image_->height);
    pb_image->mutable_data()->reserve(raw_image_->image_size); //proto中的bytes对应实际存储为std::string

    if (camera_config_->output_type() == YUYV) {
      pb_image->set_encoding("yuyv");
      pb_image->set_step(2 * raw_image_->width); //每一行像素所占的空间
    } else if (camera_config_->output_type() == RGB) {
      pb_image->set_encoding("rgb8"); //bingo
      pb_image->set_step(3 * raw_image_->width);
    }

    pb_image_buffer_.push_back(pb_image);
  } //16

  writer_ = node_->CreateWriter<Image>(camera_config_->channel_name());
  async_result_ = cyber::Async(&CameraComponent::run, this);
  return true;
}

void CameraComponent::run() {
  running_.exchange(true);
  while (!cyber::IsShutdown()) {
    if (!camera_device_->wait_for_device()) {
      // sleep for next check
      cyber::SleepFor(std::chrono::milliseconds(device_wait_)); //device_wait_ms: 2000
      continue;
    }

    if (!camera_device_->poll(raw_image_)) {
      AERROR << "camera device poll failed"; //轮询失败
      continue;
    }

    cyber::Time image_time(raw_image_->tv_sec, 1000 * raw_image_->tv_usec); //秒+毫秒 -> 测量时间
    if (index_ >= buffer_size_) {
      index_ = 0;
    }
    auto pb_image = pb_image_buffer_.at(index_++);//循环读取指针，然后向指针中赋值，注意pb_image_buffer_中存储的是指针。
    pb_image->mutable_header()->set_timestamp_sec(
        cyber::Time::Now().ToSecond()); //图像发布时间
    pb_image->set_measurement_time(image_time.ToSecond());
    pb_image->set_data(raw_image_->image, raw_image_->image_size); //将raw_image数据拷贝到pb_image中
    writer_->Write(pb_image);

    cyber::SleepFor(std::chrono::microseconds(spin_rate_)); // 1/200 *1e6 us = 1/200s 轮询
  }
}

CameraComponent::~CameraComponent() {
  if (running_.load()) {
    running_.exchange(false);
    async_result_.wait();
  }
}

}  // namespace camera
}  // namespace drivers
}  // namespace apollo
