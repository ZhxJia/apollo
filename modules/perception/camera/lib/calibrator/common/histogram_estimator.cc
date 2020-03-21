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
#include "modules/perception/camera/lib/calibrator/common/histogram_estimator.h"

#include "modules/perception/common/i_lib/core/i_basic.h"

namespace apollo {
namespace perception {
namespace camera {

void HistogramEstimatorParams::Init() {  // set default value
  nr_bins_in_histogram = 64;
  data_sp = 0;
  data_ep = 255;
  assert(nr_bins_in_histogram > 0);
  assert(data_ep > data_sp);

  step_bin = (data_ep - data_sp) / static_cast<float>(nr_bins_in_histogram);

  smooth_kernel.clear();
  smooth_kernel.push_back(1);
  smooth_kernel.push_back(2);
  smooth_kernel.push_back(1);
  smooth_kernel_width = static_cast<int>(smooth_kernel.size());
  smooth_kernel_radius = smooth_kernel_width >> 1;

  hat_min_allowed = 0.2f;
  hat_std_allowed = 6.0f;
  histogram_mass_limit = 100;
  decay_factor = 0.9f;
}

void HistogramEstimator::Init(const HistogramEstimatorParams *params) {
  if (params != nullptr) {
    params_ = *params; //HistogramEstimatorParams 默认参数在构造函数中，此处由CalibratorParams中的参数覆盖
  }
  step_bin_reversed_ = common::IRec(params_.step_bin); //1/(20/400) = 20 表示每度对应的区间的索引位置
  assert(params_.nr_bins_in_histogram > 0);
  assert(params_.nr_bins_in_histogram <= kMaxNrBins);
  int nr_bins = params_.nr_bins_in_histogram; //400
  memset(hist_.data(), 0, sizeof(uint32_t) * nr_bins);
  GenerateHat(hist_hat_.data(), nr_bins); //out: hist_hat 先验直方图形状
}

bool HistogramEstimator::Process() {
  // smooth histogram - >
  // get peak & check mass ->
  // shape anlysis ->
  // update estimates - >
  // final decay

  uint32_t *hist = hist_.data();
  uint32_t *hist_smoothed = hist_buffer_.data();
  int nr_bins = params_.nr_bins_in_histogram;
  Smooth(hist, nr_bins, hist_smoothed);
#if 0
  SaveHist("input_hist", hist, nr_bins);
  SaveHist("smoothed_hist", hist_smoothed, nr_bins);
#endif
  int max_index = 0;
  uint32_t mass = 0;
  GetPeakIndexAndMass(hist_smoothed, nr_bins, &max_index, &mass);
  if (mass < params_.histogram_mass_limit) { //100
    AERROR << "Fail: lack enough samples.";
    return false;
  }

  if (!IsGoodShape(hist, nr_bins, max_index)) {
    AERROR << "Fail: distribution is not good.";
    return false;
  }

  val_estimation_ = GetValFromIndex(max_index);
  Decay(hist, nr_bins); //400

  return true;
}

void HistogramEstimator::Smooth(const uint32_t *hist_input, int nr_bins,
                                uint32_t *hist_output) {
  assert(nr_bins == params_.nr_bins_in_histogram);

  int filter_radius = params_.smooth_kernel_radius; //2 kernel size为5
  if (filter_radius * 2 > nr_bins) {
    return;
  }

  float weight_sum_kernel = 0.0f;
  for (auto &w : params_.smooth_kernel) {
    weight_sum_kernel += static_cast<float>(w);
  } //(1,3,5,3,1)-> 1/13
  float norm_reversed = common::IRec(weight_sum_kernel);
  int sp = 0;
  int ep = 0;

  // left boundary part (0,1) 从核的右半部分开始
  for (int i = 0; i < filter_radius; ++i) {
    uint32_t sum = 0;
    sp = 0;
    ep = i + filter_radius; //(2,3)
    float kernel_sum = 0.0f;
    for (int j = sp; j <= ep; ++j) {                                      
      sum += hist_input[j] * params_.smooth_kernel[j - i + filter_radius];//(0 - 0 + 2),(1-0+2),(2-0+2)
      kernel_sum += static_cast<float>(params_.smooth_kernel[j - sp]);
    }
    float scale = common::IRec(kernel_sum);
    hist_output[i] = static_cast<uint32_t>(static_cast<float>(sum) * scale);
  }

  // mid part 
  for (int i = filter_radius; i < nr_bins - filter_radius; ++i) {
    uint32_t sum = 0;
    sp = i - filter_radius;
    ep = i + filter_radius;
    for (int j = sp; j <= ep; ++j) {
      sum += hist_input[j] * params_.smooth_kernel[j - sp];
    }
    hist_output[i] =
        static_cast<uint32_t>(static_cast<float>(sum) * norm_reversed);
  }

  // right boundary part(398,399)
  for (int i = nr_bins - filter_radius; i < nr_bins; ++i) {
    uint32_t sum = 0;
    sp = i - filter_radius;
    ep = nr_bins - 1;
    float kernel_sum = 0.0f;
    for (int j = sp; j <= ep; ++j) {
      sum += hist_input[j] * params_.smooth_kernel[j - sp];
      kernel_sum += static_cast<float>(params_.smooth_kernel[j - sp]);
    }
    float scale = common::IRec(kernel_sum);
    hist_output[i] = static_cast<uint32_t>(static_cast<float>(sum) * scale);
  }
}

void HistogramEstimator::GenerateHat(float *hist_hat, int nr_bins) {
  assert(nr_bins == params_.nr_bins_in_histogram); //400
  // Equation: e^(-(x^2 / (2 * a^2)) + b
  // a -> hat_std_allowed
  // b -> hat_min_allowed
  // max: 1 + hat_min_allowed
  float hat_std_allowed = params_.hat_std_allowed; //6.25
  float hat_min_allowed = params_.hat_min_allowed; //0.40
  int nr_bins_half = nr_bins >> 1; //200
  for (int i = 0; i < nr_bins; ++i) {
    float val = static_cast<float>(i - nr_bins_half) / hat_std_allowed;
    hist_hat_[i] = expf(-val * val * 0.5f) + hat_min_allowed;
  }
}

bool HistogramEstimator::IsGoodShape(const uint32_t *hist, int nr_bins,
                                     int max_index) {
  assert(nr_bins == params_.nr_bins_in_histogram);
  assert(max_index < params_.nr_bins_in_histogram);
  assert(max_index >= 0);

  uint32_t max_value = hist[max_index]; //该pitch对应的统计数
  int nr_bins_half = nr_bins >> 1; // 200

  int offset = max_index - nr_bins_half; //该pitch 距离中位值的距离
  int sp = std::max(offset, 0);
  int ep = std::min(offset + nr_bins, nr_bins); //(0,offset + nr_bins) (offset,nr_bins)

  for (int i = sp; i < ep; ++i) {
    float hat_val = hist_hat_[i - offset];
    if (static_cast<float>(hist[i]) > static_cast<float>(max_value) * hat_val) {
      return false;
    }
  }
  return true;
}

void HistogramEstimator::GetPeakIndexAndMass(const uint32_t *hist, int nr_bins,
                                             int *index, uint32_t *mass) {
  assert(nr_bins == params_.nr_bins_in_histogram);
  assert(hist != nullptr);
  assert(index != nullptr);
  assert(mass != nullptr);
  *index = 0;
  *mass = hist[0];
  uint32_t max_value = hist[0];
  for (int i = 1; i < nr_bins; ++i) {
    if (hist[i] > max_value) {
      *index = i;
      max_value = hist[i];
    }
    *mass += hist[i];
  }
}

// bool HistogramEstimator::SaveHist(const std::string &filename,
//                                   const uint32_t *hist, int nr_bins) {
//   if (nr_bins < 1) {
//     return false;
//   }
//   std::ofstream fout;
//   fout.open(filename.c_str());
//   if (!fout) {
//     std::cerr << "Fail to save the hist: " << filename << std::endl;
//     return false;
//   }
//   for (int i = 0; i < nr_bins; ++i) {
//     fout << hist[i] << std::endl;
//   }
//   fout.close();
//   return true;
// }

}  // namespace camera
}  // namespace perception
}  // namespace apollo
