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

#include "modules/perception/common/i_lib/core/i_blas.h"
#include "modules/perception/common/i_lib/core/i_rand.h"
#include "modules/perception/common/i_lib/core/i_struct.h"

#include <limits>

namespace apollo {
namespace perception {
namespace common {
// Compute the number of trials for Ransac. Number of trials is chosen
// sufficiently high to ensure with a probability "confidence" that at
// least one of the random samples  of data is free from outliers.
// "inlierprob" is the probability that any selected data point is an inlier
inline int IRansacTrials(int sample_size, double confidence,
                         double inlierprob) {
  return inlierprob > 0.0
             ? IRound(IDiv(ILog(1.0 - confidence),
                           ILog(1.0 - IPow(inlierprob, sample_size))))
             : std::numeric_limits<int>::max();
}

// Using Ransac to fit a model to data set which contains outliers
// The function needs 2n entries of scratch space in inliers
template <typename T, int l, int lp, int k, int s,
          void (*HypogenFunc)(const T* x, const T* xp, T* model),
          void (*CostFunc)(const T* model, const T* x, const T* xp, int n,
                           int* nr_liner, int* inliers, T* cost, T error_tol),
          void (*RefitFunc)(T* x, T* xp, int* inliers, T* model, int n,
                            int nr_liner)>
bool RobustBinaryFitRansac(T* x, T* xp, int n, T* model, int* consensus_size,
                           int* inliers, T error_tol,
                           bool re_est_model_w_inliers = false,
                           bool adaptive_trial_count = false,
                           double confidence = 0.99, double inlierprob = 0.5,
                           int min_nr_inliers = s,
                           bool random_shuffle_inputs = false) {
  const int kSize = s;    //2
  const int kLength = l;  //1
  const int kLp = lp;     //1
  const int kKsize = k;   //2
  int indices[kSize];
  T samples_x[kLength * kSize]; //2
  T samples_xp[kLp * kSize];    //2
  T tmp_model[kKsize];
  T cost = std::numeric_limits<T>::max();//返回编译器允许的该T类型的最大值
  T best_cost = std::numeric_limits<T>::max();

  if (n < min_nr_inliers) {
    return false;
  } //refs的数量小于min

  double actual_inlierprob = 0.0, tmp_inlierprob;
  int nr_trials = IRansacTrials(s, confidence, inlierprob);//inlierpron为任意选择一个点为局内点(有效点) 返回Ransac尝试次数(16)

  int nr_inliers = 0;
  int rseed = I_DEFAULT_SEED;
  int sample_count = 0;
  int i, idxl, idxlp, il, ilp;
  *consensus_size = 0;  // initialize the size of the consensus set to zero
  IZero(model, k);      // initialize the model with zeros

  if (random_shuffle_inputs) {
    IRandomizedShuffle(x, xp, n, l, lp, &rseed);//随机排列 n为样本refs的总数   x->vs xp->ds
  }

  while (nr_trials > sample_count) {
    // generate random indices
    IRandomSample(indices, s, n, &rseed); //随机采样 s=2 n为refs的数量 rseed =432 从[0,n)中每次采2个数
    // prepare data for model fitting
    for (i = 0; i < s; ++i) {
      idxl = indices[i] * l; //l=lp=1
      idxlp = indices[i] * lp;
      il = i * l;
      ilp = i * lp;
      ICopy(x + idxl, samples_x + il, l); //复制x对应索引的值到samples_x[0,1]中
      ICopy(xp + idxlp, samples_xp + ilp, lp);//复制xp的值到samples_xp[0,1]中
    } //得到两对采样值x[0],xp[0]和x[1],xp[1]

    // estimate model
    HypogenFunc(samples_x, samples_xp, tmp_model); //samples_x中存储box的y_max,samples_xp中存储该物体的深度倒数1/z

    // validate model  cost为内点的总误差，inliers存放内点的索引 nr_inliers内点数量
    CostFunc(tmp_model, x, xp, n, &nr_inliers, inliers + n, &cost, error_tol);
    if ((nr_inliers > *consensus_size) ||
        (nr_inliers == *consensus_size && cost < best_cost)) {
      *consensus_size = nr_inliers; //目前匹配的最多的内点数
      best_cost = cost;
      ICopy(tmp_model, model, k); //将匹配内点数最多的模型导出
      ICopy(inliers + n, inliers, *consensus_size);  // record inlier indices 将匹配最多的模型的内点索引记录
      if (adaptive_trial_count) { //自适应尝试
        tmp_inlierprob = IDiv(static_cast<double>(*consensus_size), n);
        if (tmp_inlierprob > actual_inlierprob) {
          actual_inlierprob = tmp_inlierprob;
          nr_trials = IRansacTrials(s, confidence, actual_inlierprob);//调整任意一个点为内点的概率，以此调整trial次数
        }
      }
    }
    sample_count++;
  }
  bool succeeded = *consensus_size >= min_nr_inliers;

  if (succeeded && re_est_model_w_inliers && RefitFunc != nullptr) {
    RefitFunc(x, xp, inliers, model, n, *consensus_size);
  }
  return succeeded;
}

}  // namespace common
}  // namespace perception
}  // namespace apollo
