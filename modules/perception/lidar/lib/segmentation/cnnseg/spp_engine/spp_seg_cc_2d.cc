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
#include <algorithm>

#include "modules/perception/lidar/common/lidar_log.h"
#include "modules/perception/lidar/common/lidar_timer.h"
#include "modules/perception/lidar/lib/segmentation/cnnseg/spp_engine/spp_seg_cc_2d.h"

namespace apollo {
namespace perception {
namespace lidar {

void SppCCDetector::SetData(const float* const* prob_map,
                            const float* offset_map, float scale,
                            float objectness_threshold) {
  prob_map_ = prob_map; //网络输出的category_pt_blob
  offset_map_ = offset_map; //网络输出的instance_pt_blob
  scale_ = scale; //对应尺度 每米对应的网格数 
  objectness_threshold_ = objectness_threshold;
  worker_.Bind(std::bind(&SppCCDetector::CleanNodes, this));
  worker_.Start();
}

bool SppCCDetector::BuildNodes(int start_row_index, int end_row_index) { //0-864
  const float* offset_row_ptr = offset_map_ + start_row_index * cols_; //行首开始指针
  const float* offset_col_ptr = offset_map_ + (rows_ + start_row_index) * cols_; //列开始指针 偏移rows*cols(相当于一个信息的特征图网格)
  const float* prob_map_ptr = prob_map_[0] + start_row_index * cols_;
  Node* node_ptr = nodes_[0] + start_row_index * cols_; //特征图行首对应的节点指针
  for (int row = start_row_index; row < end_row_index; ++row) {
    for (int col = 0; col < cols_; ++col) {
      node_ptr->set_is_object(*prob_map_ptr++ >= objectness_threshold_);
      int center_row = static_cast<int>(*offset_row_ptr++ * scale_ +
                                        static_cast<float>(row) + 0.5f); //offset_row_ptr单位为米 转换为网格即为偏移的网格数，向上取整
      int center_col = static_cast<int>(*offset_col_ptr++ * scale_ +
                                        static_cast<float>(col) + 0.5f); //同理
      center_row = std::max(0, std::min(rows_ - 1, center_row)); //限制偏移不能超出网格
      center_col = std::max(0, std::min(cols_ - 1, center_col));
      (node_ptr++)->center_node = center_row * cols_ + center_col; //合并偏移 对应得到最终的中心偏移的网格数所在的位置 相对于同一原点的一维索引
    }
  }
  return true;
}

bool SppCCDetector::CleanNodes() {
  memset(nodes_[0], 0, sizeof(Node) * rows_ * cols_);
  uint32_t node_idx = 0;
  for (int row = 0; row < rows_; ++row) {
    for (int col = 0; col < cols_; ++col) {
      nodes_[row][col].parent = node_idx++;
    }
  }
  return true;
}

size_t SppCCDetector::Detect(SppLabelImage* labels) {
  Timer timer;
  if (!first_process_) {
    worker_.Join();  // sync for cleaning nodes
  }
  first_process_ = false;
  BuildNodes(0, rows_); //在特征图每个格中创建接待你
  double init_time = timer.toc(true);

  double sync_time = timer.toc(true);

  TraverseNodes();
  double traverse_time = timer.toc(true);

  UnionNodes();
  double union_time = timer.toc(true);

  size_t num = ToLabelMap(labels);
  worker_.WakeUp();  // for next use
  double collect_time = timer.toc(true);

  AINFO << "SppSegCC2D: init: " << init_time << "\tsync: " << sync_time
        << "\ttraverse: " << traverse_time << "\tunion: " << union_time
        << "\tcollect: " << collect_time << "\t#obj: " << num;

  return num;
}

void SppCCDetector::TraverseNodes() {
  for (int row = 0; row < rows_; row++) {
    for (int col = 0; col < cols_; col++) {
      Node& node = nodes_[row][col];
      if (node.is_object() && node.get_traversed() == 0) {
        Traverse(&node);
      }
    }
  }
}

void SppCCDetector::UnionNodes() {
  for (int row = 0; row < rows_; ++row) {
    for (int col = 0; col < cols_; ++col) {
      Node* node = &nodes_[row][col];
      if (!node->is_center()) {
        continue;
      }
      Node* node_neighbor = nullptr;
      // right
      if (col < cols_ - 1) {
        node_neighbor = &nodes_[row][col + 1];
        if (node_neighbor->is_center()) {
          DisjointSetUnion(node, node_neighbor);
        }
      }
      // down
      if (row < rows_ - 1) {
        node_neighbor = &nodes_[row + 1][col];
        if (node_neighbor->is_center()) {
          DisjointSetUnion(node, node_neighbor);
        }
      }
      // right down
      if (row < rows_ - 1 && col < cols_ - 1) {
        node_neighbor = &nodes_[row + 1][col + 1];
        if (node_neighbor->is_center()) {
          DisjointSetUnion(node, node_neighbor);
        }
      }
      // left down
      if (row < rows_ - 1 && col > 0) {
        node_neighbor = &nodes_[row + 1][col - 1];
        if (node_neighbor->is_center()) {
          DisjointSetUnion(node, node_neighbor);
        }
      }
    }
  }
}

size_t SppCCDetector::ToLabelMap(SppLabelImage* labels) {
  uint16_t id = 0;
  uint32_t pixel_id = 0;
  labels->ResetClusters(kDefaultReserveSize);
  for (int row = 0; row < rows_; ++row) {
    for (int col = 0; col < cols_; ++col, ++pixel_id) {
      Node* node = &nodes_[row][col];
      if (!node->is_object()) {
        (*labels)[row][col] = 0;
        continue;
      }
      Node* root = DisjointSetFind(node);
      // note label in label image started from 1,
      // zero is reserved from non-object
      if (!root->id) {
        root->id = ++id; //从1开始
      }
      (*labels)[row][col] = root->id;
      labels->AddPixelSample(root->id - 1, pixel_id); //向聚类cluster中添加网格对应的索引(从0开始)
    }
  }
  labels->ResizeClusters(id);
  return id; //对应每个cluster的id
}

void SppCCDetector::Traverse(SppCCDetector::Node* x) {
  std::vector<SppCCDetector::Node*> p;
  p.clear();
  while (x->get_traversed() == 0) {
    p.push_back(x); //对应原始节点的向量
    x->set_traversed(2);
    x = nodes_[0] + x->center_node; //对应偏移节点
  } //节点不断向前偏移，直到get_traversed!=0 (沿着偏移方向不断前进)
  if (x->get_traversed() == 2) { //此时x已经遍历到最终的中心
    for (int i = static_cast<int>(p.size()) - 1; i >= 0 && p[i] != x; i--) {
      p[i]->set_is_center(true); //将到最终中心点的所有路径上的节点设置为center
    }
    x->set_is_center(true);
  }
  for (size_t i = 0; i < p.size(); i++) {
    Node* y = p[i];
    y->set_traversed(1); //各节点的travrsed为1 只有从未遍历过的到初次遍历时为2
    y->parent = x->parent; //parent对应各节点的索引，将最终中心的索引值 设置为各中间节点的parent
  }
}

SppCCDetector::Node* SppCCDetector::DisjointSetFindLoop(Node* x) {
  Node* root = x;
  while (nodes_[0] + root->parent != root) {
    root = nodes_[0] + root->parent;
  }
  Node* w = x;
  while (nodes_[0] + w->parent != w) {
    Node* temp = nodes_[0] + w->parent;
    w->parent = root->parent;
    w = temp;
  }
  return root;
}

SppCCDetector::Node* SppCCDetector::DisjointSetFind(Node* x) {
  Node* y = nodes_[0] + x->parent;
  if (y == x || nodes_[0] + y->parent == y) {
    return y;
  }//父节点是自身或者指向同一个父节点
  Node* root = DisjointSetFindLoop(nodes_[0] + y->parent);
  x->parent = root->parent;
  y->parent = root->parent;
  return root;
}

void SppCCDetector::DisjointSetUnion(Node* x, Node* y) {
  x = DisjointSetFind(x); //返回该节点对应父节点
  y = DisjointSetFind(y);
  if (x == y) {
    return;
  }
  uint16_t x_node_rank = x->get_node_rank();
  uint16_t y_node_rank = y->get_node_rank();
  if (x_node_rank < y_node_rank) {
    x->parent = y->parent;
  } else if (y_node_rank < x_node_rank) {
    y->parent = x->parent;
  } else {
    y->parent = x->parent;
    x->set_node_rank(static_cast<uint16_t>(x_node_rank + 1));
  }
}

}  // namespace lidar
}  // namespace perception
}  // namespace apollo
