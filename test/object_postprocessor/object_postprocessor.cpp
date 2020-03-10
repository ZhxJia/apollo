//
// Created by jachin on 2020/2/26.
//

#include "object_postprocessor.h"
#include <iostream>
#include <vector>
#include <limits>
#include <math.h>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;

float hwl[3] = {0.8,1.0,2.4};
float w_half = hwl[1] / 2;
float l_half = hwl[2] / 2;
float x_cor[4] = {l_half, l_half, -l_half, -l_half};
float z_cor[4] = {w_half, -w_half, -w_half, w_half};
float pts[12] = {x_cor[0], 0.0f, z_cor[0], x_cor[1], 0.0f, z_cor[1],
                 x_cor[2], 0.0f, z_cor[2], x_cor[3], 0.0f, z_cor[3]}; //3D box位于地面的边界框(注意y坐标之前已经减了h/2，故为0)

float pt_proj[3] = {0};
float pt_c[3] = {0};
float *pt = pts;
float center[3]={20.0,30.4,0.8};
float k_mat_[9]={1975.43120868945, 0, 958.0687372936763, 0, 1979.281590341125, 460.0715543199537, 0, 0, 1};
// Compute x=[R|t]*X, assuming R is 3x3 rotation matrix and t is a 3-vector.
template <typename T>
inline void IProjectThroughExtrinsic(const T *R, const T *t, const T *X, T *x) {

    IMultAx3x3(R, X, x);//x=RX
    IAdd3(t, x);
}

int main(int argc, char const *argv[]) {
    float rot[9] = {0};
    GenRotMatrix<float>(2.1, rot);
    for (int i = 0; i < 4; ++i) {
        IProjectThroughExtrinsic(rot,center,pt,pt_c);
        IProjectThroughIntrinsic(k_mat_, pt_c, pt_proj); //投影到图像平面pt_proj

    }
    for(auto &p : pt_proj){

    }

    cout << std::max(-1.5,-2.0)<<endl;
    float l_[3]={0};
    std::vector<float> ground3(l_,l_+3);

    int nr_inliers = 3;
    int inliers[3] = {1,2,3};
    int vd[12] = {1,2,3,4,5,6,7,8,9,10,11,12};
    // re-fit using inliers
    int count = 0;
    for (int i = 0; i < nr_inliers; ++i) {
        int i2 = inliers[i] << 1; //内点的索引 * 2 为在inliers中的索引位置
        int count2 = count << 1; //0 2 4 ..
        std::swap(vd[i2], vd[count2]);
        std::swap(vd[i2 + 1], vd[count2 + 1]);
        ++count;
//        cout << count <<endl;
    }//将模型的内点提前
//    for (auto &v : vd){
//        cout << v << endl;
//    }

//    std::vector<float> ph(2, 0);
//    int track_length = 3;
//    std::vector<float> const_weight_temporal_;
//    const_weight_temporal_.resize(track_length,0.0);
//    for (int i = track_length - 1; i >= 0; --i) {
//        const_weight_temporal_.at(i) =
//                powf(sqrtf(2.0f), track_length - 1 - i);//根2
//        cout << const_weight_temporal_.at(i) << endl;
//    }
    Eigen::Vector3d measure;

    float v[3]={1,2,3};
    measure << v[0], v[1],v[2];

    cout << measure.head(2) << endl;




    return 0;
}