//
// Created by jachin on 2020/2/26.
//

#ifndef OBJECT_POSTPROCESSOR_OBJECT_POSTPROCESSOR_H
#define OBJECT_POSTPROCESSOR_OBJECT_POSTPROCESSOR_H

//#include <bits/shared_ptr.h>
#include <cmath>
#include <iostream>
#include "i_blas.h"
using namespace std;


// Compute x=K*X, assuming K is 3x3 upper triangular with K[8] = 1.0, do not
// consider radial distortion. //不考虑畸变
template <typename T>
inline void IProjectThroughIntrinsic(const T *K, const T *X, T *x) {
    x[0] = K[0] * X[0] + K[1] * X[1] + K[2] * X[2];
    x[1] = K[4] * X[1] + K[5] * X[2];
    x[2] = X[2];
}

template <typename T>
void GenRotMatrix(const T &ry, T *rot) {
    rot[0] = static_cast<T>(cos(ry));
    rot[2] = static_cast<T>(sin(ry));
    rot[4] = static_cast<T>(1.0f);
    rot[6] = static_cast<T>(-sin(ry));
    rot[8] = static_cast<T>(cos(ry));
    rot[1] = rot[3] = rot[5] = rot[7] = static_cast<T>(0);
}

#endif //OBJECT_POSTPROCESSOR_OBJECT_POSTPROCESSOR_H
