





#ifndef I_BLAS_H
#define I_BLAS_H

// Multiply 3 x 3 matrix A with 3-dimensional vector x
template <typename T>
inline void IMultAx3x3(const T A[9], const T x[3], T Ax[3]) {
  T x0, x1, x2;
  x0 = x[0];
  x1 = x[1];
  x2 = x[2];
  Ax[0] = A[0] * x0 + A[1] * x1 + A[2] * x2;
  Ax[1] = A[3] * x0 + A[4] * x1 + A[5] * x2;
  Ax[2] = A[6] * x0 + A[7] * x1 + A[8] * x2;
}

template <typename T>
inline void IAdd3(const T x[3], T y[3]) {
  y[0] += x[0];
  y[1] += x[1];
  y[2] += x[2];
}

#endif