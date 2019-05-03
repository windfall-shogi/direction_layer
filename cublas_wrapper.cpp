#include "cublas_wrapper.hpp"

template <>
cublasStatus_t cublasDot(cublasHandle_t handle, int n, const float* x, int incx,
                         const float* y, int incy, float* result) {
  return cublasSdot(handle, n, x, incx, y, incy, result);
}
template <>
cublasStatus_t cublasDot(cublasHandle_t handle, int n, const double* x,
                         int incx, const double* y, int incy, double* result) {
  return cublasDdot(handle, n, x, incx, y, incy, result);
}

template <>
cublasStatus_t cublasGemmStridedBatched(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const float* alpha, const float* A, int lda,
    long long int strideA, const float* B, int ldb, long long int strideB,
    const float* beta, float* C, int ldc, long long int strideC,
    int batchCount) {
  return cublasSgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A,
                                   lda, strideA, B, ldb, strideB, beta, C, ldc,
                                   strideC, batchCount);
}
template <>
cublasStatus_t cublasGemmStridedBatched(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const double* alpha, const double* A, int lda,
    long long int strideA, const double* B, int ldb, long long int strideB,
    const double* beta, double* C, int ldc, long long int strideC,
    int batchCount) {
  return cublasDgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A,
                                   lda, strideA, B, ldb, strideB, beta, C, ldc,
                                   strideC, batchCount);
}

template <>
cublasStatus_t cublasGemv(cublasHandle_t handle, cublasOperation_t trans, int m,
                          int n, const float* alpha, const float* A, int lda,
                          const float* x, int incx, const float* beta, float* y,
                          int incy) {
  return cublasSgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y,
                     incy);
}
template <>
cublasStatus_t cublasGemv(cublasHandle_t handle, cublasOperation_t trans, int m,
                          int n, const double* alpha, const double* A, int lda,
                          const double* x, int incx, const double* beta,
                          double* y, int incy) {
  return cublasDgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y,
                     incy);
}