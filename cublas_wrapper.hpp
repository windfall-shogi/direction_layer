#pragma once
#ifndef CUBLAS_WRAPPER_HPP_INCLUDED
#define CUBLAS_WRAPPER_HPP_INCLUDED

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <cublasOperation_t OP, typename M>
int Rows(const M& m) {
  return OP == CUBLAS_OP_N ? m.rows : m.cols;
}
template <cublasOperation_t OP, typename M>
int Cols(const M& m) {
  return OP == CUBLAS_OP_N ? m.cols : m.rows;
}

template <typename scalar_t>
cublasStatus_t cublasDot(cublasHandle_t handle, int n, const scalar_t* x,
                         int incx, const scalar_t* y, int incy,
                         scalar_t* result);
template <typename scalar_t>
cublasStatus_t cublasGemmStridedBatched(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const scalar_t* alpha, const scalar_t* A, int lda,
    long long int strideA, const scalar_t* B, int ldb, long long int strideB,
    const scalar_t* beta, scalar_t* C, int ldc, long long int strideC,
    int batchCount);
template <typename scalar_t>
cublasStatus_t cublasGemv(cublasHandle_t handle, cublasOperation_t trans, int m,
                          int n, const scalar_t* alpha, const scalar_t* A,
                          int lda, const scalar_t* x, int incx,
                          const scalar_t* beta, scalar_t* y, int incy);

template <cublasOperation_t OP1, cublasOperation_t OP2, typename scalar_t,
          typename A, typename B, typename C>
void BatchGemm(const scalar_t alpha, const A& a, const B& b,
               const scalar_t beta, C& c, const int batch_size,
               cublasHandle_t& handle, const int space_index) {
  cublasStatus_t status = cublasGemmStridedBatched<scalar_t>(
      handle, OP1, OP2, Rows<OP1>(a), Cols<OP2>(b), Cols<OP1>(a), &alpha,
      a.data + space_index * a.delta, a.ld, a.stride, b.data, b.ld, b.stride,
      &beta, c.data + space_index * c.delta, c.ld, c.stride, batch_size);
}
template <cublasOperation_t OP1, cublasOperation_t OP2, typename scalar_t,
          typename A, typename B, typename C>
void BatchGemm(const scalar_t alpha, const A& a, const B& b,
               const scalar_t beta, C& c, const int batch_size,
               cublasHandle_t& handle, const int index_a, const int index_b) {
  cublasStatus_t status = cublasGemmStridedBatched<scalar_t>(
      handle, OP1, OP2, Rows<OP1>(a), Cols<OP2>(b), Cols<OP1>(a), &alpha,
      a.data + index_a * a.delta, a.ld, a.stride, b.data + index_b * b.delta,
      b.ld, b.stride, &beta, c.data, c.ld, c.stride, batch_size);
}

template <cublasOperation_t OP, typename scalar_t, typename A, typename X,
          typename Y>
void Gemv(const scalar_t alpha, const A& a, const X& x, const scalar_t beta,
          Y& y, cublasHandle_t& handle) {
  cublasStatus_t status =
      cublasGemv<scalar_t>(handle, OP, a.rows, a.cols, &alpha, a.data, a.ld,
                           x.data, x.inc, &beta, y.data, y.inc);
}

template <typename scalar_t, typename Vector1, typename Vector2>
void Dot(const Vector1& x, const Vector2& y, scalar_t* out_ptr,
         cublasHandle_t& handle) {
  cublasStatus_t status = cublasDot<scalar_t>(handle, x.size, x.data, x.inc,
                                              y.data, y.inc, out_ptr);
}
#endif
