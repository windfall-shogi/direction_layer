#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

#include "cublas_wrapper.hpp"
#include "direction.hpp"
#include "wrapper.hpp"

namespace {
template <typename scalar_t>
__global__ void AddBiasForwardKernel(const scalar_t *__restrict__ bias,
                                     scalar_t *__restrict__ output,
                                     const size_t channel_size,
                                     const size_t feature_size) {
  // バイアス項を先に設定
  const int sample_id = blockIdx.x;
  const int x = threadIdx.x, y = threadIdx.y;

  const scalar_t b = bias[x];
  output[sample_id * channel_size * feature_size + x * feature_size + y] = b;
}
}  // namespace

template <Direction D>
at::Tensor GatherForwardCuda(const at::Tensor &input, const at::Tensor &weights,
                             const at::Tensor &bias, const int offset,
                             const int size, cudaStream_t &stream,
                             cublasHandle_t &handle) {
  // gatherとボトルネックの1x1の畳み込みを同時に計算
  // チャネルの方向に1x1の領域ずつ計算
  // leading dimension = 81
  // 出力はchannel x n
  // 次の処理は1 x nごとにn x nの行列と計算 (g=1のgroup convolutionに対応)

  // A: 1 x in_channels lda=81 strideA=in_channels x 81
  // B: in_channels x out_channels ldb=in_channels strideB=0
  // C: 1 x out_channels ldc=n strideC=n x out_channels
  const int in_channels = input.size(1);
  // weightはfortranスタイルである必要があるので、
  // 列方向はaxia=1(in_channels)、行方向はaxis=0(out_channels)
  const int out_channels = weights.size(0);

  const int batch_size = input.size(0);
  auto output = at::empty({batch_size, out_channels, size}, input.type());

  const int blocks = batch_size;
  const dim3 threads(out_channels, size);
  AT_DISPATCH_FLOATING_TYPES(
      output.type(), "add_bias_fowawrd_cuda", ([&] {
        // バイアス項で初期化
        AddBiasForwardKernel<scalar_t><<<blocks, threads, 0, stream>>>(
            bias.data<scalar_t>(), output.data<scalar_t>(), out_channels, size);

        constexpr scalar_t alpha = 1, beta = 1;
        // gatherと1x1の畳み込み

        // x: [batch] x 1 x in_channels (column major)
        // w: in_cannels x out_channels (column major)
        // y: [batch] x 1 x out_channels (column major)
        const auto x = SpaceSlicedInput<const scalar_t>(input, D, offset);
        const Weight<const scalar_t> w(weights);
        SpaceSlicedOutput<scalar_t> y(output);
        for (int i = 0; i < size; ++i) {
          BatchGemm<CUBLAS_OP_N, CUBLAS_OP_N>(alpha, x, w, beta, y, batch_size,
                                              handle, i);
        }
      }));

  return output;
}

template <Direction D>
void GatherBackwardInputCuda(const at::Tensor &grad_output,
                             const at::Tensor &weights, const int offset,
                             const int size, at::Tensor &grad_input,
                             cublasHandle_t &handle) {
  // outputのサイズはbatch_size x out_channels x size
  // input: x[k, i, j] (実際には4次元配列 ここではiが空間,jが深さに対応)
  // weight: w[m, j]  (mは出力のチャネル)
  // bias: b[m]
  // output: y[k, m, i] (実際の配列の形に対応)
  // y[k, m, i] = \sum_j x[k, i, j] * w[m, j] + b[m]
  // (df / dw)[m, j] = \sum_k \sum_i (df / dy)[k, m, i] x[k, i, j]
  // (df / db)[m] =  \sum_k \sum_i (df / dy)[k, m, i]
  // (df / dx)[k, i, j] = \sum_m (df / dy)[k, m, i]  w[m, j]
  const int batch_size = grad_input.size(0);

  // inputで微分
  AT_DISPATCH_FLOATING_TYPES(
      grad_input.type(), "scatter_backward_input_cuda", ([&] {
        // y: [batch] x 1 x out_channels (column major)
        // w: in_cannels x out_channels (column major)
        // x: [batch] x 1 x in_channels (column major)
        const SpaceSlicedOutput<const scalar_t> y(grad_output);
        const Weight<const scalar_t> w(weights);
        SpaceSlicedInput<scalar_t> x(grad_input, D, offset);
        constexpr scalar_t alpha = 1, beta = 0;
        for (int i = 0; i < size; ++i) {
          BatchGemm<CUBLAS_OP_N, CUBLAS_OP_T>(alpha, y, w, beta, x, batch_size,
                                              handle, i);
        }
      }));
}
template <Direction D>
void GatherBackwardWeightCuda(const at::Tensor &grad_output,
                              const at::Tensor &input, const int offset,
                              const int size, at::Tensor &grad_weights,
                              cublasHandle_t &handle) {
  const int batch_size = input.size(0);
  const int in_channels = input.size(1);
  // weightはfortranスタイルである必要があるので、
  // 列方向はaxia=1(in_channels)、行方向はaxis=0(out_channels)
  const int out_channels = grad_weights.size(0);

  const auto options = at::TensorOptions(at::kCUDA)
                           .dtype(grad_weights.type().scalarType())
                           .layout(at::kStrided);
  // バッチ方向は同時に計算できないので、一時バッファを用意
  auto tmp = at::empty({batch_size}, options);
  auto one = at::ones_like(tmp, options);

  AT_DISPATCH_FLOATING_TYPES(
      grad_weights.type(), "scatter_backward_weights_cuda", ([&] {
        // y: [batch] x size x 1 (column major)
        // x: [batch] x 1 x size (column major)
        // buffer: [batch] x 1 x 1
        // mat_one: [batch] x 1 x 1 (column major)
        Matrix<scalar_t> buffer(tmp);
        const Matrix<const scalar_t> mat_one(one);
        const DepthSlicedOutput<const scalar_t> y(grad_output);
        const DepthSlicedInput<const scalar_t> x(input, D, offset, size);

        // vec_one: batch_size
        // vec_buffer: batch_size
        const Vector<const scalar_t> vec_one(one, batch_size);
        Vector<scalar_t> vec_buffer(tmp, batch_size);

        constexpr scalar_t alpha = 1, beta = 0;

        // dotの計算結果をgpuのメモリで受け取れないので、cpuでメモリを確保
        std::vector<scalar_t> result(in_channels * out_channels);

        for (int i = 0; i < out_channels; ++i) {
          for (int j = 0; j < in_channels; ++j) {
            // メモリの局所性から先に空間方向を計算、バッチ方向は後
            BatchGemm<CUBLAS_OP_T, CUBLAS_OP_T>(alpha, y, x, beta, buffer,
                                                batch_size, handle, i, j);
            // バッチ方向に合計
            Dot(vec_one, vec_buffer, result.data() + i * in_channels + j,
                handle);
          }
        }
        cublasSetVector(result.size(), sizeof(scalar_t), result.data(), 1,
                        grad_weights.data<scalar_t>(), 1);
      }));
}
template <Direction D>
void GatherBackwardBiasCuda(const at::Tensor &grad_output, const int offset,
                            const int size, at::Tensor &grad_bias,
                            cublasHandle_t &handle) {
  const int batch_size = grad_output.size(0);
  const int out_channels = grad_output.size(1);

  const auto options = at::TensorOptions(at::kCUDA)
                           .dtype(grad_bias.type().scalarType())
                           .layout(at::kStrided);
  auto one = at::ones({std::max(batch_size, size)}, options);
  // 空間方向に足し合わせて、その後、バッチ方向に合計
  auto tmp = at::empty({batch_size * out_channels}, options);

  AT_DISPATCH_FLOATING_TYPES(
      grad_bias.type(), "scatter_backward_bias_cuda", ([&] {
        constexpr scalar_t alpha = 1.0, beta = 0.0;
        {
          // A: size x (batch_size x out_channels) (column major)
          // x: size
          // y: (batch_size x out_channels)
          const Matrix<const scalar_t> a(grad_output, size,
                                         batch_size * out_channels);
          const Vector<const scalar_t> x(one);
          Vector<scalar_t> y(tmp);
          Gemv<CUBLAS_OP_T>(alpha, a, x, beta, y, handle);
        }
        {
          // A: out_channels x batch_size  (column major)
          // x: batch_size
          // y: out_channels
          const Matrix<const scalar_t> a(tmp, out_channels, batch_size);
          const Vector<const scalar_t> x(one);
          Vector<scalar_t> y(grad_bias);
          Gemv<CUBLAS_OP_N>(alpha, a, x, beta, y, handle);
        }
      }));
}

template <Direction D, int N = GetNumSlices<D>()>
std::vector<at::Tensor> GatherForward(const at::Tensor &input,
                                      const at::Tensor &weights,
                                      const at::Tensor &bias) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasSetStream(handle, stream);

  std::vector<at::Tensor> output;
  output.reserve(N);
  for (int i = 0; i < N; ++i) {
    auto w = weights[i], b = bias[i];
    const int offset = GetOffset(D, i), size = GetSize(D, i);
    output.emplace_back(
        GatherForwardCuda<D>(input, w, b, offset, size, stream, handle));
  }
  return output;
}

template <Direction D, int N = GetNumSlices<D>()>
std::vector<at::Tensor> GatherBackward(
    const std::vector<at::Tensor> &grad_outputs, const at::Tensor &input,
    const at::Tensor &weights, const at::Tensor &bias) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasSetStream(handle, stream);

  auto grad_input = at::empty_like(input),
       grad_weights = at::empty_like(weights), grad_bias = at::empty_like(bias);
  for (int i = 0; i < N; ++i) {
    const auto w = weights[i], b = bias[i];
    auto grad_w = grad_weights[i], grad_b = grad_bias[i];

    const int offset = GetOffset(D, i), size = GetSize(D, i);

    const auto grad_output = grad_outputs[i];
    GatherBackwardInputCuda<D>(grad_output, w, offset, size, grad_input,
                               handle);
    GatherBackwardWeightCuda<D>(grad_output, input, offset, size, grad_w,
                                handle);
    GatherBackwardBiasCuda<D>(grad_output, offset, size, grad_b, handle);
  }
  return {grad_input, grad_weights, grad_bias};
}

std::vector<at::Tensor> GatherVerticalForwardCuda(const at::Tensor &input,
                                                  const at::Tensor &weights,
                                                  const at::Tensor &bias) {
  return GatherForward<VERTICAL>(input, weights, bias);
}
std::vector<at::Tensor> GatherHorizontalForwardCuda(const at::Tensor &input,
                                                    const at::Tensor &weights,
                                                    const at::Tensor &bias) {
  return GatherForward<HORIZONTAL>(input, weights, bias);
}
std::vector<at::Tensor> GatherDiagoanl1ForwardCuda(const at::Tensor &input,
                                                   const at::Tensor &weights,
                                                   const at::Tensor &bias) {
  return GatherForward<DIAGONAL1>(input, weights, bias);
}
std::vector<at::Tensor> GatherDiagonal2ForwardCuda(const at::Tensor &input,
                                                   const at::Tensor &weights,
                                                   const at::Tensor &bias) {
  return GatherForward<DIAGONAL2>(input, weights, bias);
}

std::vector<at::Tensor> GatherVerticalBackwardCuda(
    const std::vector<at::Tensor> &grad_outputs, const at::Tensor &input,
    const at::Tensor &weights, const at::Tensor &bias) {
  return GatherBackward<VERTICAL>(grad_outputs, input, weights, bias);
}
std::vector<at::Tensor> GatherHorizontalBackwardCuda(
    const std::vector<at::Tensor> &grad_outputs, const at::Tensor &input,
    const at::Tensor &weights, const at::Tensor &bias) {
  return GatherBackward<HORIZONTAL>(grad_outputs, input, weights, bias);
}
std::vector<at::Tensor> GatherDiagonal1BackwardCuda(
    const std::vector<at::Tensor> &grad_outputs, const at::Tensor &input,
    const at::Tensor &weights, const at::Tensor &bias) {
  return GatherBackward<DIAGONAL1>(grad_outputs, input, weights, bias);
}
std::vector<at::Tensor> GatherDiagonal2BackwardCuda(
    const std::vector<at::Tensor> &grad_outputs, const at::Tensor &input,
    const at::Tensor &weights, const at::Tensor &bias) {
  return GatherBackward<DIAGONAL2>(grad_outputs, input, weights, bias);
}
