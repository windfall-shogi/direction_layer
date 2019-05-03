#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <vector>

#include "cublas_wrapper.hpp"
#include "direction.hpp"
#include "wrapper.hpp"

namespace {
template <Direction D>
__device__ int GetIndex(const int x, const int y);
template <>
__device__ int GetIndex<VERTICAL>(const int x, const int y) {
  return x;
}
template <>
__device__ int GetIndex<HORIZONTAL>(const int x, const int y) {
  return y;
}
template <>
__device__ int GetIndex<DIAGONAL1>(const int x, const int y) {
  return 8 - x + y;
}
template <>
__device__ int GetIndex<DIAGONAL2>(const int x, const int y) {
  return x + y;
}

template <Direction D, typename scalar_t>
__global__ void AddBiasForwardKernel(const scalar_t* __restrict__ bias,
                                     scalar_t* __restrict__ output,
                                     const size_t channel_size) {
  // バイアス項を先に設定
  const int sample_id = blockIdx.x;
  const int x = threadIdx.x, c = threadIdx.y;

  const int offset = (sample_id * channel_size + c) * 81 + x * 9;
  output[offset + 0] = bias[GetIndex<D>(x, 0) * channel_size + c];
  output[offset + 1] = bias[GetIndex<D>(x, 1) * channel_size + c];
  output[offset + 2] = bias[GetIndex<D>(x, 2) * channel_size + c];
  output[offset + 3] = bias[GetIndex<D>(x, 3) * channel_size + c];
  output[offset + 4] = bias[GetIndex<D>(x, 4) * channel_size + c];
  output[offset + 5] = bias[GetIndex<D>(x, 5) * channel_size + c];
  output[offset + 6] = bias[GetIndex<D>(x, 6) * channel_size + c];
  output[offset + 7] = bias[GetIndex<D>(x, 7) * channel_size + c];
  output[offset + 8] = bias[GetIndex<D>(x, 8) * channel_size + c];
}

template <typename scalar_t>
__global__ void SumSpace(const scalar_t* grad_output,
                         scalar_t* __restrict__ grad_bias,
                         const int channel_size, const Direction direction,
                         const int size, const int offset) {
  const int sample_id = blockIdx.x;
  const int channel_id = threadIdx.x;

  const int index = sample_id * channel_size * 81 + channel_id * 81;
  scalar_t tmp = 0;
  for (int i = 0; i < size; ++i) {
    tmp += grad_output[index + direction * i + offset];
  }
  grad_bias[sample_id * channel_size + channel_id] = tmp;
}
}  // namespace

template <Direction D>
void ScatterForwardBiasCuda(const at::Tensor& bias, at::Tensor& output,
                            cudaStream_t& stream) {
  const int out_channels = output.size(1);
  const int blocks = output.size(0);
  const dim3 threads(9, out_channels);

  AT_DISPATCH_FLOATING_TYPES(
      output.type(), "scatter_add_bias_fowawrd_cuda", ([&] {
        AddBiasForwardKernel<D, scalar_t><<<blocks, threads, 0, stream>>>(
            bias.data<scalar_t>(), output.data<scalar_t>(), out_channels);
      }));
}

template <Direction D>
void ScatterForwardWeightCuda(const at::Tensor& input,
                              const at::Tensor& weights, const int offset,
                              const int size, at::Tensor& output,
                              cublasHandle_t& handle) {
  // scatterとボトルネックの1x1の畳み込みを同時に計算
  // 空間方向を分割して、ループで処理する

  // A: 1 x in_channels lda=size strideA=in_channels x size
  // B: in_channels x out_channels ldb=in_channels strideB=0
  // C: 1 x out_channels ldc=81 strideC=out_channels x 81
  const int in_channels = input.size(1);
  // weightはfortranスタイルである必要があるので、
  // 列方向はaxia=1(in_channels)、行方向はaxis=0(out_channels)
  const int out_channels = weights.size(0);

  const int batch_size = input.size(0);

  AT_DISPATCH_FLOATING_TYPES(
      output.type(), "scatter_fowawrd_weight_cuda", ([&] {
        constexpr scalar_t alpha = 1, beta = 1;

        // x: [batch] x 1 x in_channels (column major)
        // w: in_cannels x out_channels (column major)
        // y: [batch] x 1 x out_channels (column major)
        const SpaceSlicedInput<const scalar_t> x(input);
        const Weight<const scalar_t> w(weights);
        SpaceSlicedOutput<scalar_t> y(output, D, offset);
        for (int i = 0; i < size; ++i) {
          BatchGemm<CUBLAS_OP_N, CUBLAS_OP_N>(alpha, x, w, beta, y, batch_size,
                                              handle, i);
        }
      }));
}

template <Direction D>
void ScatterBackwardInputCuda(const at::Tensor& grad_output,
                              const at::Tensor& weights, const int offset,
                              const int size, at::Tensor& grad_input,
                              cublasHandle_t& handle) {
  const int batch_size = grad_output.size(0);
  // weightはfortranスタイルである必要があるので、
  // 列方向はaxia=1(in_channels)、行方向はaxis=0(out_channels)
  const int out_channels = weights.size(0);
  const int in_channels = weights.size(1);

  // inputで微分
  AT_DISPATCH_FLOATING_TYPES(
      grad_input.type(), "input_backward_cuda", ([&] {
        // y: [batch] x 1 x out_channels (column major)
        // w: in_cannels x out_channels (column major)
        // x: [batch] x 1 x in_channels (column major)
        const SpaceSlicedOutput<const scalar_t> y(grad_output, D, offset);
        const Weight<const scalar_t> w(weights);
        SpaceSlicedInput<scalar_t> x(grad_input);
        constexpr scalar_t alpha = 1, beta = 0;
        for (int i = 0; i < size; ++i) {
          BatchGemm<CUBLAS_OP_N, CUBLAS_OP_T>(alpha, y, w, beta, x, batch_size,
                                              handle, i);
        }
      }));
}
template <Direction D>
void ScatterBackwardWeightCuda(const at::Tensor& grad_output,
                               const at::Tensor& input, const int offset,
                               const int size, at::Tensor& grad_weights,
                               cublasHandle_t& handle) {
  const int batch_size = input.size(0);
  const int in_channels = input.size(1);
  const int out_channels = grad_output.size(1);

  const auto options = at::TensorOptions(at::kCUDA)
                           .dtype(grad_weights.type().scalarType())
                           .layout(at::kStrided);
  // バッチ方向は同時に計算できないので、一時バッファを用意
  auto tmp = at::empty({batch_size}, options);
  const auto one = at::ones_like(tmp, options);

  AT_DISPATCH_FLOATING_TYPES(
      grad_weights.type(), "weights_backward_cuda", ([&] {
        // y: [batch] x 1 xsize (column major)
        // x: [batch] x size x  1 (column major)
        // buffer: [batch] x 1 x 1
        // mat_one: [batch] x 1 x 1 (column major)
        Matrix<scalar_t> buffer(tmp);
        const Matrix<const scalar_t> mat_one(one);
        const DepthSlicedOutput<const scalar_t> y(grad_output, D, size, offset);
        const DepthSlicedInput<const scalar_t> x(input);

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
            BatchGemm<CUBLAS_OP_N, CUBLAS_OP_N>(alpha, y, x, beta, buffer,
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
void ScatterBackwardBiasCuda(const at::Tensor& grad_output, const int offset,
                             const int size, at::Tensor& grad_bias,
                             cublasHandle_t& handle) {
  const int batch_size = grad_output.size(0);
  const int out_channels = grad_output.size(1);

  const auto options = at::TensorOptions(at::kCUDA)
                           .dtype(grad_bias.type().scalarType())
                           .layout(at::kStrided);
  const auto one = at::ones({std::max(batch_size, size)}, options);
  // 空間方向に足し合わせて、その後、バッチ方向に合計
  auto tmp = at::empty({batch_size * out_channels}, options);

  AT_DISPATCH_FLOATING_TYPES(
      grad_bias.type(), "bias_backward_cuda", ([&] {
        constexpr scalar_t alpha = 1, beta = 0;
        {
          // A: [batch_size x out_channels] x 1 x size (column major)
          // x: size x 1 (column major)
          // y: [batch_size x out_channels] x 1 x 1 (column major)
          const DepthSlicedOutput<const scalar_t> a(grad_output, D, size,
                                                    offset, 0);
          const Matrix<const scalar_t> x(one, size, 1, 0);
          Matrix<scalar_t> y(tmp, 1, 1);
          BatchGemm<CUBLAS_OP_N, CUBLAS_OP_N>(
              alpha, a, x, beta, y, batch_size * out_channels, handle, 0);

          // SumSpace<scalar_t><<<batch_size, out_channels>>>(
          //     grad_output.data<scalar_t>(), tmp.data<scalar_t>(),
          //     out_channels, direction, size, offset);
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
at::Tensor ScatterForward(const std::vector<at::Tensor>& inputs,
                          const at::Tensor& weights, const at::Tensor& bias) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasSetStream(handle, stream);

  at::Tensor output =
      at::empty({inputs[0].size(0), bias.size(1), 9, 9}, inputs[0].type());
  ScatterForwardBiasCuda<D>(bias, output, stream);
  for (int i = 0; i < N; ++i) {
    const auto in = inputs[i], w = weights[i];
    const int offset = GetOffset(D, i), size = GetSize(D, i);
    ScatterForwardWeightCuda<D>(in, w, offset, size, output, handle);
  }
  return output;
}

template <Direction D, int N = GetNumSlices<D>()>
std::vector<at::Tensor> ScatterBackward(const at::Tensor& grad_output,
                                        const std::vector<at::Tensor>& inputs,
                                        const at::Tensor& weights,
                                        const at::Tensor& bias) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasSetStream(handle, stream);

  std::vector<at::Tensor> outputs;
  outputs.reserve(inputs.size() + 2);

  at::Tensor grad_weights = at::empty_like(weights),
             grad_bias = at::empty_like(bias);
  for (int i = 0; i < N; ++i) {
    at::Tensor w = weights[i];

    at::Tensor grad_in = at::empty_like(inputs[i]);
    at::Tensor grad_w = grad_weights[i], grad_b = grad_bias[i];

    const int offset = GetOffset(D, i), size = GetSize(D, i);

    ScatterBackwardInputCuda<D>(grad_output, w, offset, size, grad_in,
                                handle);
    ScatterBackwardWeightCuda<D>(grad_output, inputs[i], offset, size, grad_w,
                                 handle);
    ScatterBackwardBiasCuda<D>(grad_output, offset, size, grad_b, handle);

    outputs.emplace_back(std::move(grad_in));
  }
  outputs.emplace_back(std::move(grad_weights));
  outputs.emplace_back(std::move(grad_bias));
  return outputs;
}

at::Tensor ScatterVerticalForwardCuda(const std::vector<at::Tensor>& inputs,
                                      const at::Tensor& weights,
                                      const at::Tensor& bias) {
  return ScatterForward<VERTICAL>(inputs, weights, bias);
}
at::Tensor ScatterHorizontalForwardCuda(const std::vector<at::Tensor>& inputs,
                                        const at::Tensor& weights,
                                        const at::Tensor& bias) {
  return ScatterForward<HORIZONTAL>(inputs, weights, bias);
}
at::Tensor ScatterDiagonal1ForwardCuda(const std::vector<at::Tensor>& inputs,
                                       const at::Tensor& weights,
                                       const at::Tensor& bias) {
  return ScatterForward<DIAGONAL1>(inputs, weights, bias);
}
at::Tensor ScatterDiagonal2ForwardCuda(const std::vector<at::Tensor>& inputs,
                                       const at::Tensor& weights,
                                       const at::Tensor& bias) {
  return ScatterForward<DIAGONAL2>(inputs, weights, bias);
}

std::vector<at::Tensor> ScatterVerticalBackwardCuda(
    const at::Tensor& grad_output, const std::vector<at::Tensor>& inputs,
    const at::Tensor& weights, const at::Tensor& bias) {
  return ScatterBackward<VERTICAL>(grad_output, inputs, weights, bias);
}
std::vector<at::Tensor> ScatterHorizontalBackwardCuda(
    const at::Tensor& grad_output, const std::vector<at::Tensor>& inputs,
    const at::Tensor& weights, const at::Tensor& bias) {
  return ScatterBackward<HORIZONTAL>(grad_output, inputs, weights, bias);
}
std::vector<at::Tensor> ScatterDiagonal1BackwardCuda(
    const at::Tensor& grad_output, const std::vector<at::Tensor>& inputs,
    const at::Tensor& weights, const at::Tensor& bias) {
  return ScatterBackward<DIAGONAL1>(grad_output, inputs, weights, bias);
}
std::vector<at::Tensor> ScatterDiagonal2BackwardCuda(
    const at::Tensor& grad_output, const std::vector<at::Tensor>& inputs,
    const at::Tensor& weights, const at::Tensor& bias) {
  return ScatterBackward<DIAGONAL2>(grad_output, inputs, weights, bias);
}
