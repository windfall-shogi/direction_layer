#include <ATen/ATen.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace {
template <typename scalar_t>
__global__ void OnehotForwardKernel(const int64_t* __restrict__ input,
                                    scalar_t* __restrict__ output) {
  // 1blockで1サンプルを処理
  // 9x9のthreadで各マスを処理
  const int sample_id = blockIdx.x;
  const int x = threadIdx.x, y = threadIdx.y;

  const int k = input[sample_id * 81 + x * 9 + y];
  output[sample_id * 81 * 29 + k * 81 + x * 9 + y] = 1;
}
}  // namespace

std::vector<at::Tensor> OnehotForwardCuda(at::Tensor input) {
  const auto batch_size = input.size(0);
  auto output = at::zeros({batch_size, 29, 9, 9}, at::TensorOptions(at::kCUDA)
                                                      .dtype(at::kFloat)
                                                      .layout(at::kStrided)
                                                      .requires_grad(false)
                                                      .is_variable(true));

  const int blocks = batch_size;
  const dim3 threads(9, 9);
  AT_DISPATCH_FLOATING_TYPES(
      output.type(), "onehot_fowawrd_cuda", ([&] {
        OnehotForwardKernel<scalar_t><<<blocks, threads>>>(
            input.data<int64_t>(), output.data<scalar_t>());
      }));

  return {output};
}