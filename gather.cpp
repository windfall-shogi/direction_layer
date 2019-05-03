#include <torch/torch.h>

#include <vector>

#include "check.hpp"
#include "direction.hpp"

std::vector<at::Tensor> GatherVerticalForwardCuda(const at::Tensor &input,
                                                  const at::Tensor &weights,
                                                  const at::Tensor &bias);
std::vector<at::Tensor> GatherHorizontalForwardCuda(const at::Tensor &input,
                                                    const at::Tensor &weights,
                                                    const at::Tensor &bias);
std::vector<at::Tensor> GatherDiagoanl1ForwardCuda(const at::Tensor &input,
                                                   const at::Tensor &weights,
                                                   const at::Tensor &bias);
std::vector<at::Tensor> GatherDiagonal2ForwardCuda(const at::Tensor &input,
                                                   const at::Tensor &weights,
                                                   const at::Tensor &bias);

std::vector<at::Tensor> GatherVerticalBackwardCuda(
    const std::vector<at::Tensor> &grad_outputs, const at::Tensor &input,
    const at::Tensor &weights, const at::Tensor &bias);
std::vector<at::Tensor> GatherHorizontalBackwardCuda(
    const std::vector<at::Tensor> &grad_outputs, const at::Tensor &input,
    const at::Tensor &weights, const at::Tensor &bias);
std::vector<at::Tensor> GatherDiagonal1BackwardCuda(
    const std::vector<at::Tensor> &grad_outputs, const at::Tensor &input,
    const at::Tensor &weights, const at::Tensor &bias);
std::vector<at::Tensor> GatherDiagonal2BackwardCuda(
    const std::vector<at::Tensor> &grad_outputs, const at::Tensor &input,
    const at::Tensor &weights, const at::Tensor &bias);

//
// Vertical
//
std::vector<at::Tensor> GatherVerticalForward(const at::Tensor &input,
                                              const at::Tensor &weights,
                                              const at::Tensor &bias) {
  return GatherVerticalForwardCuda(input, weights, bias);
}
std::vector<at::Tensor> GatherVerticalBackward(
    const std::vector<at::Tensor> &grad_outputs, const at::Tensor &input,
    const at::Tensor &weights, const at::Tensor &bias) {
  return GatherVerticalBackwardCuda(grad_outputs, input, weights, bias);
}

//
// Horizontal
//
std::vector<at::Tensor> GatherHorizontalForward(const at::Tensor &input,
                                                const at::Tensor &weights,
                                                const at::Tensor &bias) {
  return GatherHorizontalForwardCuda(input, weights, bias);
}
std::vector<at::Tensor> GatherHorizontalBackward(
    const std::vector<at::Tensor> &grad_outputs, const at::Tensor &input,
    const at::Tensor &weights, const at::Tensor &bias) {
  return GatherHorizontalBackwardCuda(grad_outputs, input, weights, bias);
}

//
// Diagonal1
//
std::vector<at::Tensor> GatherDiagonal1Forward(const at::Tensor &input,
                                               const at::Tensor &weights,
                                               const at::Tensor &bias) {
  return GatherDiagoanl1ForwardCuda(input, weights, bias);
}
std::vector<at::Tensor> GatherDiagonal1Backward(
    const std::vector<at::Tensor> &grad_outputs, const at::Tensor &input,
    const at::Tensor &weights, const at::Tensor &bias) {
  return GatherDiagonal1BackwardCuda(grad_outputs, input, weights, bias);
}

//
// Diagonal2
//
std::vector<at::Tensor> GatherDiagonal2Forward(const at::Tensor &input,
                                               const at::Tensor &weights,
                                               const at::Tensor &bias) {
  return GatherDiagonal2ForwardCuda(input, weights, bias);
}
std::vector<at::Tensor> GatherDiagonal2Backward(
    const std::vector<at::Tensor> &grad_outputs, const at::Tensor &input,
    const at::Tensor &weights, const at::Tensor &bias) {
  return GatherDiagonal2BackwardCuda(grad_outputs, input, weights, bias);
}
