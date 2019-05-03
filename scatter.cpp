#include <torch/torch.h>

#include <vector>

#include "check.hpp"
#include "direction.hpp"

at::Tensor ScatterVerticalForwardCuda(const std::vector<at::Tensor>& inputs,
                                      const at::Tensor& weights,
                                      const at::Tensor& bias);
at::Tensor ScatterHorizontalForwardCuda(const std::vector<at::Tensor>& inputs,
                                        const at::Tensor& weights,
                                        const at::Tensor& bias);
at::Tensor ScatterDiagonal1ForwardCuda(const std::vector<at::Tensor>& inputs,
                                       const at::Tensor& weights,
                                       const at::Tensor& bias);
at::Tensor ScatterDiagonal2ForwardCuda(const std::vector<at::Tensor>& inputs,
                                       const at::Tensor& weights,
                                       const at::Tensor& bias);

std::vector<at::Tensor> ScatterVerticalBackwardCuda(
    const at::Tensor& grad_output, const std::vector<at::Tensor>& inputs,
    const at::Tensor& weights, const at::Tensor& bias) ;
std::vector<at::Tensor> ScatterHorizontalBackwardCuda(
    const at::Tensor& grad_output, const std::vector<at::Tensor>& inputs,
    const at::Tensor& weights, const at::Tensor& bias);
std::vector<at::Tensor> ScatterDiagonal1BackwardCuda(
    const at::Tensor& grad_output, const std::vector<at::Tensor>& inputs,
    const at::Tensor& weights, const at::Tensor& bias) ;
std::vector<at::Tensor> ScatterDiagonal2BackwardCuda(
    const at::Tensor& grad_output, const std::vector<at::Tensor>& inputs,
    const at::Tensor& weights, const at::Tensor& bias);

//
// Vertical
//
at::Tensor ScatterVerticalForward(const std::vector<at::Tensor>& inputs,
                                  const at::Tensor& weights,
                                  const at::Tensor& bias) {
  return ScatterVerticalForwardCuda(inputs, weights, bias);
}
std::vector<at::Tensor> ScatterVerticalBackward(
    const at::Tensor& grad_output, const std::vector<at::Tensor>& inputs,
    const at::Tensor& weights, const at::Tensor& bias) {
  return ScatterVerticalBackwardCuda(grad_output, inputs, weights, bias);
}

//
// Horizontal
//
at::Tensor ScatterHorizontalForward(const std::vector<at::Tensor>& inputs,
                                    const at::Tensor& weights,
                                    const at::Tensor& bias) {
  return ScatterHorizontalForwardCuda(inputs, weights, bias);
}
std::vector<at::Tensor> ScatterHorizontalBackward(
    const at::Tensor& grad_output, const std::vector<at::Tensor>& inputs,
    const at::Tensor& weights, const at::Tensor& bias) {
  return ScatterHorizontalBackwardCuda(grad_output, inputs, weights, bias);
}

//
// Diagonal1
//
at::Tensor ScatterDiagonal1Forward(const std::vector<at::Tensor>& inputs,
                                   const at::Tensor& weights,
                                   const at::Tensor& bias) {
  return ScatterDiagonal1ForwardCuda(inputs, weights, bias);
}
std::vector<at::Tensor> ScatterDiagonal1Backward(
    const at::Tensor& grad_output, const std::vector<at::Tensor>& inputs,
    const at::Tensor& weights, const at::Tensor& bias) {
  return ScatterDiagonal1BackwardCuda(grad_output, inputs, weights, bias);
}

//
// Diagonal2
//
at::Tensor ScatterDiagonal2Forward(const std::vector<at::Tensor>& inputs,
                                   const at::Tensor& weights,
                                   const at::Tensor& bias) {
  return ScatterDiagonal2ForwardCuda(inputs, weights, bias);
}
std::vector<at::Tensor> ScatterDiagonal2Backward(
    const at::Tensor& grad_output, const std::vector<at::Tensor>& inputs,
    const at::Tensor& weights, const at::Tensor& bias) {
  return ScatterDiagonal2BackwardCuda(grad_output, inputs, weights, bias);
}
