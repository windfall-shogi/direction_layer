#include <torch/torch.h>

#include <vector>

#include "check.hpp"

std::vector<at::Tensor> OnehotForwardCuda(at::Tensor input);

std::vector<at::Tensor> OnehotForward(at::Tensor input){
    CHECK_INPUT(input);
    return OnehotForwardCuda(input);
}

// 宣言
std::vector<at::Tensor> GatherVerticalForward(const at::Tensor &input,
                                              const at::Tensor &weights,
                                              const at::Tensor &bias);
std::vector<at::Tensor> GatherVerticalBackward(
    const std::vector<at::Tensor> &grad_output, const at::Tensor &input,
    const at::Tensor &weights, const at::Tensor &bias);
std::vector<at::Tensor> GatherHorizontalForward(const at::Tensor &input,
                                                const at::Tensor &weights,
                                                const at::Tensor &bias);
std::vector<at::Tensor> GatherHorizontalBackward(
    const std::vector<at::Tensor> &grad_outputs, const at::Tensor &input,
    const at::Tensor &weights, const at::Tensor &bias);
std::vector<at::Tensor> GatherDiagonal1Forward(const at::Tensor &input,
                                               const at::Tensor &weights,
                                               const at::Tensor &bias);
std::vector<at::Tensor> GatherDiagonal1Backward(
    const std::vector<at::Tensor> &grad_outputs, const at::Tensor &input,
    const at::Tensor &weights, const at::Tensor &bias);
std::vector<at::Tensor> GatherDiagonal2Forward(const at::Tensor &input,
                                               const at::Tensor &weights,
                                               const at::Tensor &bias);
std::vector<at::Tensor> GatherDiagonal2Backward(
    const std::vector<at::Tensor> &grad_outputs, const at::Tensor &input,
    const at::Tensor &weights, const at::Tensor &bias);

at::Tensor ScatterVerticalForward(const std::vector<at::Tensor>& inputs,
                                  const at::Tensor& weights,
                                  const at::Tensor& bias);
std::vector<at::Tensor> ScatterVerticalBackward(
    const at::Tensor& grad_output, const std::vector<at::Tensor>& inputs,
    const at::Tensor& weights, const at::Tensor& bias);
at::Tensor ScatterHorizontalForward(const std::vector<at::Tensor>& inputs,
                                    const at::Tensor& weights,
                                    const at::Tensor& bias);
std::vector<at::Tensor> ScatterHorizontalBackward(
    const at::Tensor& grad_output, const std::vector<at::Tensor>& inputs,
    const at::Tensor& weights, const at::Tensor& bias);
at::Tensor ScatterDiagonal1Forward(const std::vector<at::Tensor>& inputs,
                                   const at::Tensor& weights,
                                   const at::Tensor& bias);
std::vector<at::Tensor> ScatterDiagonal1Backward(
    const at::Tensor& grad_output, const std::vector<at::Tensor>& inputs,
    const at::Tensor& weights, const at::Tensor& bias);
at::Tensor ScatterDiagonal2Forward(const std::vector<at::Tensor>& inputs,
                                   const at::Tensor& weights,
                                   const at::Tensor& bias);
std::vector<at::Tensor> ScatterDiagonal2Backward(
    const at::Tensor& grad_output, const std::vector<at::Tensor>& inputs,
    const at::Tensor& weights, const at::Tensor& bias);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  auto m_onehot = m.def_submodule("onehot", "A submodule of 'area_cuda'");
  m_onehot.def("forward", &OnehotForward, "Onehot forward (CUDA)");

  auto m_gather = m.def_submodule("gather", "A submodule of 'area_cuda'");
  m_gather.def("forward_vertical", &GatherVerticalForward,
               "Gather forward (CUDA)");
  m_gather.def("backward_vertical", &GatherVerticalBackward,
               "Gather backward (CUDA)");
  m_gather.def("forward_horizontal", &GatherHorizontalForward,
               "Gather forward (CUDA)");
  m_gather.def("backward_horizontal", &GatherHorizontalBackward,
               "Gather backward (CUDA)");
  m_gather.def("forward_diagonal1", &GatherDiagonal1Forward,
               "Gather forward (CUDA)");
  m_gather.def("backward_diagonal1", &GatherDiagonal1Backward,
               "Gather backward (CUDA)");
  m_gather.def("forward_diagonal2", &GatherDiagonal2Forward,
               "Gather forward (CUDA)");
  m_gather.def("backward_diagonal2", &GatherDiagonal2Backward,
               "Gather backward (CUDA)");

  auto m_scatter = m.def_submodule("scatter", "A submodule of 'area_cuda'");
  m_scatter.def("forward_vertical", &ScatterVerticalForward,
                "Scatter forward (CUDA)");
  m_scatter.def("backward_vertical", &ScatterVerticalBackward,
                "Scatter backward (CUDA)");
  m_scatter.def("forward_horizontal", &ScatterHorizontalForward,
                "Scatter forward (CUDA)");
  m_scatter.def("backward_horizontal", &ScatterHorizontalBackward,
                "Scatter backward (CUDA)");
  m_scatter.def("forward_diagonal1", &ScatterDiagonal1Forward,
                "Scatter forward (CUDA)");
  m_scatter.def("backward_diagonal1", &ScatterDiagonal1Backward,
                "Scatter backward (CUDA)");
  m_scatter.def("forward_diagonal2", &ScatterDiagonal2Forward,
                "Scatter forward (CUDA)");
  m_scatter.def("backward_diagonal2", &ScatterDiagonal2Backward,
                "Scatter backward (CUDA)");
}
