#include <torch/extension.h>

#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")

torch::Tensor cuda_double2float_forward(
    torch::Tensor input, const std::string direction);

torch::Tensor double2float_foward(
    torch::Tensor input, const std::string direction) {
  TORCH_CHECK((direction == "down") || (direction == "up"), "Unsupported direction, must be down or up.");
  TORCH_CHECK(input.type().scalarType() == torch::ScalarType::Double, "This function only supports DoubleTensor as inputs.");
  CHECK_CUDA(input);
  return cuda_double2float_forward(input, direction);
}

/* 
 * Usage: double2float(tensor, direction)
 * "tensor" must be a DoubleTensor on GPU.
 * "direction" is a string, can be "up" (round up) or "down" (round down).
 */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("double2float", &double2float_foward, "Convert double to float with rounding direction control (direction = 'up' or 'down').");
}
