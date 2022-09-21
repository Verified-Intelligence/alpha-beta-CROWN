#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

__global__ void cuda_double2float_rd_kernel(const double* __restrict__ inputs,
    float* __restrict__ outputs, const size_t tensor_size) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < tensor_size) {
    outputs[idx] = __double2float_rd(inputs[idx]);
  }
}

__global__ void cuda_double2float_ru_kernel(const double* __restrict__ inputs,
    float* __restrict__ outputs, const size_t tensor_size) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < tensor_size) {
    outputs[idx] = __double2float_ru(inputs[idx]);
  }
}

torch::Tensor cuda_double2float_forward(torch::Tensor input,
    const std::string direction) {
  auto total_elem = input.numel();
  auto output = torch::empty_like(input, torch::ScalarType::Float);

  const int threads = 1024;
  const int blocks = (total_elem + threads - 1) / threads;
  
  if (direction == "down") {
    cuda_double2float_rd_kernel<<<blocks, threads>>>(input.data<double>(), output.data<float>(), total_elem);
  }
  else {
    cuda_double2float_ru_kernel<<<blocks, threads>>>(input.data<double>(), output.data<float>(), total_elem);
  }
  return output;
}

