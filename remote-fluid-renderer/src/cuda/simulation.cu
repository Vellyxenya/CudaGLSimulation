#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "cuda/cuda_utils.cuh"

// Define dimensions (can be made dynamic)
constexpr int WIDTH = 800;
constexpr int HEIGHT = 600;

// Example kernel: Write color based on position + time
__global__ void updateBufferKernel(float4* buffer, float time) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= WIDTH || y >= HEIGHT) return;

    int idx = y * WIDTH + x;

    float fx = (float)x / WIDTH;
    float fy = (float)y / HEIGHT;

    buffer[idx] = make_float4(fx, fy, 0.5f + 0.5f * sinf(time), 1.0f);  // RGBA
}

// Public launcher
void launchSimulationKernel(void* devPtr, size_t size, float time) {
    float4* buffer = reinterpret_cast<float4*>(devPtr);

    dim3 block(16, 16);
    dim3 grid((WIDTH + block.x - 1) / block.x,
              (HEIGHT + block.y - 1) / block.y);

    updateBufferKernel<<<grid, block>>>(buffer, time);
    cudaDeviceSynchronize();  // Ensure completion
}
