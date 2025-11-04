#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <glad/gl.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "cuda/cuda_utils.cuh"
#include <iostream>

// Define dimensions (can be made dynamic)
constexpr int WIDTH = 800;
constexpr int HEIGHT = 600;

__device__ __forceinline__ float saturatef(float x) {
    return fminf(fmaxf(x, 0.0f), 1.0f);
}

__device__ uchar4 floatToUchar4(float4 f) {
    return make_uchar4(
        saturatef(f.x) * 255.0f,
        saturatef(f.y) * 255.0f,
        saturatef(f.z) * 255.0f,
        saturatef(f.w) * 255.0f
    );
}

// Example kernel: Write color based on position + time
__global__ void updateBufferKernel(uchar4* buffer, float time) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= WIDTH || y >= HEIGHT) return;

    int idx = y * WIDTH + x;

    float fx = (float)x / WIDTH;
    float fy = (float)y / HEIGHT;

    float4 color = make_float4(fx, fy, 0.5f + 0.5f * sinf(time), 1.0f); // RGBA
    buffer[idx] = floatToUchar4(color);
}

// Public launcher
void launchSimulationKernel(void* devPtr, size_t size, float time) {
    uchar4* buffer = reinterpret_cast<uchar4*>(devPtr);

    dim3 block(16, 16);
    dim3 grid((WIDTH + block.x - 1) / block.x,
              (HEIGHT + block.y - 1) / block.y);

    updateBufferKernel<<<grid, block>>>(buffer, time);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA launch error: " << cudaGetErrorString(err) << std::endl;
    }

    // For production, consider using async streams instead of cudaDeviceSynchronize() once you integrate with NVENC.
    cudaDeviceSynchronize();  // Ensure completion

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA post-sync error: " << cudaGetErrorString(err) << std::endl;
    }
}
