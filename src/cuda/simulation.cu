#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <glad/gl.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <device_launch_parameters.h>
#include "cuda/cuda_utils.cuh"
#include "gl/simulation.hpp"
#include <iostream>

// Kernel that fills the PBO with a gradient pattern that changes colors over time
__global__ void fillKernel(uchar4* buf, int w, int h, float t) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int idx = y * w + x;

    float fx = (float)x / w;
    float fy = (float)y / h;

    float4 color = make_float4(fx, fy, 0.5f + 0.5f * sinf(t), 1.0f); // RGBA
    buf[idx] = floatToUchar4(color);
}

// Step the simulation and write output to PBO
void step(uchar4* pbo, int width, int height, float time) {
    // CUDA kernel launch dimensions: 16x16 threads per block
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    // Launch kernel
    fillKernel<<<grid, block>>>(pbo, width, height, time);
    cudaDeviceSynchronize();

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA post-sync error: " << cudaGetErrorString(err) << std::endl;
    }
}
