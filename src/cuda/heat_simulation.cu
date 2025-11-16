#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <glad/gl.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <device_launch_parameters.h>
#include "cuda/cuda_utils.cuh"
#include "gl/heat_simulation.hpp"
#include <iostream>

__device__ __forceinline__ float laplacian(const float* grid, int x, int y, int width, int height) {
    float center = grid[y * width + x];
    float sum = 0.0f;
    if (x > 0) sum += grid[y * width + (x - 1)] - center;
    if (x < width - 1) sum += grid[y * width + (x + 1)] - center;
    if (y > 0) sum += grid[(y - 1) * width + x] - center;
    if (y < height - 1) sum += grid[(y + 1) * width + x] - center;
    return sum;
}

__global__ void heatStepKernel(const float* current, float* next, int width, int height, float dt, float diffusion, float sourceValue) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    float lap = laplacian(current, x, y, width, height);
    next[idx] = current[idx] + diffusion * lap * dt;

    // Add constant heat source
    if (x == 0) {
        next[idx] = sourceValue;
    }
}

__device__ float4 heatToColor(float value) {
    // simple blue->red map
    return make_float4(saturatef(value), 0.0f, saturatef(1.0f - value), 1.0f);
}

__global__ void heatToColorKernel(const float* heat, uchar4* buffer, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    buffer[idx] = floatToUchar4(heatToColor(heat[idx]));
}

HeatSimulation::HeatSimulation(int width, int height, float dt, float diffusion, float sourceHeat)
    : Simulation(width, height), m_dt(dt), m_diffusion(diffusion), m_sourceHeat(sourceHeat)
{
    size_t size = width * height * sizeof(float);
    cudaMalloc(&m_devCurrent, size);
    cudaMalloc(&m_devNext, size);

    // Zero init
    cudaMemset(m_devCurrent, 0, size);
    cudaMemset(m_devNext, 0, size);
}

HeatSimulation::~HeatSimulation() { }

void HeatSimulation::step(uchar4* pbo) {
    dim3 block(16, 16);
    dim3 grid((m_width + block.x - 1) / block.x, (m_height + block.y - 1) / block.y);

    heatStepKernel<<<grid, block>>>(m_devCurrent, m_devNext, m_width, m_height, m_dt, m_diffusion, m_sourceHeat);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA post-sync error: " << cudaGetErrorString(err) << std::endl;
    }

    heatToColorKernel<<<grid, block>>>(m_devNext, pbo, m_width, m_height);
    cudaDeviceSynchronize();

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA post-sync error: " << cudaGetErrorString(err) << std::endl;
    }

    std::swap(m_devCurrent, m_devNext);
}
