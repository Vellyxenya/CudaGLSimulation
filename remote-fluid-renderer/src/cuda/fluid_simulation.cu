#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <glad/gl.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <device_launch_parameters.h>
#include "cuda/cuda_utils.cuh"
#include "gl/fluid_simulation.hpp"
#include <iostream>

// Compute 2D laplacian for diffusion
__device__ __forceinline__ float laplacian(const float* grid, int x, int y, int width, int height) {
    float center = grid[y * width + x];
    float sum = 0.0f;
    if (x > 0) sum += grid[y * width + (x - 1)] - center;
    if (x < width - 1) sum += grid[y * width + (x + 1)] - center;
    if (y > 0) sum += grid[(y - 1) * width + x] - center;
    if (y < height - 1) sum += grid[(y + 1) * width + x] - center;
    return sum;
}

// Advect values (simple semi-Lagrangian)
__device__ __forceinline__ float advect(const float* grid, int x, int y, int width, int height, float vx, float vy) {
    float px = x - vx;
    float py = y - vy;

    int x0 = max(0, min(width - 1, (int)px));
    int y0 = max(0, min(height - 1, (int)py));
    int x1 = max(0, min(width - 1, x0 + 1));
    int y1 = max(0, min(height - 1, y0 + 1));

    float sx = px - x0;
    float sy = py - y0;

    float val00 = grid[y0 * width + x0];
    float val10 = grid[y0 * width + x1];
    float val01 = grid[y1 * width + x0];
    float val11 = grid[y1 * width + x1];

    float val0 = val00 * (1.0f - sx) + val10 * sx;
    float val1 = val01 * (1.0f - sx) + val11 * sx;

    return val0 * (1.0f - sy) + val1 * sy;
}

// Kernel to step fluid simulation
__global__ void fluidStepKernel(const float* current, float* next, int width, int height, float dt, float diffusion) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;

    // Simple diffusion
    float diff = current[idx] + diffusion * laplacian(current, x, y, width, height) * dt;

    // Simple circular advection: move right + down
    float vx = 0.5f; // velocity in x
    float vy = 0.3f; // velocity in y
    next[idx] = advect(current, x, y, width, height, vx, vy) * 0.99f + diff * 0.01f;

    // Add a left wall source
    if (x == 0) next[idx] += 5.0f * dt;
}

// Map fluid density to color
__device__ float4 fluidToColor(float val) {
    return make_float4(saturatef(val), saturatef(val * 0.2f), saturatef(1.0f - val), 1.0f);
}

__global__ void fluidToColorKernel(const float* grid, uchar4* buffer, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    buffer[idx] = floatToUchar4(fluidToColor(grid[idx]));
}

// ----------------- Class implementation -----------------

FluidSimulation::FluidSimulation(int width, int height, float dt)
    : m_width(width), m_height(height), m_dt(dt), m_diffusion(0.2f), m_sourceHeat(5.0f)
{
    size_t size = width * height * sizeof(float);
    cudaMalloc(&m_devCurrent, size);
    cudaMalloc(&m_devNext, size);
    cudaMemset(m_devCurrent, 0, size);
    cudaMemset(m_devNext, 0, size);
}

FluidSimulation::~FluidSimulation() {
    cudaFree(m_devCurrent);
    cudaFree(m_devNext);
}

void FluidSimulation::setInitialCondition(const float* hostData) {
    size_t size = m_width * m_height * sizeof(float);
    if (hostData)
        cudaMemcpy(m_devCurrent, hostData, size, cudaMemcpyHostToDevice);
    else
        cudaMemset(m_devCurrent, 0, size);
}

void FluidSimulation::step(uchar4* pbo) {
    dim3 block(16, 16);
    dim3 grid((m_width + block.x - 1) / block.x, (m_height + block.y - 1) / block.y);

    fluidStepKernel << <grid, block >> > (m_devCurrent, m_devNext, m_width, m_height, m_dt, m_diffusion);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cerr << "CUDA post-sync error: " << cudaGetErrorString(err) << std::endl;

    fluidToColorKernel << <grid, block >> > (m_devNext, pbo, m_width, m_height);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cerr << "CUDA post-sync error: " << cudaGetErrorString(err) << std::endl;

    std::swap(m_devCurrent, m_devNext);
}