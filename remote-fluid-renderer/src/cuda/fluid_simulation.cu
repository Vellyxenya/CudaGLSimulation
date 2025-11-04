#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <glad/gl.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <device_launch_parameters.h>
#include "cuda/cuda_utils.cuh"
#include "gl/fluid_simulation.hpp"
#include <iostream>

// ------------------- Utilities -------------------

__device__ __forceinline__ float sample(const float* grid, int x, int y, int width, int height) {
    x = max(0, min(width - 1, x));
    y = max(0, min(height - 1, y));
    return grid[y * width + x];
}

__device__ __forceinline__ float bilinear(const float* grid, float x, float y, int width, int height) {
    x = fmaxf(0.0f, fminf(x, width - 1.001f));
    y = fmaxf(0.0f, fminf(y, height - 1.001f));

    int x0 = (int)x;
    int y0 = (int)y;
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    float sx = x - x0;
    float sy = y - y0;

    float val00 = grid[y0 * width + x0];
    float val10 = grid[y0 * width + x1];
    float val01 = grid[y1 * width + x0];
    float val11 = grid[y1 * width + x1];

    float val0 = val00 * (1 - sx) + val10 * sx;
    float val1 = val01 * (1 - sx) + val11 * sx;
    return val0 * (1 - sy) + val1 * sy;
}

// ------------------- Kernels -------------------

// Diffuse & advect density
__global__ void advectDensityKernel(
    const float* density, float* nextDensity,
    const float* u, const float* v,
    int width, int height, float dt)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;

    // Trace backward along velocity
    float px = x - u[idx] * dt;
    float py = y - v[idx] * dt;

    nextDensity[idx] = bilinear(density, px, py, width, height);
}

// Simple velocity update (adds swirl and diffusion)
__global__ void updateVelocityKernel(
    float* u, float* v, float* nextU, float* nextV,
    int width, int height, float dt, float diffusion, float time)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;

    // Small swirl field
    float cx = (float)x / width - 0.5f;
    float cy = (float)y / height - 0.5f;
    float swirl = 2.0f * sinf(3.0f * time + cx * 10.0f) * cosf(3.0f * time + cy * 10.0f);

    float u_next = u[idx] + swirl * dt;
    float v_next = v[idx] - swirl * dt;

    // Simple velocity diffusion
    float lapU = sample(u, x - 1, y, width, height) + sample(u, x + 1, y, width, height)
        + sample(u, x, y - 1, width, height) + sample(u, x, y + 1, width, height)
        - 4.0f * u[idx];
    float lapV = sample(v, x - 1, y, width, height) + sample(v, x + 1, y, width, height)
        + sample(v, x, y - 1, width, height) + sample(v, x, y + 1, width, height)
        - 4.0f * v[idx];

    nextU[idx] = u_next + diffusion * lapU * dt;
    nextV[idx] = v_next + diffusion * lapV * dt;
}

// Map density to color
__device__ float4 densityToColor(float val) {
    return make_float4(
        saturatef(val),
        saturatef(val * 0.3f),
        saturatef(1.0f - val),
        1.0f
    );
}

__global__ void densityToColorKernel(const float* density, uchar4* buffer, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int idx = y * width + x;
    buffer[idx] = floatToUchar4(densityToColor(density[idx]));
}

__global__ void injectDensityKernel(float* grid, int width, int height, float dt, float amount) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int cx = width / 2;
    int cy = height / 2;

    int radius = 5; // number of pixels around the center
    if ((x - cx) * (x - cx) + (y - cy) * (y - cy) < radius * radius) {
        grid[y * width + x] += amount * dt;
    }
}

// ------------------- Class implementation -------------------

FluidSimulation::FluidSimulation(int width, int height, float dt)
    : m_width(width), m_height(height), m_dt(dt), m_diffusion(1.4f), m_sourceDensity(15.0f)
{
    size_t size = width * height * sizeof(float);
    cudaMalloc(&m_devCurrent, size);
    cudaMalloc(&m_devNext, size);
    cudaMalloc(&m_velNextU, size);
    cudaMalloc(&m_velNextV, size);

    cudaMemset(m_devCurrent, 0, size);
    cudaMemset(m_devNext, 0, size);
    cudaMemset(m_velNextU, 0, size);
    cudaMemset(m_velNextV, 0, size);

    cudaMalloc(&m_velU, size);
    cudaMalloc(&m_velV, size);
    // Allocate host temporary arrays to initialize velocities
    float* h_velU = new float[width * height];
    float* h_velV = new float[width * height];

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float dx = (float)x / width - 0.5f;
            float dy = (float)y / height - 0.5f;
            float radius = sqrtf(dx * dx + dy * dy) + 1e-5f;
            h_velU[y * width + x] = -dy / radius * 2.0f; // circular flow
            h_velV[y * width + x] = dx / radius * 2.0f;
        }
    }

    // Copy to GPU
    cudaMemcpy(m_velU, h_velU, size, cudaMemcpyHostToDevice);
    cudaMemcpy(m_velV, h_velV, size, cudaMemcpyHostToDevice);

    delete[] h_velU;
    delete[] h_velV;
}

FluidSimulation::~FluidSimulation() {
    cudaFree(m_devCurrent);
    cudaFree(m_devNext);
    cudaFree(m_velU);
    cudaFree(m_velV);
    cudaFree(m_velNextU);
    cudaFree(m_velNextV);
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

    // 1) Update velocity
    float time = (float)clock() / CLOCKS_PER_SEC;
    updateVelocityKernel<<<grid, block>>>(m_velU, m_velV, m_velNextU, m_velNextV, m_width, m_height, m_dt, m_diffusion, time);
    cudaDeviceSynchronize();

    // Swap velocity
    std::swap(m_velU, m_velNextU);
    std::swap(m_velV, m_velNextV);

    // 2) Inject density at left wall
    injectDensityKernel<<<grid, block>>>(m_devCurrent, m_width, m_height, m_dt, m_sourceDensity * m_dt);
    cudaDeviceSynchronize();

    // 3) Advect density
    advectDensityKernel<<<grid, block>>>(m_devCurrent, m_devNext, m_velU, m_velV, m_width, m_height, m_dt);
    cudaDeviceSynchronize();

    // 4) Map density to color
    densityToColorKernel<<<grid, block>>>(m_devNext, pbo, m_width, m_height);
    cudaDeviceSynchronize();

    // Swap density
    std::swap(m_devCurrent, m_devNext);
}
