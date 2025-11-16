#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <glad/gl.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <device_launch_parameters.h>
#include "cuda/cuda_utils.cuh"
#include "gl/heat_simulation.hpp"
#include <iostream>

// Compute the discrete Laplacian for a 2D grid at (x, y)
// The Laplacian approximates the second spatial derivative, ∇²u, which governs diffusion.
// Each neighbor contributes the difference from the center cell.
__device__ __forceinline__ float laplacian(const float* grid, int x, int y, int width, int height) {
    float center = grid[y * width + x];
    float sum = 0.0f;

    if (x > 0) sum += grid[y * width + (x - 1)] - center;
    if (x < width - 1) sum += grid[y * width + (x + 1)] - center;
    if (y > 0) sum += grid[(y - 1) * width + x] - center;
    if (y < height - 1) sum += grid[(y + 1) * width + x] - center;

    return sum;
}

// Advance one timestep of the heat simulation using explicit Euler method:
// newValue = oldValue + dt * diffusion * Laplacian(oldValue)
// This models heat spreading from each cell to neighbors.
__global__ void heatStepKernel(const float* current, float* next, int width, int height, float dt, float diffusion, float sourceValue) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    float lap = laplacian(current, x, y, width, height);

    // Update value according to diffusion PDE: u_t = D * ∇²u
    next[idx] = current[idx] + diffusion * lap * dt;

    // Apply constant heat source at left boundary (x == 0)
    // This ensures the simulation always has a “hot edge”
    if (x == 0) {
        next[idx] = sourceValue;
    }
}

// Map heat value to RGB color for visualization
// Simple linear mapping: low values -> blue, high values -> red
__device__ float4 heatToColor(float value) {
    // Saturate value to [0,1] to avoid color overflow
    return make_float4(saturatef(value), 0.0f, saturatef(1.0f - value), 1.0f);
}

// Convert entire heat grid to uchar4 buffer for OpenGL rendering
__global__ void heatToColorKernel(const float* heat, uchar4* buffer, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    buffer[idx] = floatToUchar4(heatToColor(heat[idx]));
}

// -----------------------------------------------------------------------------
// Constructor: allocate device memory and zero-initialize grids
// -----------------------------------------------------------------------------
HeatSimulation::HeatSimulation(int width, int height, float dt, float diffusion, float sourceHeat)
    : Simulation(width, height), m_dt(dt), m_diffusion(diffusion), m_sourceHeat(sourceHeat)
{
    size_t size = width * height * sizeof(float);
    cudaMalloc(&m_devCurrent, size); // Current heat grid
    cudaMalloc(&m_devNext, size);    // Next heat grid

    // Initialize grids to zero temperature
    cudaMemset(m_devCurrent, 0, size);
    cudaMemset(m_devNext, 0, size);
}

// -----------------------------------------------------------------------------
// Destructor: free device memory
// -----------------------------------------------------------------------------
HeatSimulation::~HeatSimulation() {
    cudaFree(m_devCurrent);
    cudaFree(m_devNext);
}

// -----------------------------------------------------------------------------
// Step the simulation and write output to PBO
// -----------------------------------------------------------------------------
void HeatSimulation::step(uchar4* pbo) {
    // CUDA kernel launch dimensions: 16x16 threads per block
    dim3 block(16, 16);
    dim3 grid((m_width + block.x - 1) / block.x, (m_height + block.y - 1) / block.y);

    // 1) Advance simulation one timestep
    heatStepKernel<<<grid, block>>>(m_devCurrent, m_devNext, m_width, m_height, m_dt, m_diffusion, m_sourceHeat);
    cudaDeviceSynchronize();

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA post-sync error: " << cudaGetErrorString(err) << std::endl;
    }

    // 2) Convert heat values to RGBA colors for rendering
    heatToColorKernel<<<grid, block>>>(m_devNext, pbo, m_width, m_height);
    cudaDeviceSynchronize();

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA post-sync error: " << cudaGetErrorString(err) << std::endl;
    }

    // Swap grids for next timestep (ping-pong buffer)
    std::swap(m_devCurrent, m_devNext);
}
