#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <glad/gl.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <device_launch_parameters.h>
#include "cuda/cuda_utils.cuh"
#include "gl/fluid_simulation.hpp"
#include <iostream>
#include <cmath>

static const int ITER = 30; // Jacobi iterations for diffusion / Poisson solves

// ------------------- helpers -------------------

// Bilinear sampling of a 2D field, used in semi-Lagrangian advection
__device__ __forceinline__ float bilinearSample(const float* grid, float x, float y, int width, int height) {
    // Clamp coordinates slightly inside to avoid out-of-bounds access
    float x_clamped = clampf(x, 0.0f, (float)width - 1.001f);
    float y_clamped = clampf(y, 0.0f, (float)height - 1.001f);

    int x0 = (int)x_clamped;
    int y0 = (int)y_clamped;
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    if (x1 >= width) x1 = width - 1;
    if (y1 >= height) y1 = height - 1;

    float sx = x_clamped - x0;
    float sy = y_clamped - y0;

    // Fetch 4 neighbors
    float v00 = grid[y0 * width + x0];
    float v10 = grid[y0 * width + x1];
    float v01 = grid[y1 * width + x0];
    float v11 = grid[y1 * width + x1];

    // Interpolate along x, then y
    float v0 = v00 * (1.0f - sx) + v10 * sx;
    float v1 = v01 * (1.0f - sx) + v11 * sx;
    return v0 * (1.0f - sy) + v1 * sy;
}

// ------------------- kernels -------------------

// Add swirling force to velocity field (for visualization/demo purposes)
__global__ void addSwirlKernel(float* u, float* v, int width, int height, float time, float strength) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int id = idx(x, y, width);

    // swirl centered at middle of domain
    float cx = ((float)x / width) - 0.5f;
    float cy = ((float)y / height) - 0.5f;
    float r = sqrtf(cx * cx + cy * cy) + 1e-6f;

    // time-dependent swirling pattern
    float angle = 3.0f * time + r * 10.0f;
    float s = sinf(angle);
    float c = cosf(angle);

    // compute small velocity perturbation
    float du = (-cy / r) * 0.05f * s * strength;
    float dv = (cx / r)  * 0.05f * c * strength;

    u[id] += du;
    v[id] += dv;
}

// Add density at center (circular blob)
__global__ void injectDensityKernel(float* dens, int width, int height, float amount, float radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int id = idx(x, y, width);

    int cx = width / 2;
    int cy = height / 2;
    float dx = (float)(x - cx);
    float dy = (float)(y - cy);
    if (dx*dx + dy*dy <= radius*radius) {
        dens[id] += amount;
    }
}

// Jacobi iteration for solving linear systems (e.g., diffusion or Poisson equation)
__global__ void jacobiKernel(const float* b, const float* x_old, float* x_out, float a, float c, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int id = idx(x, y, width);

    // Sample neighbors (clamped)
    float left  = x > 0          ? x_old[id - 1] : x_old[id];
    float right = x < width - 1  ? x_old[id + 1] : x_old[id];
    float up    = y > 0          ? x_old[id - width] : x_old[id];
    float down  = y < height-1   ? x_old[id + width] : x_old[id];

    // Jacobi formula: x_out = (b + a*(sum of neighbors)) / c
    x_out[id] = (b[id] + a * (left + right + up + down)) / c;
}

// Compute divergence = du/dx + dv/dy (negative convention for projection)
__global__ void computeDivergenceKernel(const float* u, const float* v, float* div, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int id = idx(x, y, width);

    float leftU  = (x > 0) ? u[id - 1] : u[id];
    float rightU = (x < width - 1) ? u[id + 1] : u[id];
    float upV    = (y > 0) ? v[id - width] : v[id];
    float downV  = (y < height - 1) ? v[id + width] : v[id];

    // divergence = -0.5 * (du/dx + dv/dy)
    div[id] = -0.5f * ((rightU - leftU) + (downV - upV));
}

// Subtract pressure gradient from velocity to enforce incompressibility
__global__ void projectKernel(float* u, float* v, const float* p, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int id = idx(x, y, width);

    float leftP  = (x > 0) ? p[id - 1] : p[id];
    float rightP = (x < width - 1) ? p[id + 1] : p[id];
    float upP    = (y > 0) ? p[id - width] : p[id];
    float downP  = (y < height - 1) ? p[id + width] : p[id];

    u[id] -= 0.5f * (rightP - leftP);
    v[id] -= 0.5f * (downP - upP);
}

// Semi-Lagrangian advection for scalar or velocity components
// Trace backward along velocity field to interpolate previous value
__global__ void advectKernel(const float* d0, float* d, const float* u, const float* v, int width, int height, float dt) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int id = idx(x, y, width);

    float px = x - u[id] * dt; // backtrace x
    float py = y - v[id] * dt; // backtrace y

    // bilinear interpolation from previous grid
    d[id] = bilinearSample(d0, px, py, width, height);
}

// Add velocity at a point (e.g., from mouse)
__global__ void AddVelocityKernel(float* u, float* v, int width, int height, int cx, int cy, int radius, float2 force) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int dx = x - cx;
    int dy = y - cy;
    if (dx*dx + dy*dy < radius*radius) {
        int id = y * width + x;
        u[id] += force.x;
        v[id] += force.y;
    }
}

// Add density at a point
__global__ void AddDensityKernel(float* dens, int width, int height, int cx, int cy, int radius, float amount) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int dx = x - cx;
    int dy = y - cy;
    if (dx*dx + dy*dy < radius*radius) {
        int id = y * width + x;
        dens[id] += amount;
    }
}

// Convert input grid to uchar4 buffer for OpenGL rendering
__global__ void densityToColorKernel(const float* input, uchar4* buffer, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int id = idx(x, y, width);
    buffer[id] = floatToUchar4(toColor(input[id]));
}

// ------------------- FluidSimulation class -------------------

FluidSimulation::FluidSimulation(int width, int height, float dt)
    : Simulation(width, height), m_dt(dt), m_diffusion(0.005f), m_viscosity(0.005f), m_sourceDensity(2.0f)
{
    size_t size = (size_t)width * height * sizeof(float);
    // Allocate density and velocity grids, pressure, divergence
    cudaMalloc(&m_devCurrent, size);
    cudaMalloc(&m_devNext, size);
    cudaMalloc(&m_velU, size);
    cudaMalloc(&m_velV, size);
    cudaMalloc(&m_velNextU, size);
    cudaMalloc(&m_velNextV, size);
    cudaMalloc(&m_pressure, size);
    cudaMalloc(&m_divergence, size);

    // Zero all grids
    cudaMemset(m_devCurrent, 0, size);
    cudaMemset(m_devNext, 0, size);
    cudaMemset(m_velU, 0, size);
    cudaMemset(m_velV, 0, size);
    cudaMemset(m_velNextU, 0, size);
    cudaMemset(m_velNextV, 0, size);
    cudaMemset(m_pressure, 0, size);
    cudaMemset(m_divergence, 0, size);
}

FluidSimulation::~FluidSimulation() {
    cudaFree(m_velU);
    cudaFree(m_velV);
    cudaFree(m_velNextU);
    cudaFree(m_velNextV);
    cudaFree(m_pressure);
    cudaFree(m_divergence);
}

// Add velocity / density from mouse input
void FluidSimulation::injectFromMouse(int x, int y, float2 force, bool addDensity) {
    dim3 block(16, 16);
    dim3 grid((m_width + block.x - 1) / block.x, (m_height + block.y - 1) / block.y);

    int radius = 10;
    AddVelocityKernel<<<grid, block>>>(m_velU, m_velV, m_width, m_height, x, y, radius, force);

    if (addDensity) {
        AddDensityKernel<<<grid, block>>>(m_devCurrent, m_width, m_height, x, y, radius, 2.0f);
    }
    cudaDeviceSynchronize();
}

// Perform Jacobi iterations for implicit solves (diffusion, Poisson)
void jacobiSolve(const float* b, float* x, float* x_temp, float a, float c, int width, int height) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    for (int i = 0; i < ITER; ++i) {
        jacobiKernel<<<grid, block>>>(b, x, x_temp, a, c, width, height);
        cudaDeviceSynchronize();
        std::swap(x, x_temp);
    }
}

// Generic diffusion using implicit method: solves (field - a*Laplace(field)) = field0
// Parameters:
//   field: input/output field to diffuse in-place
//   temp: temporary buffer for Jacobi iterations
//   width, height: domain dimensions
//   diffusionCoeff: diffusion coefficient (viscosity for velocity, diffusion rate for density)
//   dt: timestep
void diffuse(float* field, float* temp, int width, int height, float diffusionCoeff, float dt) {
    float a = dt * diffusionCoeff;
    float c = 1.0f + 4.0f * a;
    jacobiSolve(field, temp, field, a, c, width, height);
}

// Solve Poisson equation for pressure: Laplacian(p) = divergence
// Used in projection step to enforce incompressibility
void solvePressurePoisson(const float* divergence, float* pressure, float* temp, int width, int height) {
    // Standard Stam coefficients for Poisson solver
    float a = 1.0f;
    float c = 4.0f;

    // Jacobi iterations to solve Laplace(p) = divergence
    for (int i = 0; i < ITER; ++i) {
        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
        jacobiKernel<<<grid, block>>>(divergence, pressure, temp, a, c, width, height);
        cudaDeviceSynchronize();
        std::swap(pressure, temp);
    }
}

// Project velocity field to be divergence-free (incompressible)
// 1) compute divergence
// 2) solve Poisson equation for pressure
// 3) subtract gradient of pressure
void project(float* u, float* v, float* pressure, float* divergence, float* temp, int width, int height) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    // compute divergence
    computeDivergenceKernel<<<grid, block>>>(u, v, divergence, width, height);
    cudaDeviceSynchronize();

    // initialize pressure to zero
    cudaMemset(pressure, 0, (size_t)width * height * sizeof(float));

    // solve for pressure using Poisson solver
    solvePressurePoisson(divergence, pressure, temp, width, height);

    // subtract pressure gradient from velocity
    projectKernel<<<grid, block>>>(u, v, pressure, width, height);
    cudaDeviceSynchronize();
}

// Semi-Lagrangian advection wrapper
void advectField(const float* d0, float* d, const float* u, const float* v, int width, int height, float dt) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    advectKernel<<<grid, block>>>(d0, d, u, v, width, height, dt);
    cudaDeviceSynchronize();
}

// Step the fluid simulation
void FluidSimulation::step(uchar4* pbo) {
    dim3 block(16, 16);
    dim3 grid((m_width + block.x - 1) / block.x, (m_height + block.y - 1) / block.y);

    // 1) add swirling force
    float time = (float)clock() / (float)CLOCKS_PER_SEC;
    addSwirlKernel<<<grid, block>>>(m_velU, m_velV, m_width, m_height, time, 1.0f);
    cudaDeviceSynchronize();

    // 2) diffuse velocity
    diffuse(m_velU, m_velNextU, m_width, m_height, m_viscosity, m_dt);
    diffuse(m_velV, m_velNextV, m_width, m_height, m_viscosity, m_dt);

    // 3) project to divergence-free
    project(m_velU, m_velV, m_pressure, m_divergence, m_velNextU, m_width, m_height);

    // 4) advect velocity (semi-Lagrangian)
    advectField(m_velU, m_velNextU, m_velU, m_velV, m_width, m_height, m_dt);
    advectField(m_velV, m_velNextV, m_velU, m_velV, m_width, m_height, m_dt);

    // swap advected velocities into current
    std::swap(m_velU, m_velNextU);
    std::swap(m_velV, m_velNextV);

    // 5) project again
    project(m_velU, m_velV, m_pressure, m_divergence, m_velNextU, m_width, m_height);

    // 6) inject density at center
    injectDensityKernel<<<grid, block>>>(m_devCurrent, m_width, m_height, m_sourceDensity * m_dt, 6.0f);
    cudaDeviceSynchronize();

    // 7) diffuse density using the same implicit method as velocity
    diffuse(m_devCurrent, m_devNext, m_width, m_height, m_diffusion, m_dt);

    // 8) advect density
    advectField(m_devCurrent, m_devNext, m_velU, m_velV, m_width, m_height, m_dt);

    // 9) render density to PBO
    densityToColorKernel<<<grid, block>>>(m_devNext, pbo, m_width, m_height);
    cudaDeviceSynchronize();

    // Swap grids for next timestep (ping-pong buffer)
    std::swap(m_devCurrent, m_devNext);
}
