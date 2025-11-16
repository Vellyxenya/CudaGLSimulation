// src/cuda/fluid_simulation.cu
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

static const int ITER = 20; // Jacobi iterations for diffusion / Poisson

// ------------------- helpers -------------------
__device__ __forceinline__ int idx(int x, int y, int w) { return y * w + x; }

__device__ __forceinline__ float clampf(float v, float a, float b) {
    return v < a ? a : (v > b ? b : v);
}

// safe sampling for bilinear interpolation (clamp coordinates to valid interpolation range)
__device__ __forceinline__ float bilinear_sample(const float* grid, float x, float y, int width, int height) {
    // clamp coordinates slightly inside to allow x0,x1 to be valid
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

    float v00 = grid[y0 * width + x0];
    float v10 = grid[y0 * width + x1];
    float v01 = grid[y1 * width + x0];
    float v11 = grid[y1 * width + x1];

    float v0 = v00 * (1.0f - sx) + v10 * sx;
    float v1 = v01 * (1.0f - sx) + v11 * sx;
    return v0 * (1.0f - sy) + v1 * sy;
}

// ------------------- kernels -------------------

// add a small swirling force to velocity (source)
__global__ void add_swirl_kernel(float* u, float* v, int width, int height, float time, float strength) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int id = idx(x, y, width);

    // simple time-varying swirl centered in the middle
    float cx = ((float)x / width) - 0.5f;
    float cy = ((float)y / height) - 0.5f;
    float r = sqrtf(cx * cx + cy * cy) + 1e-6f;
    // swirl pattern that depends on time and position
    float angle = 3.0f * time + r * 10.0f;
    float s = sinf(angle);
    float c = cosf(angle);

    float du = (-cy / r) * 0.05f * s * strength;
    float dv = (cx / r)  * 0.05f * c * strength;

    u[id] += du;
    v[id] += dv;
}

// add density at center (circular)
__global__ void inject_density_kernel(float* dens, int width, int height, float amount, float radius) {
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

// Jacobi relaxation for linear solve: x = (b + a * (xL+xR+xU+xD)) / c
__global__ void jacobi_kernel(const float* b, const float* x_old, float* x_out, float a, float c, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int id = idx(x, y, width);

    // sample neighbors with clamping
    float left  = x > 0          ? x_old[id - 1] : x_old[id];
    float right = x < width - 1  ? x_old[id + 1] : x_old[id];
    float up    = y > 0          ? x_old[id - width] : x_old[id];
    float down  = y < height-1   ? x_old[id + width] : x_old[id];

    x_out[id] = (b[id] + a * (left + right + up + down)) / c;
}

// compute divergence = -0.5*(du/dx + dv/dy)
__global__ void compute_divergence_kernel(const float* u, const float* v, float* div, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int id = idx(x, y, width);

    float leftU  = (x > 0) ? u[id - 1] : u[id];
    float rightU = (x < width - 1) ? u[id + 1] : u[id];
    float upV    = (y > 0) ? v[id - width] : v[id];
    float downV  = (y < height - 1) ? v[id + width] : v[id];

    div[id] = -0.5f * ((rightU - leftU) + (downV - upV));
}

// subtract pressure gradient from velocity
__global__ void project_kernel(float* u, float* v, const float* p, int width, int height) {
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

// semi-Lagrangian advect for a scalar field (density or velocity component)
__global__ void advect_kernel(const float* d0, float* d, const float* u, const float* v, int width, int height, float dt) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int id = idx(x, y, width);

    // backtrace position (use velocity at current cell)
    float px = x - u[id] * dt;
    float py = y - v[id] * dt;

    d[id] = bilinear_sample(d0, px, py, width, height);
}

__global__ void AddVelocityKernel(
    float* u, float* v,
    int width, int height,
    int cx, int cy, int radius,
    float2 force)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int dx = x - cx;
    int dy = y - cy;
    if (dx * dx + dy * dy < radius * radius) {
        int idx = y * width + x;
        u[idx] += force.x;
        v[idx] += force.y;
    }
}

__global__ void AddDensityKernel(float* dens, int width, int height,
    int cx, int cy, int radius, float amount) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int dx = x - cx;
    int dy = y - cy;
    if (dx * dx + dy * dy < radius * radius) {
        int idx = y * width + x;
        dens[idx] += amount;
    }
}

// map density to color
__device__ float4 density_to_color(float v) {
    return make_float4(saturatef(v), saturatef(v*0.3f), saturatef(1.0f - v), 1.0f);
}

__global__ void density_to_color_kernel(const float* dens, uchar4* out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int id = idx(x, y, width);
    out[id] = floatToUchar4(density_to_color(dens[id]));
}

// ------------------- Heat / FluidSimulation class -------------------

FluidSimulation::FluidSimulation(int width, int height, float dt)
    : Simulation(width, height), m_dt(dt), m_diffusion(0.0005f), m_viscosity(0.0005f), m_sourceDensity(3.0f)
{
    size_t size = (size_t)width * height * sizeof(float);
    // density
    cudaMalloc(&m_devCurrent, size);
    cudaMalloc(&m_devNext, size);
    // velocity
    cudaMalloc(&m_velU, size);
    cudaMalloc(&m_velV, size);
    cudaMalloc(&m_velNextU, size);
    cudaMalloc(&m_velNextV, size);
    // pressure and divergence buffers for projection
    cudaMalloc(&m_pressure, size);
    cudaMalloc(&m_divergence, size);

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

// helper: perform Jacobi iterations (b is RHS, x_old input, x_out will hold next)
void jacobi_solve(const float* b, float* x, float* x_temp, float a, float c, int width, int height) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    for (int i = 0; i < ITER; ++i) {
        jacobi_kernel<<<grid, block>>>(b, x, x_temp, a, c, width, height);
        cudaDeviceSynchronize();
        // swap
        std::swap(x, x_temp);
    }
    // final result in x (if swapped odd number of times then x contains latest)
}

// velocity diffusion (implicit): solve (u - a * Laplacian(u)) = u0  with a = dt * viscosity
void diffuse_velocity(float* u, float* u_temp, int width, int height, float visc, float dt) {
    float a = dt * visc;
    float c = 1.0f + 4.0f * a;
    jacobi_solve(u, u_temp, u, a, c, width, height); // note: reusing buffers (x and x_temp)
}

// project velocity to be divergence-free:
// 1) compute divergence = -0.5*(du/dx + dv/dy)
// 2) solve Poisson: Laplacian(p) = divergence  (Jacobi)
// 3) u -= 0.5*(dp/dx), v -= 0.5*(dp/dy)
void project(float* u, float* v, float* pressure, float* divergence, float* temp, int width, int height) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    // compute divergence
    compute_divergence_kernel<<<grid, block>>>(u, v, divergence, width, height);
    cudaDeviceSynchronize();

    // initialize pressure to zero
    cudaMemset(pressure, 0, (size_t)width * height * sizeof(float));

    // solve for pressure (Poisson) using Jacobi: Laplacian(p) = divergence
    // rewrite as p - a * Laplacian(p) = b with a = 1 and c = 1 + 4*1 = 5 (Jacobi form)
    // but standard Stam uses a = 1, c = 4
    float a = 1.0f;
    float c = 4.0f;

    // use jacobi iterations: x_old (pressure) and temp as buffer; b = divergence
    for (int i = 0; i < ITER; ++i) {
        jacobi_kernel<<<grid, block>>>(divergence, pressure, temp, a, c, width, height);
        cudaDeviceSynchronize();
        std::swap(pressure, temp);
    }

    // subtract gradient
    project_kernel<<<grid, block>>>(u, v, pressure, width, height);
    cudaDeviceSynchronize();
}

// semi-Lagrangian advection wrapper for scalar fields
void advect_field(const float* d0, float* d, const float* u, const float* v, int width, int height, float dt) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    advect_kernel<<<grid, block>>>(d0, d, u, v, width, height, dt);
    cudaDeviceSynchronize();
}

void FluidSimulation::step(uchar4* pbo) {
    dim3 block(16, 16);
    dim3 grid((m_width + block.x - 1) / block.x, (m_height + block.y - 1) / block.y);

    // 1) add time-varying swirl force to velocity
    float time = (float)clock() / (float)CLOCKS_PER_SEC;
    add_swirl_kernel<<<grid, block>>>(m_velU, m_velV, m_width, m_height, time, 1.0f);
    cudaDeviceSynchronize();

    // 2) diffuse velocity (viscosity) implicitly
    // use m_velNextU / m_velNextV as temp
    diffuse_velocity(m_velU, m_velNextU, m_width, m_height, m_viscosity, m_dt);
    diffuse_velocity(m_velV, m_velNextV, m_width, m_height, m_viscosity, m_dt);

    // 3) project velocity to be divergence-free
    project(m_velU, m_velV, m_pressure, m_divergence, m_velNextU, m_width, m_height);

    // 4) advect velocity by itself (semi-Lagrangian)
    advect_field(m_velU, m_velNextU, m_velU, m_velV, m_width, m_height, m_dt);
    advect_field(m_velV, m_velNextV, m_velU, m_velV, m_width, m_height, m_dt);

    // swap advected velocities into current
    std::swap(m_velU, m_velNextU);
    std::swap(m_velV, m_velNextV);

    // 5) project again
    project(m_velU, m_velV, m_pressure, m_divergence, m_velNextU, m_width, m_height);

    // 6) inject density at center
    inject_density_kernel<<<grid, block>>>(m_devCurrent, m_width, m_height, m_sourceDensity * m_dt, 6.0f);
    cudaDeviceSynchronize();

    // 7) optional: diffuse density (small)
    {
        float a = m_dt * m_diffusion;
        float c = 1.0f + 4.0f * a;
        // perform Jacobi iterations: b = dens, x = dens, temp = m_devNext
        // we reuse m_devNext as temp buffer
        for (int i = 0; i < ITER; ++i) {
            jacobi_kernel<<<grid, block>>>(m_devCurrent, m_devCurrent, m_devNext, a, c, m_width, m_height);
            cudaDeviceSynchronize();
            std::swap(m_devCurrent, m_devNext);
        }
    }

    // 8) advect density using velocity field
    advect_field(m_devCurrent, m_devNext, m_velU, m_velV, m_width, m_height, m_dt);

    // 9) render density to PBO
    density_to_color_kernel<<<grid, block>>>(m_devNext, pbo, m_width, m_height);
    cudaDeviceSynchronize();

    // swap density buffers
    std::swap(m_devCurrent, m_devNext);
}
