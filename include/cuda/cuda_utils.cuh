#pragma once
// -----------------------------------------------------------------------------
// cuda_utils.cuh
// Small collection of inline CUDA helper functions used across the simulation.
// These functions run on the GPU (`__device__`) and are inlined for performance.
// -----------------------------------------------------------------------------

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// -----------------------------------------------------------------------------
// saturatef(x)
// Clamps a float to the [0, 1] range.
// __forceinline__ ensures the compiler always inlines the function,
// reducing call overhead inside inner GPU loops.
// -----------------------------------------------------------------------------
__device__ __forceinline__ float saturatef(float x) {
    return fminf(fmaxf(x, 0.0f), 1.0f);
}

// -----------------------------------------------------------------------------
// floatToUchar4(float4 f)
// Converts a float4 where each component is expected in [0,1]
// into a uchar4 where each component is in [0,255].
// The result can be written directly to the CUDA-mapped PBO the renderer uses.
// -----------------------------------------------------------------------------
__device__ __forceinline__ uchar4 floatToUchar4(float4 f) {
    return make_uchar4(
        saturatef(f.x) * 255.0f,  // R
        saturatef(f.y) * 255.0f,  // G
        saturatef(f.z) * 255.0f,  // B
        saturatef(f.w) * 255.0f   // A
    );
}

// Linear index from 2D coordinates
__device__ __forceinline__ int idx(int x, int y, int w) { return y * w + x; }

// clamp float value between a and b
__device__ __forceinline__ float clampf(float v, float a, float b) {
    return v < a ? a : (v > b ? b : v);
}

// Map value to RGB color for visualization
// Simple linear mapping: low values -> blue, high values -> red
__device__ __forceinline__ float4 toColor(float value) {
    // Saturate value to [0,1] to avoid color overflow
    return make_float4(saturatef(value),  saturatef(value*0.3f), saturatef(1.0f - value), 1.0f);
}