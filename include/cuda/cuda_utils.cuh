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
// Equivalent to HLSL's saturate() and extremely common in graphics code.
//
// Why?
//   - Simulation values occasionally overshoot due to numerical drift.
//   - Clamping prevents invalid colors (e.g., negative RGB or >1).
//   - Normalized color conversion (float → uchar) requires this clamp.
//
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
//
// Why?
//   - CUDA kernels output floating-point values for convenience.
//   - The OpenGL texture that displays the final image expects 8-bit pixels.
//   - This function performs:
//         clamp to [0,1]
//         scale to [0,255]
//         cast to unsigned char
//
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
