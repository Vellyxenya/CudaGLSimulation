#pragma once

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

__device__ __forceinline__ float saturatef(float x) {
    return fminf(fmaxf(x, 0.0f), 1.0f);
}

__device__ __forceinline__ uchar4 floatToUchar4(float4 f) {
    return make_uchar4(
        saturatef(f.x) * 255.0f,
        saturatef(f.y) * 255.0f,
        saturatef(f.z) * 255.0f,
        saturatef(f.w) * 255.0f
    );
}