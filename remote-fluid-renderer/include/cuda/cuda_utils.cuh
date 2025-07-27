#pragma once

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

void cudaCheck(cudaError_t result, const char* msg);
void cudaRegisterBuffer(struct cudaGraphicsResource** resource, unsigned int vbo);
void cudaMapBuffer(struct cudaGraphicsResource* resource, void** devPtr, size_t* size);
void cudaUnmapBuffer(struct cudaGraphicsResource* resource);
void cudaUnregisterBuffer(struct cudaGraphicsResource* resource);

#ifdef __cplusplus
extern "C" {
#endif

void launchSimulationKernel(void* devPtr, size_t size, float time);

#ifdef __cplusplus
}
#endif