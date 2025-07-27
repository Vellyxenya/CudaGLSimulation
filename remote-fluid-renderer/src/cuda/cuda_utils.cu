#include "cuda/cuda_utils.cuh"
#include <iostream>

void cudaCheck(cudaError_t result, const char* msg) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(result) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void cudaRegisterBuffer(cudaGraphicsResource** resource, unsigned int vbo) {
    cudaCheck(cudaGraphicsGLRegisterBuffer(resource, vbo, cudaGraphicsMapFlagsWriteDiscard),
              "Register OpenGL buffer with CUDA");
}

void cudaMapBuffer(cudaGraphicsResource* resource, void** devPtr, size_t* size) {
    cudaCheck(cudaGraphicsMapResources(1, &resource), "Map CUDA graphics resource");
    cudaCheck(cudaGraphicsResourceGetMappedPointer(devPtr, size, resource),
              "Get pointer to mapped CUDA resource");
}

void cudaUnmapBuffer(cudaGraphicsResource* resource) {
    cudaCheck(cudaGraphicsUnmapResources(1, &resource), "Unmap CUDA graphics resource");
}

void cudaUnregisterBuffer(cudaGraphicsResource* resource) {
    cudaCheck(cudaGraphicsUnregisterResource(resource), "Unregister CUDA graphics resource");
}
