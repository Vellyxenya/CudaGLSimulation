#define WIN32_LEAN_AND_MEAN
#include <windows.h>       // must come first for Windows headers

#include "gl/renderer.hpp"
#include "gl/gl_utils.hpp"

// #include <cuda_gl_interop.h>
#include <iostream>

namespace gl {

Renderer::Renderer(int width, int height)
    : width(width), height(height), glProgram(0), pbo(0), tex(0), vao(0), cudaPBO(nullptr) {
    initGLResources(); // Initialize all GL and CUDA resources
}

Renderer::~Renderer() {
    cleanup(); // Ensure all resources are released
}

void Renderer::initGLResources() {
    // -----------------------------
    // Create Pixel Buffer Object (PBO)
    // -----------------------------
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    // Allocate GPU memory (width*height*RGBA8) without initializing
    glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 4 * sizeof(uint8_t), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // -----------------------------
    // Register PBO with CUDA
    // -----------------------------
    cudaError_t err = cudaGraphicsGLRegisterBuffer(&cudaPBO, pbo, cudaGraphicsMapFlagsWriteDiscard);
    if (err != cudaSuccess) {
        std::cerr << "cudaGraphicsGLRegisterBuffer failed: "
                  << cudaGetErrorString(err) << std::endl;
    }

    // -----------------------------
    // Create OpenGL texture to display PBO contents
    // -----------------------------
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    // Allocate empty RGBA8 texture
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);

    // -----------------------------
    // Compile and link shader program for fullscreen quad
    // -----------------------------
    // Note: these paths may need to be adjusted based on your working directory
    // Ideally, we would use a more robust method to locate shader files, but
    // for simplicity we use relative paths here.
    glProgram = createProgram("../shaders/quad.vert", "../shaders/quad.frag");
    if (!glProgram) {
        std::cerr << "Failed to create GL shader program." << std::endl;
    }

    // -----------------------------
    // Setup empty VAO — quad uses gl_VertexID, no vertex attributes
    // -----------------------------
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glBindVertexArray(0);
}

uchar4* Renderer::mapCudaResource() {
    // Map the PBO for CUDA writing
    cudaGraphicsMapResources(1, &cudaPBO, 0);

    uchar4* dptr;
    size_t size;
    // Retrieve device pointer to the PBO
    cudaGraphicsResourceGetMappedPointer((void**)&dptr, &size, cudaPBO);

    return dptr;
}

void Renderer::unmapCudaResource() {
    // Unmap the PBO so OpenGL can safely read it
    cudaGraphicsUnmapResources(1, &cudaPBO, 0);
}

void Renderer::draw() {
    // -----------------------------
    // Copy PBO contents to texture
    // -----------------------------
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBindTexture(GL_TEXTURE_2D, tex);
    // Updates the texture with the PBO data
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // -----------------------------
    // Draw fullscreen quad
    // -----------------------------
    glUseProgram(glProgram);
    glBindVertexArray(vao);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex);

    // Set sampler uniform (texture unit 0)
    GLint loc = glGetUniformLocation(glProgram, "screenTexture");
    if (loc != -1) {
        glUniform1i(loc, 0);
    }

    // Draw quad using triangle strip
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    // Cleanup bindings
    glBindVertexArray(0);
    glUseProgram(0);
}

void Renderer::cleanup() {
    // -----------------------------
    // Unregister CUDA resource
    // -----------------------------
    if (cudaPBO) {
        cudaGraphicsUnregisterResource(cudaPBO);
        cudaPBO = nullptr;
    }

    // -----------------------------
    // Delete OpenGL buffers/textures/programs
    // -----------------------------
    if (pbo) {
        glDeleteBuffers(1, &pbo);
        pbo = 0;
    }
    if (tex) {
        glDeleteTextures(1, &tex);
        tex = 0;
    }
    if (glProgram) {
        glDeleteProgram(glProgram);
        glProgram = 0;
    }
    if (vao) {
        glDeleteVertexArrays(1, &vao);
        vao = 0;
    }
}

}  // namespace gl
