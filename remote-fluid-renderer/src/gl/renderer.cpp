// gl/renderer.cpp

#include "gl/renderer.hpp"
#include "gl/gl_utils.hpp"

#include <cuda_gl_interop.h>
#include <iostream>

namespace gl {

Renderer::Renderer(int width, int height)
    : width(width), height(height), glProgram(0), pbo(0), tex(0), vao(0), vbo(0), cudaPBO(nullptr) {
    initGLResources();
}

Renderer::~Renderer() {
    cleanup();
}

void Renderer::initGLResources() {
    // Create PBO
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 4 * sizeof(uint8_t), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // Register with CUDA
    cudaError_t err = cudaGraphicsGLRegisterBuffer(&cudaPBO, pbo, cudaGraphicsMapFlagsWriteDiscard);
    if (err != cudaSuccess) {
        std::cerr << "cudaGraphicsGLRegisterBuffer failed: "
                << cudaGetErrorString(err) << std::endl;
    }

    // Create texture
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);

    // Load shaders
    glProgram = createProgram("../shaders/quad.vert", "../shaders/quad.frag");
    if (!glProgram) {
        std::cerr << "Failed to create GL shader program." << std::endl;
    }

    // Setup empty VAO — full screen quad uses gl_VertexID, no vertex attributes
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glBindVertexArray(0);
}

uchar4* Renderer::mapCudaResource() {
    cudaGraphicsMapResources(1, &cudaPBO, 0);
    uchar4* dptr;
    size_t size;
    cudaGraphicsResourceGetMappedPointer((void**)&dptr, &size, cudaPBO);
    std::cout << "Mapped CUDA buffer size: " << size << " bytes" << std::endl;
    return dptr;
}

void Renderer::unmapCudaResource() {
    cudaGraphicsUnmapResources(1, &cudaPBO, 0);
}

void Renderer::draw() {
    // Upload data from PBO to texture
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // Draw full screen quad
    glUseProgram(glProgram);
    glBindVertexArray(vao);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex);

    // Set sampler uniform if needed (only once is enough, but harmless here)
    GLint loc = glGetUniformLocation(glProgram, "screenTexture");
    if (loc != -1) {
        glUniform1i(loc, 0); // Texture unit 0
    }

    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    // Cleanup
    glBindVertexArray(0);
    glUseProgram(0);
}

void Renderer::cleanup() {
    if (cudaPBO) {
        cudaGraphicsUnregisterResource(cudaPBO);
        cudaPBO = nullptr;
    }
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
