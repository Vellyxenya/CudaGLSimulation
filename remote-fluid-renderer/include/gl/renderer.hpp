#pragma once

#include <glad/gl.h>
#include <cuda_gl_interop.h>

namespace gl {

class Renderer {
public:
    Renderer(int width, int height);
    ~Renderer();

    void draw();                    // Renders the texture to screen
    void cleanup();                // Frees GL and CUDA resources

    uchar4* mapCudaResource();     // Maps PBO for CUDA
    void unmapCudaResource();      // Unmaps PBO from CUDA

private:
    void initGLResources();

    int width, height;
    GLuint glProgram;
    GLuint pbo;
    GLuint tex;
    cudaGraphicsResource* cudaPBO;

     GLuint vao, vbo;
};

}  // namespace gl
