#pragma once
// -----------------------------------------------------------------------------
// renderer.hpp
//
// OpenGL + CUDA interoperability renderer.
// This class is responsible for:
//   1) Creating an OpenGL Pixel Buffer Object (PBO) for CUDA writes
//   2) Mapping/unmapping it to CUDA
//   3) Uploading it as a texture
//   4) Drawing a fullscreen quad to display the simulation
//
// All CUDA-based simulations output pixels to the PBO. The Renderer handles
// displaying those pixels on screen efficiently without extra CPU copies.
// -----------------------------------------------------------------------------

#include <glad/gl.h>
#include <cuda_gl_interop.h>

namespace gl {

class Renderer {
public:
    // -------------------------------------------------------------------------
    // Constructor
    // Initializes OpenGL resources and CUDA interop for a given viewport size.
    // -------------------------------------------------------------------------
    Renderer(int width, int height);

    // -------------------------------------------------------------------------
    // Destructor
    // Ensures that OpenGL and CUDA resources are freed properly.
    // -------------------------------------------------------------------------
    ~Renderer();

    // -------------------------------------------------------------------------
    // draw()
    // Draws the fullscreen quad with the texture that is currently updated
    // by CUDA.
    // -------------------------------------------------------------------------
    void draw();

    // -------------------------------------------------------------------------
    // cleanup()
    // Frees OpenGL buffers, textures, VAO, and unregisters the PBO from CUDA.
    // This is necessary to prevent memory/resource leaks on exit.
    // -------------------------------------------------------------------------
    void cleanup();

    // -------------------------------------------------------------------------
    // mapCudaResource()
    // Maps the OpenGL PBO to CUDA and returns a device pointer.
    // CUDA kernels write pixel data directly to this pointer.
    //
    // Usage:
    //    uchar4* devPtr = renderer.mapCudaResource();
    //    simulation->step(devPtr);
    //    renderer.unmapCudaResource();
    // -------------------------------------------------------------------------
    uchar4* mapCudaResource();

    // -------------------------------------------------------------------------
    // unmapCudaResource()
    // Unmaps the PBO from CUDA so OpenGL can safely use it.
    // Must be called after finishing CUDA writes each frame.
    // -------------------------------------------------------------------------
    void unmapCudaResource();

private:
    // -------------------------------------------------------------------------
    // initGLResources()
    // Internal helper that sets up:
    //   - Vertex Array & buffer for fullscreen quad
    //   - Shader program
    //   - OpenGL texture & PBO
    //   - Registers PBO with CUDA
    // -------------------------------------------------------------------------
    void initGLResources();

private:
    int width, height;                   // Viewport size
    GLuint glProgram;                    // Shader program ID
    GLuint pbo;                          // Pixel Buffer Object (CUDA writes here)
    GLuint tex;                          // Texture used for drawing PBO
    cudaGraphicsResource* cudaPBO;       // CUDA resource handle for the PBO
    GLuint vao;                          // Fullscreen quad VAO
};

}  // namespace gl
