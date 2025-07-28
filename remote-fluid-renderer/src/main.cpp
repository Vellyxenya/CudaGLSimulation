// src/main.cpp

#include <iostream>
#include <glad/gl.h>
#include <GLFW/glfw3.h>

#include "gl/renderer.hpp"
#include "cuda/cuda_utils.cuh"

const int WIDTH  = 800;
const int HEIGHT = 600;

GLFWwindow* initWindow() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        return nullptr;
    }

    // Request OpenGL 4.3 core profile
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "CUDA FluidSim", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return nullptr;
    }
    glfwMakeContextCurrent(window);

    // Load all GL entry points
    if (!gladLoaderLoadGL()) {
        std::cerr << "Failed to initialize GLAD\n";
        glfwDestroyWindow(window);
        glfwTerminate();
        return nullptr;
    }

    glViewport(0, 0, WIDTH, HEIGHT);
    glClearColor(0.8f, 0.8f, 1.0f, 1.0f);
    return window;
}

int main() {
    GLFWwindow* window = initWindow();
    if (!window) return -1;

    cudaSetDevice(0);

    // Create the renderer (sets up PBO, texture, CUDA interop, VAO, shaders)
    gl::Renderer renderer(WIDTH, HEIGHT);

    // Main loop
    while (!glfwWindowShouldClose(window)) {
        std::cout << "Expected size: " << WIDTH * HEIGHT * sizeof(uchar4) << " bytes" << std::endl;

        // 1) Map the CUDA-accessible PBO and get the device pointer
        uchar4* devPtr = renderer.mapCudaResource();

        // 2) Launch the simulation kernel (grid size inferred inside)
        launchSimulationKernel(devPtr, WIDTH * HEIGHT * sizeof(uchar4),
                               static_cast<float>(glfwGetTime()));

        // 3) Unmap so OpenGL can use the updated PBO
        renderer.unmapCudaResource();

        // 4) Render: upload PBO→texture and draw fullscreen quad
        glClear(GL_COLOR_BUFFER_BIT);
        renderer.draw();

        // Swap & poll
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Cleanup
    renderer.cleanup();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
