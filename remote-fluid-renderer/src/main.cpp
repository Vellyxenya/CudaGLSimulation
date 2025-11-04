// src/main.cpp
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

extern "C" {
    __declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001; // NVIDIA GPU
    __declspec(dllexport) DWORD AmdPowerXpressRequestHighPerformance = 0x00000001; // AMD GPU
}

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

    // Allocate two float buffers: current and next state
    float* devCurrent;
    float* devNext;
    size_t gridSize = WIDTH * HEIGHT * sizeof(float);

    cudaMalloc(&devCurrent, gridSize);
    cudaMalloc(&devNext, gridSize);

    // Set initial conditions (host buffer)
    float* h_init = new float[WIDTH * HEIGHT](); // zero-initialized

    // Example: central hot spot
    //int cx = WIDTH / 2;
    //int cy = HEIGHT / 2;
    //h_init[cy * WIDTH + cx] = 25.0f;  // max heat in center

    cudaMemcpy(devCurrent, h_init, gridSize, cudaMemcpyHostToDevice);
    delete[] h_init;

    const float dt = 0.1f;
    const float diffusion = 0.2f;
    const float sourceHeat = 5.0f;

    // Create the renderer (sets up PBO, texture, CUDA interop, VAO, shaders)
    gl::Renderer renderer(WIDTH, HEIGHT);

    const GLubyte* r = glGetString(GL_RENDERER);
    std::cout << "Renderer: " << r << std::endl;

    // Main loop
    while (!glfwWindowShouldClose(window)) {
        // 1) Map the CUDA-accessible PBO and get the device pointer
        uchar4* devPtr = renderer.mapCudaResource();

        // 2) Launch the heat simulation kernel
        launchHeatSimulation(devCurrent, devNext, devPtr, WIDTH, HEIGHT, dt, diffusion, sourceHeat);

        // Swap buffers for next iteration
        std::swap(devCurrent, devNext);

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

    cudaFree(devCurrent);
    cudaFree(devNext);

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
