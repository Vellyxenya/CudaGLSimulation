// src/main.cpp
// -----------------------------------------------------------------------------
// Entry point for a CUDA + OpenGL fluid/heat simulation.
// Creates an OpenGL window, initializes CUDA-interop rendering,
// and runs either a heat diffusion sim or a Navier–Stokes fluid sim.
// -----------------------------------------------------------------------------

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

// -----------------------------------------------------------------------------
// GPU selection exports
// These are special Windows exports that force laptops with hybrid GPUs
// (Optimus / AMD switchable graphics) to launch the program on the high-performance GPU.
// -----------------------------------------------------------------------------
extern "C" {
    __declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001;               // Prefer NVIDIA GPU
    __declspec(dllexport) DWORD AmdPowerXpressRequestHighPerformance = 0x00000001; // Prefer AMD GPU
}

#include <chrono>
#include <iostream>
#include <string>
#include <glad/gl.h>     // Modern OpenGL function loader
#include <GLFW/glfw3.h>  // Window + context + input
#include <glm/glm.hpp>

#include "gl/renderer.hpp"         // CUDA-PBO interop renderer
#include "gl/heat_simulation.hpp"  // Heat diffusion simulation

// -----------------------------------------------------------------------------
// Default window size. These can be overridden using:
//   --width=<N>
//   --height=<N>
// -----------------------------------------------------------------------------
const int DEFAULT_WIDTH  = 800;
const int DEFAULT_HEIGHT = 600;

// -----------------------------------------------------------------------------
// Initializes a GLFW window and an OpenGL 4.3 core context.
// Also loads OpenGL function pointers using GLAD.
// -----------------------------------------------------------------------------
GLFWwindow* initWindow(int width, int height) {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        return nullptr;
    }

    // Request a reasonably modern OpenGL version (needed for compute shaders, SSBO, etc.)
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Create window + OpenGL context
    GLFWwindow* window = glfwCreateWindow(width, height, "CUDA HeatSim", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return nullptr;
    }

    glfwMakeContextCurrent(window);

    // Load OpenGL function pointers through GLAD
    if (!gladLoaderLoadGL()) {
        std::cerr << "Failed to initialize GLAD\n";
        glfwDestroyWindow(window);
        glfwTerminate();
        return nullptr;
    }

    // Set initial viewport and clear color
    glViewport(0, 0, width, height);
    glClearColor(0.8f, 0.8f, 1.0f, 1.0f);

    return window;
}

// -----------------------------------------------------------------------------
// MAIN
// Usage: sim.exe [--width=<n>] [--height=<n>]
// -----------------------------------------------------------------------------
int main(int argc, char** argv) {

    // ---------------------------------------------------------
    // Parse CLI arguments (very minimal parser)
    // This could (and should) be replaced with a proper CLI
    // parsing library such as cxxopts or CLI11 but I am keeping
    // it simple for this demo.
    // ---------------------------------------------------------

    int width  = DEFAULT_WIDTH;
    int height = DEFAULT_HEIGHT;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a.rfind("--width=", 0) == 0)
            width = std::stoi(a.substr(8));
        if (a.rfind("--height=", 0) == 0)
            height = std::stoi(a.substr(9));
    }

    // ---------------------------------------------------------
    // Initialize OpenGL window
    // ---------------------------------------------------------
    GLFWwindow* window = initWindow(width, height);
    if (!window) return -1;

    // Force CUDA to use device 0 (no multi-GPU support here)
    cudaSetDevice(0);

    // Create renderer: sets up:
    // - OpenGL pixel buffer (PBO)
    // - CUDA access to that PBO
    // - Texture used for on-screen display
    // - Fullscreen quad & shaders
    gl::Renderer renderer(width, height);

    // Print GPU renderer information
    const GLubyte* r = glGetString(GL_RENDERER);
    std::cout << "Renderer: " << r << std::endl;

    // ---------------------------------------------------------
    // Create simulation object based on CLI flags
    // ---------------------------------------------------------
    const float heatDt        = 0.6f;
    const float diffusion     = 0.2f;
    const float sourceHeat    = 5.0f;
    Simulation* simulation = new HeatSimulation(width, height, heatDt, diffusion, sourceHeat);

    // ---------------------------------------------------------
    // FPS measurement setup
    // ---------------------------------------------------------
    int frames = 0;
    auto lastTime = std::chrono::high_resolution_clock::now();

    // -------------------------------------------------------------------------
    // MAIN RENDER LOOP
    // -------------------------------------------------------------------------
    while (!glfwWindowShouldClose(window)) {
        auto currentTime = std::chrono::high_resolution_clock::now();
        frames++;

        // Every second: update window title with FPS
        float delta = std::chrono::duration<float>(currentTime - lastTime).count();
        if (delta >= 1.0f) {
            std::string title = "CUDA HeatSim - FPS: " + std::to_string(frames);
            glfwSetWindowTitle(window, title.c_str());
            frames = 0;
            lastTime = currentTime;
        }

        // ---------------------------------------------------------------------
        // STEP 1: Map the CUDA PBO and get direct device pointer to pixel buffer
        // This pointer is written into directly by the simulation kernel.
        // ---------------------------------------------------------------------
        uchar4* devPtr = renderer.mapCudaResource();

        // ---------------------------------------------------------------------
        // STEP 2: Run one simulation step (CUDA kernel inside each sim class)
        // Writes directly into devPtr (mapped PBO).
        // ---------------------------------------------------------------------
        simulation->step(devPtr);

        // ---------------------------------------------------------------------
        // STEP 3: Unmap CUDA resource so OpenGL can safely read it again.
        // ---------------------------------------------------------------------
        renderer.unmapCudaResource();

        // ---------------------------------------------------------------------
        // STEP 4: Render the texture updated by CUDA
        // (PBO → texture upload, then fullscreen quad draw)
        // ---------------------------------------------------------------------
        glClear(GL_COLOR_BUFFER_BIT);
        renderer.draw();

        // Swap buffers and poll events
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // -------------------------------------------------------------------------
    // Clean shutdown
    // -------------------------------------------------------------------------
    renderer.cleanup();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
