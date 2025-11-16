// src/main.cpp
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

extern "C" {
    __declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001; // NVIDIA GPU
    __declspec(dllexport) DWORD AmdPowerXpressRequestHighPerformance = 0x00000001; // AMD GPU
}

#include <chrono>
#include <iostream>
#include <string>
#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include "gl/renderer.hpp"
#include "gl/heat_simulation.hpp"
#include "gl/fluid_simulation.hpp"

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

    const GLubyte* r = glGetString(GL_RENDERER);
    std::cout << "Renderer: " << r << std::endl;

    Simulation* simulation = nullptr;
    bool useHeat = false;

    if (useHeat) {
        const float dt = 0.6f;
        const float diffusion = 0.2f;
        const float sourceHeat = 5.0f;
        simulation = new HeatSimulation(WIDTH, HEIGHT, dt, diffusion, sourceHeat);
    }
    else {
        float dt = 0.09f;
        simulation = new FluidSimulation(WIDTH, HEIGHT, dt);
    }

    int frames = 0;
    auto lastTime = std::chrono::high_resolution_clock::now();

    // Main loop
    while (!glfwWindowShouldClose(window)) {
        auto currentTime = std::chrono::high_resolution_clock::now();
        frames++;
        float delta = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - lastTime).count();
        if (delta >= 1.0f) {
            std::string title = "CUDA HeatSim - FPS: " + std::to_string(frames);
            glfwSetWindowTitle(window, title.c_str());
            frames = 0;
            lastTime = currentTime;
        }

        // 1) Map the CUDA-accessible PBO and get the device pointer
        uchar4* devPtr = renderer.mapCudaResource();

        // --- Mouse interaction (only if it's a FluidSimulation) ---
        FluidSimulation* fluidSim = dynamic_cast<FluidSimulation*>(simulation);
        if (fluidSim) {
            static double prevX = 0.0, prevY = 0.0;
            double mouseX, mouseY;
            glfwGetCursorPos(window, &mouseX, &mouseY);

            int leftDown = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT);
            int rightDown = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT);

            // Map window coords to simulation coords
            int simX = static_cast<int>(mouseX / WIDTH * fluidSim->width());
            int simY = static_cast<int>((HEIGHT - mouseY) / HEIGHT * fluidSim->height()); // flip Y

            float2 mouseVel = make_float2(static_cast<float>(mouseX - prevX),
                static_cast<float>(prevY - mouseY)); // note inverted Y
            prevX = mouseX;
            prevY = mouseY;

            if (leftDown == GLFW_PRESS) {
                float scale = 0.3f;
                fluidSim->injectFromMouse(simX, simY,
                    make_float2(mouseVel.x * scale, mouseVel.y * scale),
                    true);
            }
            else if (rightDown == GLFW_PRESS) {
                float scale = 0.3f;
                fluidSim->injectFromMouse(simX, simY,
                    make_float2(-mouseVel.x * scale, -mouseVel.y * scale),
                    false);
            }
        }

        // 2) Launch the heat simulation kernel
        simulation->step(devPtr);

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
