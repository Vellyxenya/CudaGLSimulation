// src/main.cpp

/*
Detailed overview and notes (for documentation / Medium article):

This file is the top-level application for the RemoteSimGL demo. It does
the following at a high level:
- Initializes a GLFW window and OpenGL context (via `initWindow`).
- Loads OpenGL entry points with `glad`.
- Selects a CUDA device with `cudaSetDevice(0)` for running CUDA kernels.
- Creates a `gl::Renderer` which sets up an OpenGL Pixel Buffer Object
    (PBO) and a texture; the PBO is registered with CUDA so kernels can
    write pixels directly into GPU memory.
- Creates either a `FluidSimulation` or `HeatSimulation` object which
    implements `step()` to run CUDA kernels that write into the PBO.
- Enters the main loop where, each frame, it maps the PBO for CUDA, runs
    the simulation step, unmaps the PBO, and draws the texture to the
    screen. Mouse input is translated into simulation-space forces for the
    fluid simulation.

Key symbols / calls explained:
- `WIN32_LEAN_AND_MEAN`:
    - Tells `<Windows.h>` to exclude rarely-used APIs, reducing namespace
        pollution and slightly speeding compilation.

- `__declspec(dllexport)` variables (below):
    - The exported globals `NvOptimusEnablement` and
        `AmdPowerXpressRequestHighPerformance` are a common, pragmatic way
        to hint to laptop GPU drivers that the program prefers the
        high-performance discrete GPU. They are not a CUDA call; they cause
        the loader/driver to prefer the discrete GPU when available. This is
        useful on hybrid-graphics machines where an integrated GPU is the
        default.

- `<chrono>, <iostream>, <string>`:
    - Standard C++ headers used for timing, logging, and simple CLI
        parsing / string manipulation.

- `<glad/gl.h>`:
    - glad is a GL loader. On many platforms (including Windows), OpenGL
        functions are not available as static link symbols; glad queries the
        function pointers at runtime after an OpenGL context is created.

- `<GLFW/glfw3.h>`:
    - GLFW provides a simple cross-platform API to create windows and
        OpenGL contexts and to handle input (keyboard, mouse). `glfwInit()`
        initializes the library; `glfwCreateWindow(...)` creates a window +
        context; `glfwMakeContextCurrent(...)` binds the context to the
        calling thread.

- `glViewport(...)` and `glClearColor(...)`:
    - `glViewport` sets the pixel-to-NDC mapping for rendering; match it
        to the window size. `glClearColor` specifies the default color used
        when clearing the color buffer.

- `cudaSetDevice(0)`:
    - Selects the CUDA device index for the current host thread. On a
        multi-GPU system you may change the index; `0` is commonly the
        first visible device. This does not change the GL context's GPU by
        itself, but it sets which GPU subsequent CUDA calls will target.

- `mapCudaResource()` / `unmapCudaResource()` (renderer):
    - These functions coordinate CUDA-OpenGL interop. Mapping returns a
        device pointer that CUDA kernels can write to; unmapping returns
        control to OpenGL so the texture can be used for rendering.

- `dynamic_cast<FluidSimulation*>(simulation)`:
    - At runtime we check whether the `Simulation*` is a `FluidSimulation`
        so we can enable mouse-driven interaction only for the fluid sim.

Notes about maintainability and suggestions:
- The main uses `new` to allocate the simulation and never `delete`s it.
    This is fine for a short-lived demo that exits immediately, but for
    production code prefer `std::unique_ptr` to make ownership explicit.
- CLI parsing here is intentionally minimal. If you want better UX,
    add a `--help` flag and validate numeric args with clear errors.

The remainder of the file contains the concrete implementation; inline
comments throughout the file provide more focused explanations at each
call site (GLFW initialization, window creation, mapping/unmapping,
mouse coordinate mapping, and cleanup).
*/

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

// Default window size; can be overridden on the command line with
// `--width=<n>` and `--height=<n>`.
const int DEFAULT_WIDTH  = 800;
const int DEFAULT_HEIGHT = 600;

GLFWwindow* initWindow(int width, int height) {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        return nullptr;
    }

    // Request OpenGL 4.3 core profile
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(width, height, "CUDA FluidSim", nullptr, nullptr);
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

    glViewport(0, 0, width, height);
    glClearColor(0.8f, 0.8f, 1.0f, 1.0f);
    return window;
}

// Usage: sim.exe [--heat] [--width=<n>] [--height=<n>]
int main(int argc, char** argv) {
    // Simulation selection and parameter defaults. Use command-line flags
    // to override these simple defaults.
    enum class SimulationType { Fluid, Heat };
    SimulationType simType = SimulationType::Fluid;

    // Window size (override with CLI)
    int width = DEFAULT_WIDTH;
    int height = DEFAULT_HEIGHT;

    // Minimal CLI parsing: supports --heat, --width=, --height= flags.
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--heat") simType = SimulationType::Heat;
        else if (a.rfind("--width=", 0) == 0) width = std::stoi(a.substr(8));
        else if (a.rfind("--height=", 0) == 0) height = std::stoi(a.substr(9));
    }

    // Initialize the GL window with the chosen size.
    GLFWwindow* window = initWindow(width, height);
    if (!window) return -1;

    cudaSetDevice(0);

    // Create the renderer (sets up PBO, texture, CUDA interop, VAO, shaders)
    gl::Renderer renderer(width, height);

    const GLubyte* r = glGetString(GL_RENDERER);
    std::cout << "Renderer: " << r << std::endl;

    // Instantiate the chosen simulation with the configured parameters.
    Simulation* simulation = nullptr;
    if (simType == SimulationType::Heat) {
        const float heatDt = 0.6f;
        const float diffusion = 0.2f;
        const float sourceHeat = 5.0f;
        simulation = new HeatSimulation(width, height, heatDt, heatDiffusion, heatSource);
    } else {
        float fluidDt = 0.09f;
        simulation = new FluidSimulation(width, height, fluidDt);
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

        // 2) Handle input / user interaction (mouse). Only applies to `FluidSimulation`.
        FluidSimulation* fluidSim = dynamic_cast<FluidSimulation*>(simulation);
        if (fluidSim) {
            static double prevX = 0.0, prevY = 0.0;
            double mouseX, mouseY;
            glfwGetCursorPos(window, &mouseX, &mouseY);

            int leftDown = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT);
            int rightDown = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT);

            // Map window coords to simulation coords (note Y flip)
            int simX = static_cast<int>(mouseX / width * fluidSim->width());
            int simY = static_cast<int>((height - mouseY) / height * fluidSim->height()); // flip Y

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

        // 3) Advance the simulation (one timestep). Implementation depends on concrete sim.
        simulation->step(devPtr);

        // 4) Unmap so OpenGL can use the updated PBO
        renderer.unmapCudaResource();

        // 5) Render: upload PBO→texture and draw fullscreen quad
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
