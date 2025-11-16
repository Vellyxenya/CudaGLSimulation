#pragma once
// -----------------------------------------------------------------------------
// simulation.hpp
//
// Base class for all CUDA-accelerated simulations in this project.
// Both the heat diffusion simulator and the fluid simulator inherit from this
// interface. The renderer and main loop only depend on this abstract class,
// which lets us swap simulations at runtime without changing the rendering
// pipeline.
//
// The design goal:
//     - Keep the rendering code independent of the physics implementation.
//     - Each simulation is responsible only for computing pixel output.
// -----------------------------------------------------------------------------

#include <cuda_runtime.h>

class Simulation {
public:
    // -------------------------------------------------------------------------
    // Constructor
    // Initializes simulation grid size and allocates device memory
    // -------------------------------------------------------------------------
    Simulation(int width, int height) : m_width(width), m_height(height),
        m_devCurrent(nullptr), m_devNext(nullptr) {}

    // -------------------------------------------------------------------------
    // Destructor
    // Frees device memory allocated for simulation grids
    // -------------------------------------------------------------------------
    virtual ~Simulation() {
        cudaFree(m_devCurrent);
        cudaFree(m_devNext);
    }

    // -------------------------------------------------------------------------
    // step(pbo)
    //
    // Advance the simulation by one timestep and write the final RGBA image
    // directly into the CUDA-mapped Pixel Buffer Object (PBO).
    //
    // Arguments:
    //     pbo — device pointer to an array of uchar4 pixels.
    //           This memory is owned by OpenGL but mapped into CUDA for
    //           writing. The simulation never allocates or frees this buffer.
    // -------------------------------------------------------------------------
    virtual void step(uchar4* pbo) = 0;

protected:
    // -----------------------------
    // Density grids (float per cell)
    // Ping-pong buffers for current/next values
    // -----------------------------
    float* m_devCurrent;
    float* m_devNext;

    int m_width;       // Grid width
    int m_height;      // Grid height
};
