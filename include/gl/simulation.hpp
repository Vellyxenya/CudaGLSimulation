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
    virtual ~Simulation() = default;

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
};
