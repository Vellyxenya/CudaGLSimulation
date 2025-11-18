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

// Advance the simulation and write pixels into the provided PBO pointer.
void step(uchar4* pbo, int width, int height, float t);
