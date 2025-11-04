// sim/simulation.hpp
#pragma once
#include <cuda_runtime.h>

class Simulation {
public:
    virtual ~Simulation() = default;

    // Allocate device memory / initialize simulation
    virtual void init(int width, int height) = 0;

    // Perform one simulation step and write to PBO
    virtual void step(void* pbo, float dt) = 0;

    // Cleanup device memory
    virtual void cleanup() = 0;
};
