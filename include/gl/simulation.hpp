// sim/simulation.hpp
#pragma once
#include <cuda_runtime.h>

class Simulation {
public:
    virtual ~Simulation() = default;

    // Main function to update the simulation and write to PBO
    virtual void step(uchar4* pbo) = 0;
};