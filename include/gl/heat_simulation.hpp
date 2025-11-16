#pragma once
// -----------------------------------------------------------------------------
// heat_simulation.hpp
//
// CUDA-based heat diffusion simulation. Inherits from Simulation.
// Writes RGBA output to a CUDA-mapped PBO for visualization.
//
// The simulation maintains two device buffers (current & next) to swap between
// timesteps efficiently. The heat equation is solved on the GPU using simple
// finite difference discretization.
// -----------------------------------------------------------------------------

#include "simulation.hpp"

class HeatSimulation : public Simulation {
public:
    // -------------------------------------------------------------------------
    // Constructor
    // Initializes simulation parameters and allocates CUDA buffers
    // width/height: simulation grid size
    // dt: timestep
    // diffusion: diffusion coefficient
    // sourceHeat: heat added per timestep in the source region
    // -------------------------------------------------------------------------
    HeatSimulation(int width, int height, float dt, float diffusion, float sourceHeat);

    // -------------------------------------------------------------------------
    // Destructor
    // Frees device memory (CUDA buffers)
    // -------------------------------------------------------------------------
    ~HeatSimulation();

    // -------------------------------------------------------------------------
    // Advances simulation one timestep
    // Writes RGBA output to the CUDA PBO (uchar4* pbo)
    // -------------------------------------------------------------------------
    void step(uchar4* pbo) override;

private:
    float m_dt;        // Timestep for simulation
    float m_diffusion; // Diffusion coefficient
    float m_sourceHeat; // Heat injected at source(s) per timestep
};
