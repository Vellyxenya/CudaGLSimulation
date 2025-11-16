#pragma once
// -----------------------------------------------------------------------------
// fluid_simulation.hpp
//
// CUDA-based 2D fluid simulation. Inherits from Simulation.
// Solves the Navier-Stokes equations using a simple semi-Lagrangian method
// with diffusion, advection, and pressure projection. Outputs an RGBA texture
// to a CUDA-mapped PBO for visualization.
//
// Supports mouse interaction to inject velocity and/or density.
// -----------------------------------------------------------------------------

#include "simulation.hpp"

class FluidSimulation : public Simulation {
public:
    // -------------------------------------------------------------------------
    // Constructor
    // Initializes simulation parameters and allocates CUDA buffers
    // width/height: simulation grid size
    // dt: timestep
    // -------------------------------------------------------------------------
    FluidSimulation(int width, int height, float dt);

    // -------------------------------------------------------------------------
    // Destructor
    // Frees all device memory
    // -------------------------------------------------------------------------
    ~FluidSimulation();

    // -------------------------------------------------------------------------
    // Advances simulation one timestep
    // Computes velocity, pressure, density, and writes RGBA output to PBO
    // -------------------------------------------------------------------------
    void step(uchar4* pbo) override;

    // -------------------------------------------------------------------------
    // Accessor functions
    // -------------------------------------------------------------------------
    int width() const { return m_width; }
    int height() const { return m_height; }

    // -------------------------------------------------------------------------
    // Mouse interaction
    // Inject velocity and/or density at simulation coordinates (x, y)
    // force: velocity to add
    // addDensity: whether to add density at this location
    // -------------------------------------------------------------------------
    void injectFromMouse(int x, int y, float2 force, bool addDensity);

private:
    // -----------------------------
    // Simulation parameters
    // -----------------------------
    float m_dt;        // Timestep
    float m_diffusion; // Diffusion coefficient for density
    float m_viscosity; // Fluid viscosity for velocity
    float m_sourceDensity; // Density injected per timestep

    // -----------------------------
    // Velocity grids (float per cell)
    // Separate U and V components
    // Ping-pong buffers for advection/force updates
    // -----------------------------
    float* m_velU;
    float* m_velV;
    float* m_velNextU;
    float* m_velNextV;

    // -----------------------------
    // Pressure & divergence fields for incompressibility
    // Used in projection step
    // -----------------------------
    float* m_pressure;
    float* m_divergence;
};
