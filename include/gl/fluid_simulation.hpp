// sim/fluid_simulation.hpp
#pragma once
#include "simulation.hpp"

class FluidSimulation : public Simulation {
public:
    FluidSimulation(int width, int height, float dt);
    ~FluidSimulation();

    // Set initial condition (optional)
    void setInitialCondition(const float* hostData = nullptr);

    // Advance one step of simulation and write to a CUDA buffer (PBO)
    void step(uchar4* pbo) override;

    int width() const { return m_width; }
    int height() const { return m_height; }

    void injectFromMouse(int x, int y, float2 force, bool addDensity);

private:
    int m_width;
    int m_height;
    float m_dt;
    float m_diffusion;
    float m_viscosity;
    float m_sourceDensity;

    // Density grids
    float* m_devCurrent;
    float* m_devNext;

    // Velocity grids
    float* m_velU;
    float* m_velV;
    float* m_velNextU;
    float* m_velNextV;

    float* m_pressure;
    float* m_divergence;
};