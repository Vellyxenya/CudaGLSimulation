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

private:
    int m_width;
    int m_height;
    float m_dt;
    float m_diffusion;
    float m_sourceHeat;

    float* m_devCurrent;
    float* m_devNext;
};
