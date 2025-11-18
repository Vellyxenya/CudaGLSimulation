#pragma once
#include <cuda_runtime.h>

// Advance the simulation and write pixels into the provided PBO pointer.
void step(uchar4* pbo, int width, int height, float t);
