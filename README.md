# Cuda-GL Simulation

Real-time 2D heat and fluid simulation with CUDA and OpenGL interop.

## Project Structure

```
CudaGLSimulation/
├── CMakeLists.txt                 # CMake build configuration
├── build/                         # Build output directory
│   ├── ...
├── deps/                          # External dependencies
│   └── glad/                      # Glad dependency
│       ├── include/
│       └── src/
├── include/                       # C++ headers
│   ├── cuda/
│   └── gl/
├── shaders/                       # GLSL shader files
│   ├── quad.frag
│   └── quad.vert
└── src/                           # C++ source files
    ├── main.cpp
    ├── cuda/
    └── gl/
```

## Build and Run
Please note that this project has been setup to run on Windows.
If you are running on Linux or MacOS, you may need to slightly adapt the code.

1. Create the `build` directory and `cd` into it:
```bash
mkdir build; cd build
```

2. Configure cmake:
```bash
cmake ..
```

3. Build:
```bash
cmake --build . --config Release --target sim
```

4. Run:
```bash
.\Release\sim.exe
```