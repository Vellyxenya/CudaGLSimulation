# Remote Fluid Renderer

Real-time 2D fluid simulation with CUDA and OpenGL interop.

## Structure
- **deps/**: External dependencies (glad)
- **include/**: C++ headers
- **shaders/**: GLSL shaders
- **src/**: C++ source files

## Build and Run
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