#pragma once
// -----------------------------------------------------------------------------
// gl_utils.hpp
//
// A small collection of helper functions for loading, compiling, and linking
// OpenGL shaders. These utilities are used by the renderer to create the GPU
// programs responsible for drawing the simulation output.
// -----------------------------------------------------------------------------

#include <string>
#include <glad/gl.h>

// -----------------------------------------------------------------------------
// loadShaderSource(path)
// Reads a text file from disk and returns its contents as a std::string.
// Used for loading GLSL vertex/fragment shader source code.
//
// Expected usage:
//     std::string vs = loadShaderSource("shaders/fullscreen.vert");
//     std::string fs = loadShaderSource("shaders/heat.frag");
// -----------------------------------------------------------------------------
std::string loadShaderSource(const std::string& path);

// -----------------------------------------------------------------------------
// compileShader(type, source)
// Compiles a GLSL shader of a given type (GL_VERTEX_SHADER or
// GL_FRAGMENT_SHADER) from a source string.
//
// Returns:
//     - GLuint shader handle (non-zero) on success
//     - 0 on compilation failure (errors printed to console)
//
// Notes:
//   - Compilation errors are extremely common during development.
//   - This helper isolates the boilerplate needed to check and report them.
// -----------------------------------------------------------------------------
GLuint compileShader(GLenum type, const std::string& source);

// -----------------------------------------------------------------------------
// createProgram(vertexPath, fragmentPath)
// Loads two shader files, compiles them, attaches them to a program,
// links the program, and returns the resulting pipeline.
//
// This is the highest-level helper: the renderer typically calls only this.
// Example:
//     GLuint program = createProgram("fullscreen.vert", "fluid.frag");
//
// Returns:
//     - Linked program ID on success
//     - 0 on failure
// -----------------------------------------------------------------------------
GLuint createProgram(const std::string& vertexPath, const std::string& fragmentPath);
