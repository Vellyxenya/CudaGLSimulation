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
// Reads the entire contents of a text file into a string
// Used for loading GLSL shader source code from disk
// -----------------------------------------------------------------------------
std::string loadShaderSource(const std::string& path);

// -----------------------------------------------------------------------------
// Compiles a GLSL shader from a source string
// type: GL_VERTEX_SHADER or GL_FRAGMENT_SHADER
// Returns shader ID on success, 0 on failure
// -----------------------------------------------------------------------------
GLuint compileShader(GLenum type, const std::string& source);

// -----------------------------------------------------------------------------
// Creates a shader program from vertex and fragment shader files
// Steps:
//   1) Load shader sources from disk
//   2) Compile shaders
//   3) Attach and link program
//   4) Delete individual shaders after linking
// Returns program ID on success, 0 on failure
// -----------------------------------------------------------------------------
GLuint createProgram(const std::string& vertexPath, const std::string& fragmentPath);
