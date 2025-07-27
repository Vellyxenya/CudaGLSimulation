// gl/gl_utils.hpp

#ifndef GL_UTILS_HPP
#define GL_UTILS_HPP

#include <string>
#include <glad/gl.h>

void checkGLError(const char* context = "");

std::string loadShaderSource(const std::string& path);

GLuint compileShader(GLenum type, const std::string& source);

GLuint createProgram(const std::string& vertexPath, const std::string& fragmentPath);

#endif // GL_UTILS_HPP
