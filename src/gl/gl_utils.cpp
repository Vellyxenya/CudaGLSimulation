#define WIN32_LEAN_AND_MEAN
#include <windows.h>       // must come first
#include "gl/gl_utils.hpp"

#include <fstream>
#include <iostream>
#include <sstream>


std::string loadShaderSource(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Failed to load shader file: " << path << std::endl;
        return "";
    }

    // Read file into a string buffer
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

GLuint compileShader(GLenum type, const std::string& source) {
    GLuint shader = glCreateShader(type);

    const char* src = source.c_str();
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);

    // Check compilation status
    GLint compiled = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
    if (!compiled) {
        // Retrieve and print the compilation log
        GLint len;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &len);
        std::string log(len, ' ');
        glGetShaderInfoLog(shader, len, nullptr, &log[0]);
        std::cerr << "Shader compile error:\n" << log << std::endl;

        glDeleteShader(shader);
        return 0;
    }

    return shader;
}

GLuint createProgram(const std::string& vertexPath, const std::string& fragmentPath) {
    // Load GLSL sources
    std::string vertSrc = loadShaderSource(vertexPath);
    std::string fragSrc = loadShaderSource(fragmentPath);
    if (vertSrc.empty() || fragSrc.empty()) return 0;

    // Compile vertex & fragment shaders
    GLuint vertShader = compileShader(GL_VERTEX_SHADER, vertSrc);
    GLuint fragShader = compileShader(GL_FRAGMENT_SHADER, fragSrc);
    if (!vertShader || !fragShader) return 0;

    // Create program and attach shaders
    GLuint program = glCreateProgram();
    glAttachShader(program, vertShader);
    glAttachShader(program, fragShader);
    glLinkProgram(program);

    // Check linking status
    GLint linked;
    glGetProgramiv(program, GL_LINK_STATUS, &linked);
    if (!linked) {
        // Retrieve and print link errors
        GLint len;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &len);
        std::string log(len, ' ');
        glGetProgramInfoLog(program, len, nullptr, &log[0]);
        std::cerr << "Program link error:\n" << log << std::endl;

        glDeleteProgram(program);
        program = 0;
    }

    // Clean up shaders; they are no longer needed after linking
    glDeleteShader(vertShader);
    glDeleteShader(fragShader);

    return program;
}
