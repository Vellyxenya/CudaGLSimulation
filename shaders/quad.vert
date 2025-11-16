#version 330 core
out vec2 uv;
void main() {
    const vec2 pos[4] = vec2[](
        vec2(-1.0, -1.0), vec2(1.0, -1.0),
        vec2(-1.0,  1.0), vec2(1.0,  1.0)
    );
    const vec2 tex[4] = vec2[](
        vec2(0.0, 0.0), vec2(1.0, 0.0),
        vec2(0.0, 1.0), vec2(1.0, 1.0)
    );
    gl_Position = vec4(pos[gl_VertexID], 0.0, 1.0);
    uv = tex[gl_VertexID];
}