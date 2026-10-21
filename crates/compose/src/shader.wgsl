struct Coefficients {
    y_offset: f32,
    y_scale: f32,
    chroma_offset: f32,
    chroma_scale: f32,
    kr: f32,
    kb: f32,
    code_scale: f32,
    code_divisor: f32,
}

@group(0) @binding(0) var y_tex: texture_2d<f32>;
@group(0) @binding(1) var c_tex: texture_2d<f32>;
@group(0) @binding(2) var<uniform> coef: Coefficients;

struct Vertex {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vertex(@builtin(vertex_index) index: u32) -> Vertex {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    let clip = positions[index];
    var out: Vertex;
    out.position = vec4<f32>(clip, 0.0, 1.0);
    // wgpu's framebuffer origin is the top left, and clip y points up.
    out.uv = vec2<f32>((clip.x + 1.0) * 0.5, (1.0 - clip.y) * 0.5);
    return out;
}

@fragment
fn fragment(in: Vertex) -> @location(0) vec4<f32> {
    let size = vec2<i32>(textureDimensions(y_tex));
    let pixel = clamp(vec2<i32>(in.uv * vec2<f32>(size)), vec2<i32>(0), size - vec2<i32>(1));
    let y_code = textureLoad(y_tex, pixel, 0).r * coef.code_scale / coef.code_divisor;
    let chroma = textureLoad(c_tex, pixel / 2, 0).rg * coef.code_scale / coef.code_divisor;
    let y = (y_code - coef.y_offset) * coef.y_scale;
    let cb = (chroma.r - coef.chroma_offset) * coef.chroma_scale;
    let cr = (chroma.g - coef.chroma_offset) * coef.chroma_scale;
    let kg = 1.0 - coef.kr - coef.kb;
    let red = y + 2.0 * (1.0 - coef.kr) * cr;
    let green = y - 2.0 * (1.0 - coef.kb) * coef.kb / kg * cb - 2.0 * (1.0 - coef.kr) * coef.kr / kg * cr;
    let blue = y + 2.0 * (1.0 - coef.kb) * cb;
    return vec4<f32>(clamp(vec3<f32>(red, green, blue), vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
