struct Palette { colors: array<vec4<f32>, 16>, count: vec4<u32> }
@group(0) @binding(0) var<uniform> palette: Palette;
@group(0) @binding(1) var annotation: texture_2d<f32>;

@vertex
fn vertex(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}
fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    return select(pow((c + 0.055) / 1.055, vec3<f32>(2.4)), c / 12.92, c <= vec3<f32>(0.04045));
}
@fragment
fn fragment(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let value = textureLoad(annotation, vec2<i32>(position.xy), 0);
    let mask = u32(value.x);
    var rgb = vec3<f32>(0.0);
    for (var i = 0u; i < palette.count.x; i++) {
        if (mask & (1u << i)) != 0u { rgb += palette.colors[i].xyz; }
    }
    // UI composition performs the final linear-to-sRGB conversion. Keep the
    // declared RGB8 additive code intact; never send bit masks through PBR/HDR.
    return vec4<f32>(srgb_to_linear(rgb), 1.0);
}
