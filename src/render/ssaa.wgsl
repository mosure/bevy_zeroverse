// Exact box/area average of an integer-sized pixel footprint. RGBA32Float
// inputs are linear light; there is no sRGB-encoded averaging or bilinear tap.
struct Params { factor: u32, normal: u32, padding: vec2<u32> }
@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var<uniform> params: Params;
@vertex
fn vertex(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}
@fragment
fn fragment(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let origin = vec2<i32>(position.xy) * i32(params.factor);
    var value = vec4<f32>(0.0);
    for (var y = 0u; y < params.factor; y++) {
        for (var x = 0u; x < params.factor; x++) {
            value += textureLoad(source, origin + vec2<i32>(i32(x), i32(y)), 0);
        }
    }
    value /= f32(params.factor * params.factor);
    if params.normal != 0u {
        let n = value.xyz * 2.0 - vec3<f32>(1.0);
        if dot(n, n) > 1e-12 { value = vec4<f32>(normalize(n) * 0.5 + vec3<f32>(0.5), value.w); }
    }
    return value;
}
