#import bevy_pbr::forward_io::VertexOutput


fn depth_to_rgb(depth: f32) -> vec3<f32> {
    let normalized_depth = clamp(depth, 0.0, 1.0);

    let r = smoothstep(0.5, 1.0, normalized_depth);
    let g = 1.0 - abs(normalized_depth - 0.5) * 2.0;
    let b = 1.0 - smoothstep(0.0, 0.5, normalized_depth);

    return vec3<f32>(r, g, b);
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(vec3<f32>(in.position.w), 1.0);
}
