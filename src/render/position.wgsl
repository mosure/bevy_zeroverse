#import bevy_pbr::{
    forward_io::VertexOutput,
    mesh_view_bindings::view,
    pbr_bindings::material,
}


@fragment
fn fragment(
#ifdef MULTISAMPLED
    @builtin(sample_index) sample_index: u32,
#endif
    in: VertexOutput,
) -> @location(0) vec4<f32> {
#ifndef MULTISAMPLED
    let sample_index = 0u;
#endif

    let position = in.world_position.xyz;
    let min_extent = material.base_color.xyz;
    let max_extent = material.emissive.xyz;
    let range = max_extent - min_extent;
    let zero_mask = abs(range) < vec3<f32>(1e-5);
    let safe_range = select(range, vec3<f32>(1.0), zero_mask);
    let normalized_position = clamp((position - min_extent) / safe_range, vec3<f32>(0.0), vec3<f32>(1.0));
    return vec4<f32>(normalized_position, 1.0);
}
