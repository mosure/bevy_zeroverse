import bevy_pbr::{
    render::forward_io::VertexOutput,
    render::mesh_view_bindings::view,
    render::pbr_bindings::material,
};


@fragment
fn fragment(
@if(MULTISAMPLED)
    @builtin(sample_index) sample_index: u32,
    in: VertexOutput,
) -> @location(0) vec4<f32> {
@if(!MULTISAMPLED)
    let sample_index = 0u;

    let position = in.world_position.xyz;
    let min_extent = material.base_color.xyz;
    let max_extent = material.emissive.xyz;
    let range = max_extent - min_extent;
    let zero_mask = abs(range) < vec3<f32>(1e-5);
    let safe_range = select(range, vec3<f32>(1.0), zero_mask);
    // The AABB is the reconstruction region. Preserve visible context outside
    // it so min + value * range still recovers the actual world-space surface.
    let normalized_position = (position - min_extent) / safe_range;
    return vec4<f32>(normalized_position, 1.0);
}
