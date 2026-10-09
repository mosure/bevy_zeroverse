import bevy_pbr::{
    render::forward_io::VertexOutput,
    render::mesh_view_bindings::view,
    render::pbr_bindings::material,
};

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(material.base_color.rgb, 1.0);
}
