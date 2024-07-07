#import bevy_pbr::{
    forward_io::VertexOutput,
    mesh_view_bindings::view,
}


@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let view_normal = normalize((view.view_from_world * vec4<f32>(in.world_normal, 0.0)).xyz);
    let normal_color = (view_normal * 0.5) + vec3<f32>(0.5, 0.5, 0.5);
    return vec4<f32>(normal_color, 1.0);
}
