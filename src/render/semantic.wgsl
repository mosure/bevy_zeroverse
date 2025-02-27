#import bevy_pbr::{
    forward_io::VertexOutput,
    mesh_view_bindings::view,
}


@group(2) @binding(100)
var<uniform> color: vec4<f32>;


@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(color.rgb, 1.0);
}
