#import bevy_pbr::{
    forward_io::VertexOutput,
    mesh_view_bindings::view,
    prepass_utils,
}


struct PositionSettings {
    min: vec4<f32>,
    max: vec4<f32>,
}
@group(2) @binding(102) var<uniform> position_settings: PositionSettings;


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
    let normalized_position = (position - position_settings.min.xyz) / (position_settings.max.xyz - position_settings.min.xyz);
    return vec4<f32>(normalized_position, 1.0);
}
