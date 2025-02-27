#import bevy_pbr::{
    forward_io::VertexOutput,
    mesh_view_bindings::view,
    prepass_utils,
}


struct DepthSettings {
    linear_depth: vec4<f32>,
}
@group(2) @binding(102) var<uniform> depth_settings: DepthSettings;


fn depth_to_rgb(depth: f32) -> vec3<f32> {
    let normalized_depth = clamp(depth, 0.0, 1.0);

    let r = smoothstep(0.5, 1.0, normalized_depth);
    let g = 1.0 - abs(normalized_depth - 0.5) * 2.0;
    let b = 1.0 - smoothstep(0.0, 0.5, normalized_depth);

    return vec3<f32>(r, g, b);
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

    // if depth_settings.linear_depth.x > 0.5 {
        // TODO: fetch prepass depth
        // let depth = bevy_pbr::prepass_utils::prepass_depth(
        //     in.position,
        //     sample_index,
        // );
        // return vec4<f32>(vec3<f32>(depth), 1.0);
    // }

    let far_distance = dot(view.frustum[5].xyz, view.world_position) + view.frustum[5].w;
    let distance = length(view.world_position - in.world_position.xyz);
    return vec4<f32>(vec3<f32>(distance / far_distance), 1.0);

    // return vec4<f32>(vec3<f32>(in.position.z), 1.0);
    // return vec4<f32>(vec3<f32>(in.position.w), 1.0);
}
