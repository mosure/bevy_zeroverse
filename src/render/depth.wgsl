#import bevy_pbr::{
    forward_io::VertexOutput,
    mesh_view_bindings::view,
    prepass_utils,
}


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

    let n = view.frustum[5].xyz;
    let d = view.frustum[5].w;

    let far = dot(n, view.world_position) + d;
    let frag_to_cam = view.world_position - in.world_position.xyz;

#ifdef RAY_DEPTH
    let depth = length(frag_to_cam);
#else ifdef Z_DEPTH
    let depth = dot(n, frag_to_cam);
#endif

#ifdef COLORIZED_DEPTH
    let prepass_depth = bevy_pbr::prepass_utils::prepass_depth(
        in.position,
        sample_index,
    );
    return vec4<f32>(depth_to_rgb(prepass_depth), 1.0);
#else ifdef NORMALIZED_DEPTH
    return vec4<f32>(vec3<f32>(depth / far), 1.0);
#else ifdef LINEAR_DEPTH
    return vec4<f32>(vec3<f32>(depth), 1.0);
#endif
}
