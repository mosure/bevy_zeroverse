#import bevy_pbr::{ forward_io::VertexOutput, mesh_view_bindings::view }


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

    let ndc_depth = in.position.z / in.position.w;
    let normalized_depth = ndc_depth * 0.5 + 0.5;
    let frag_to_cam = view.world_position - in.world_position.xyz;
    let ray_depth = length(frag_to_cam);

#ifdef COLORIZED_DEPTH
    return vec4<f32>(depth_to_rgb(normalized_depth), 1.0);
#else ifdef NORMALIZED_DEPTH
    return vec4<f32>(vec3<f32>(normalized_depth), 1.0);
#else ifdef LINEAR_DEPTH
    return vec4<f32>(vec3<f32>(ray_depth), 1.0);
#endif
}
