import bevy_pbr::{
    render::forward_io::VertexOutput,
    render::mesh_view_bindings::view,
    prepass::utils,
};


fn depth_to_rgb(depth: f32) -> vec3<f32> {
    let normalized_depth = clamp(depth, 0.0, 1.0);

    let r = smoothstep(0.5, 1.0, normalized_depth);
    let g = 1.0 - abs(normalized_depth - 0.5) * 2.0;
    let b = 1.0 - smoothstep(0.0, 0.5, normalized_depth);

    return vec3<f32>(r, g, b);
}


@fragment
fn fragment(
@if(MULTISAMPLED)
    @builtin(sample_index) sample_index: u32,
    in: VertexOutput,
) -> @location(0) vec4<f32> {
@if(!MULTISAMPLED)
    let sample_index = 0u;

    let n = view.frustum[5].xyz;
    let d = view.frustum[5].w;

    let far = dot(n, view.world_position) + d;
    let frag_to_cam = view.world_position - in.world_position.xyz;

@if(RAY_DEPTH)
    let depth = length(frag_to_cam);
@elif(Z_DEPTH)
    let depth = dot(n, frag_to_cam);

@if(COLORIZED_DEPTH)
    return vec4<f32>(depth_to_rgb(in.position.z), 1.0);
@elif(NORMALIZED_DEPTH)
    return vec4<f32>(vec3<f32>(depth / far), 1.0);
@elif(LINEAR_DEPTH)
    return vec4<f32>(vec3<f32>(depth), 1.0);
}
