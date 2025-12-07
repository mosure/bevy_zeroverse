#import bevy_pbr::{
    forward_io::VertexOutput,
    mesh_view_bindings::globals,
    mesh_view_bindings::view,
    prepass_utils::prepass_motion_vector,
}
#import bevy_render::color_operations::hsv_to_rgb
#import bevy_render::maths::PI_2

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

#ifdef MOTION_VECTOR_PREPASS
    // Motion vectors are stored as UV offsets per frame; convert to pixels per frame.
    let motion_vector_uv = prepass_motion_vector(in.position, sample_index);
    let flow = motion_vector_uv * view.viewport.zw;
#else
    let flow = vec2<f32>(0.0);
#endif

    let radius = length(flow);
    // Map magnitude to [0,1] with a gentle curve so small motions stay visible
    // and large motions do not immediately clamp.
    let normalized = clamp(radius / 32.0, 0.0, 1.0);
    let m = pow(normalized, 0.65);
    var angle = atan2(flow.y, flow.x);
    if (angle < 0.0) {
        angle += PI_2;
    }
    let rgb = hsv_to_rgb(vec3<f32>(angle, m, 1.0));
    return vec4<f32>(rgb, 1.0);
}
