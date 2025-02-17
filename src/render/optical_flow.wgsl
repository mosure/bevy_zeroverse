#import bevy_pbr::{
    forward_io::VertexOutput,
    mesh_view_bindings::globals,
    mesh_view_bindings::view,
    prepass_utils,
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

    let motion_vector = bevy_pbr::prepass_utils::prepass_motion_vector(
        in.position,
        sample_index,
    );
    let flow = motion_vector / globals.delta_time;

    let radius = length(flow);
    var angle = atan2(flow.y, flow.x);
    if (angle < 0.0) {
        angle += PI_2;
    }

    // let sigma: f32 = 0.15;
    // let norm_factor = sigma * length(view.viewport.zw);
    // let m = clamp(radius / norm_factor, 0.0, 1.0);
    let m = clamp(radius, 0.0, 1.0);

    let rgb = hsv_to_rgb(vec3<f32>(angle, m, 1.0));
    return vec4<f32>(rgb, 1.0);
}
