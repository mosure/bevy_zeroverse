#import bevy_pbr::{
    mesh_view_bindings::globals,
    prepass_utils,
    forward_io::VertexOutput,
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

    // TODO: calculate video_jam optical flow rgb normalization

    let motion_vector = bevy_pbr::prepass_utils::prepass_motion_vector(
        in.position,
        sample_index,
    );
    return vec4<f32>(motion_vector / globals.delta_time, 0.0, 1.0);
}
