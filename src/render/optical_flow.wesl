import bevy_pbr::{
    render::forward_io::VertexOutput,
    render::mesh_view_bindings::globals,
    render::mesh_view_bindings::view,
    prepass::utils::prepass_motion_vector,
};
import bevy_render::color_operations::hsv_to_rgb;
import bevy_render::maths::PI_2;

// seconds, full-scale pixel displacement, temporal-history validity, reserved
@group(constants::MATERIAL_BIND_GROUP) @binding(100) var<uniform> preview: vec4<f32>;

@fragment
fn fragment(
@if(MULTISAMPLED)
    @builtin(sample_index) sample_index: u32,
    in: VertexOutput,
) -> @location(0) vec4<f32> {
@if(!MULTISAMPLED)
    let sample_index = 0u;

@if(MOTION_VECTOR_PREPASS) {
    // Bevy stores frame displacement. Estimate velocity using the actual render
    // interval, then display displacement over a fixed reference interval.
    // Exported dataset flow uses the independent exact capture-time pipeline.
    let motion_vector_uv = prepass_motion_vector(in.position, sample_index);
    let scale = select(0.0, preview.x / max(globals.delta_time, 0.000001),
        globals.delta_time > 0.000001 && preview.z > 0.5);
    let flow = motion_vector_uv * view.viewport.zw * scale;
}
@else {
    let flow = vec2<f32>(0.0);
}

    let radius = length(flow);
    // Map magnitude to [0,1] with a gentle curve so small motions stay visible
    // and large motions do not immediately clamp.
    let normalized = clamp(radius / preview.y, 0.0, 1.0);
    let m = pow(normalized, 0.65);
    var angle = atan2(flow.y, flow.x);
    if (angle < 0.0) {
        angle += PI_2;
    }
    let rgb = hsv_to_rgb(vec3<f32>(angle, m, 1.0));
    return vec4<f32>(rgb, 1.0);
}
