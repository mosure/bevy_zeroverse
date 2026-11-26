// Forward surface correspondence, rasterized on the SOURCE image grid.
struct Camera {
    source_clip: mat4x4<f32>,
    target_clip: mat4x4<f32>,
    source_view: mat4x4<f32>,
    target_view: mat4x4<f32>,
    source_limits: vec4<f32>, // near, far, width, height
    target_limits: vec4<f32>,
}
@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var target_world_depth: texture_2d<f32>;

struct Input {
    @location(0) source: vec3<f32>,
    @location(1) target_position: vec3<f32>,
    @location(2) valid: f32,
}
struct Output {
    @builtin(position) position: vec4<f32>,
    @location(0) target_position: vec3<f32>,
    @location(1) source_depth: f32,
    @location(2) @interpolate(flat) valid: f32,
    @location(3) clip_delta_and_target_w: vec4<f32>,
}
@vertex
fn vertex(input: Input) -> Output {
    var out: Output;
    out.position = camera.source_clip * vec4<f32>(input.source, 1.0);
    let projected = camera.target_clip * vec4<f32>(input.target_position, 1.0);
    out.clip_delta_and_target_w = vec4<f32>(projected.xyw - out.position.xyw, projected.w);
    out.source_depth = -(camera.source_view * vec4<f32>(input.source, 1.0)).z;
    out.target_position = input.target_position;
    out.valid = input.valid;
    return out;
}
@fragment
fn fragment(input: Output) -> @location(0) vec4<f32> {
    let delta = input.clip_delta_and_target_w;
    let target_depth = -(camera.target_view * vec4<f32>(input.target_position, 1.0)).z;
    // A plane test compares depth at the target_position texel center, rather than
    // incorrectly comparing depths along two different rays on sloped surfaces.
    let plane_normal = normalize(cross(dpdx(input.target_position), dpdy(input.target_position)));
    if input.source_depth < camera.source_limits.x || input.source_depth > camera.source_limits.y {
        discard;
    }
    if input.valid == 0.0 || delta.w <= 0.0 || target_depth < camera.target_limits.x || target_depth > camera.target_limits.y {
        return vec4<f32>(0.0);
    }
    // Subtract homogeneous endpoints before perspective interpolation. Reprojecting
    // an absolute interpolated world point can invent motion on thin, grazing
    // triangles even with identical geometry, optics and camera pose.
    let source_ndc = (input.position.xy / camera.source_limits.zw - 0.5) * vec2<f32>(2.0, -2.0);
    let ndc_delta = (delta.xy - source_ndc * delta.z) / delta.w;
    let pixel_scale = camera.target_limits.zw / camera.source_limits.zw;
    let flow = input.position.xy * (pixel_scale - 1.0)
        + ndc_delta * vec2<f32>(0.5, -0.5) * camera.target_limits.zw;
    let target_pixel = input.position.xy + flow;
    var visible = 0.0;
    if all(target_pixel >= vec2<f32>(0.0)) && all(target_pixel < camera.target_limits.zw) {
        let hit = textureLoad(target_world_depth, vec2<i32>(target_pixel), 0);
        let tolerance = max(0.001, 0.0001 * target_depth);
        if hit.w > 0.0 && abs(dot(hit.xyz - input.target_position, plane_normal)) <= tolerance {
            visible = 1.0;
        }
    }
    return vec4<f32>(flow, 1.0, visible);
}
