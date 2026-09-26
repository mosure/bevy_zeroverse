// Native geometric annotations bypass HDR, tonemapping and all color intermediates.
struct CameraUniform {
    clip_from_world: mat4x4<f32>,
    view_from_world: mat4x4<f32>,
    limits: vec4<f32>,
}

struct Instance {
    world_from_local: mat4x4<f32>,
    normal_from_local: mat4x4<f32>,
    semantic: u32,
    _padding0: u32,
    _padding1: u32,
    _padding2: u32,
}

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var<storage, read> instances: array<Instance>;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) instance: u32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) view_normal: vec3<f32>,
    @location(2) linear_depth: f32,
    @location(3) @interpolate(flat) semantic: u32,
}

@vertex
fn vertex(input: VertexInput) -> VertexOutput {
    let instance = instances[input.instance];
    let world_position = instance.world_from_local * vec4<f32>(input.position, 1.0);
    let world_normal = normalize((instance.normal_from_local * vec4<f32>(input.normal, 0.0)).xyz);
    let view_position = camera.view_from_world * world_position;
    var output: VertexOutput;
    output.clip_position = camera.clip_from_world * world_position;
    output.world_position = world_position.xyz;
    output.view_normal = (camera.view_from_world * vec4<f32>(world_normal, 0.0)).xyz;
    output.linear_depth = -view_position.z;
    output.semantic = instance.semantic;
    return output;
}

struct GroundTruthOutput {
    @location(0) world_depth: vec4<f32>,
    @location(1) normal_semantic: vec4<f32>,
}

@fragment
fn fragment(input: VertexOutput) -> GroundTruthOutput {
    if input.linear_depth < camera.limits.x || input.linear_depth > camera.limits.y {
        discard;
    }
    var output: GroundTruthOutput;
    output.world_depth = vec4<f32>(input.world_position, input.linear_depth);
    output.normal_semantic = vec4<f32>(normalize(input.view_normal) * 0.5 + vec3<f32>(0.5), f32(input.semantic));
    return output;
}
