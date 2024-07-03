#import bevy_pbr::forward_io::VertexOutput


@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let normal = (in.world_normal * 0.5) + vec3<f32>(0.5, 0.5, 0.5);
    return vec4<f32>(normal, 1.0);
}
