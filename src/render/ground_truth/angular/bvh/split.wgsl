struct Triangle { p: array<vec4<f32>,3>, info: vec4<u32> }
struct Attributes { n: array<vec4<f32>,3> }
@group(0) @binding(1) var<storage,read> triangles: array<Triangle>;
@group(0) @binding(6) var<storage,read> attributes: array<Attributes>;
fn hit_attributes(index: u32) -> array<vec4<f32>,3> { return attributes[index].n; }
