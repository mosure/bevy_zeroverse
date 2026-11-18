struct Triangle { p: array<vec4<f32>,3>, n: array<vec4<f32>,3>, info: vec4<u32> }
@group(0) @binding(1) var<storage,read> triangles: array<Triangle>;
fn hit_attributes(index: u32) -> array<vec4<f32>,3> { return triangles[index].n; }
