// A half-float render target can contain infinity or NaN. Inspect the f32
// exponent after load, before multiplication: even zero weight times NaN
// can contaminate the entire scattering cone. -1 denotes missing radiance.
fn finite_radiance(value: vec4<f32>) -> bool {
    let exponent = vec4<u32>(0x7f800000u);
    return all((bitcast<vec4<u32>>(value) & exponent) != exponent);
}

fn checked_radiance(value: vec4<f32>) -> vec4<f32> {
    return select(vec4<f32>(0.0, 0.0, 0.0, -1.0), value, finite_radiance(value));
}
