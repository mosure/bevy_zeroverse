// A coarse texel whose footprint reaches foreground geometry or uncovered
// pixels cannot supply radiance. Refine just the missing bilinear weight.
fn frost_valid(texel: vec4<f32>, pane_z: f32) -> vec4<f32> {
    let valid = texel.a >= 0.0 && texel.a * 1.001 + 0.00000006 <= pane_z;
    return select(vec4<f32>(0.0), vec4<f32>(texel.rgb, 1.0), valid);
}

fn frost_weighted_level(uv: vec2<f32>, level: i32, pane_z: f32) -> vec4<f32> {
    let dims = vec2<i32>(textureDimensions(view_bindings::view_transmission_texture, level));
    let pos = uv * vec2<f32>(dims) - 0.5;
    let base = vec2<i32>(floor(pos));
    let lo = clamp(base, vec2<i32>(0), dims - 1);
    let hi = clamp(base + 1, vec2<i32>(0), dims - 1);
    let a = frost_valid(textureLoad(view_bindings::view_transmission_texture, lo, level),pane_z);
    let b = frost_valid(textureLoad(view_bindings::view_transmission_texture, vec2<i32>(hi.x,lo.y), level),pane_z);
    let c = frost_valid(textureLoad(view_bindings::view_transmission_texture, vec2<i32>(lo.x,hi.y), level),pane_z);
    let d = frost_valid(textureLoad(view_bindings::view_transmission_texture, hi, level),pane_z);
    let f = fract(pos);
    return mix(mix(a,b,f.x),mix(c,d,f.x),f.y);
}

fn frost_weighted_sample(pixel: vec2<f32>, full_size: vec2<f32>, lod: f32, pane_z: f32) -> TransmissionTap {
    var sum = vec4<f32>(0.0);
    let high = i32(ceil(lod));
    let first_weight = 1.0 - (f32(high) - lod);
    for (var level = high; level > 0; level -= 1) {
        let weight = select(1.0 - sum.a, first_weight, level == high);
        sum += weight * frost_weighted_level(pixel / full_size, level, pane_z);
        if sum.a >= 0.99999 {return TransmissionTap(sum, 0.0);}
    }
    let fine = transmission_base(pixel, full_size, pane_z);
    let missing = 1.0 - sum.a;
    return TransmissionTap(sum + missing * fine.color, missing * fine.missing);
}
