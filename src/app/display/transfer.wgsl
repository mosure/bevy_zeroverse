// Bevy composites in display-linear Rec.709. Encode only at the physical
// surface, after scene and UI rendering. Float scRGB needs no transfer pass.
@group(0) @binding(0) var linear_image: texture_2d<f32>;

@vertex fn fullscreen(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let corner = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    return vec4<f32>(corner * 2.0 - 1.0, 0.0, 1.0);
}

@fragment fn extended_srgb(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
    let rgb = textureLoad(linear_image, vec2<i32>(p.xy), 0).rgb;
    let a = abs(rgb);
    let encoded = select(1.055 * pow(a, vec3(1.0/2.4)) - 0.055, a * 12.92, a <= vec3(0.0031308));
    return vec4(sign(rgb) * encoded, 1.0);
}

@fragment fn bt2100_pq(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
    let rgb = textureLoad(linear_image, vec2<i32>(p.xy), 0).rgb;
    // Matrix columns transform linear Rec.709 to Rec.2020, both D65.
    let gamut = mat3x3<f32>(vec3(0.627404, 0.069097, 0.016391),
                            vec3(0.329283, 0.919540, 0.088013),
                            vec3(0.043313, 0.011363, 0.895596));
    // scRGB reference white is 80 cd/m². ST 2084 is absolute up to 10000 cd/m².
    let v = pow(clamp(gamut * rgb * (80.0/10000.0), vec3(0.0), vec3(1.0)), vec3(2610.0/16384.0));
    let pq = pow((3424.0/4096.0 + (2413.0/128.0)*v) / (1.0 + (2392.0/128.0)*v), vec3(2523.0/32.0));
    return vec4(pq, 1.0);
}

@fragment fn linear_copy(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
    return textureLoad(linear_image, vec2<i32>(p.xy), 0);
}
