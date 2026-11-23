// Deterministic, symmetric disk quadrature. Pixel-random rotations and the
// alternating checkerboard radius create visible grain in a *single* capture;
// they cannot converge when the dataset deliberately disables temporal jitter.
fn fetch_transmissive_background(offset_position: vec2<f32>, frag_coord: vec3<f32>, view_z: f32, perceptual_roughness: f32) -> vec4<f32> {
    let full_size = vec2<f32>(textureDimensions(view_bindings::view_transmission_texture));
    let aspect = full_size.x / full_size.y;
    let viewport_scale = view_bindings::view.viewport.z / full_size.x;
    let radius = viewport_scale * perceptual_roughness * perceptual_roughness / max(abs(view_z), 0.001);
    // A subpixel cone is already integrated by the bilinear texture footprint.
    // Typical clear glazing needs one fetch instead of 32 noisy near-identical ones.
    if radius * full_size.x < 0.5 {
        return fetch_transmissive_background_non_rough(offset_position, frag_coord);
    }
@if(SCREEN_SPACE_SPECULAR_TRANSMISSION_BLUR_TAPS)
    let taps = 2 * constants::SCREEN_SPACE_SPECULAR_TRANSMISSION_BLUR_TAPS;
@else
    let taps = 16;

    // FROST_MIP_FILTER
    var result = vec4<f32>(0.0);
    var missing = 0.0;
    var on_screen = 0.0;
    // Opposite pairs have an exactly zero first moment. A fixed low-discrepancy
    // disk stratifies the original Ultra kernel's empirical radial distribution,
    // including both checkerboard radii, without modulating adjacent pixels.
    let radial_cdf = FROST_RADIAL_CDF;
    let pairs = taps / 2;
    for (var i = 0; i < pairs; i += 1) {
        let q = clamp((f32(i) + 0.5) / f32(pairs) * 64.0 - 0.5, 0.0, 63.0);
        let r = mix(radial_cdf[u32(q)], radial_cdf[min(u32(q) + 1u, 63u)], fract(q));
        let angle = f32(i) * 2.39996323;
        let delta = vec2(cos(angle), sin(angle) * aspect) * r * radius;
        for (var side = 0; side < 2; side += 1) {
            let uv = offset_position + select(-delta, delta, side == 1);
            // Reject outside the viewport instead of smearing its last texel.
            let pixel = uv * full_size;
            let origin = view_bindings::view.viewport.xy;
            if any(pixel < origin) || any(pixel >= origin + view_bindings::view.viewport.zw) {
                continue;
            }
            let tap = transmission_base(pixel, full_size, frag_coord.z);
            result += tap.color;
            missing += tap.missing;
            on_screen += 1.0;
        }
    }
    return resolve_transmission(result, missing, on_screen, f32(taps), frag_coord);
}
