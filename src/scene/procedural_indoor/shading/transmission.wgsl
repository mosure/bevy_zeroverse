// Deterministic, symmetric disk quadrature. Pixel-random rotations and the
// alternating checkerboard radius create visible grain in a *single* capture;
// they cannot converge when the dataset deliberately disables temporal jitter.
fn fetch_transmissive_background(offset_position: vec2<f32>, frag_coord: vec3<f32>, view_z: f32, perceptual_roughness: f32) -> vec4<f32> {
    let full_size = vec2<f32>(textureDimensions(view_bindings::view_transmission_texture));
    let aspect = view_bindings::view.viewport.z / view_bindings::view.viewport.w;
    let radius = perceptual_roughness * perceptual_roughness / max(abs(view_z), 0.001);
    // A subpixel cone is already integrated by the bilinear texture footprint.
    // Typical clear glazing needs one fetch instead of 32 noisy near-identical ones.
    if radius * full_size.x < 0.5 {
        return fetch_transmissive_background_non_rough(offset_position, frag_coord);
    }
#ifdef SCREEN_SPACE_SPECULAR_TRANSMISSION_BLUR_TAPS
    let taps = 2 * #{SCREEN_SPACE_SPECULAR_TRANSMISSION_BLUR_TAPS};
#else
    let taps = 16;
#endif
    var result = vec4<f32>(0.0);
    // Opposite pairs have an exactly zero first moment. A fixed low-discrepancy
    // disk stratifies the original Ultra kernel's empirical radial distribution,
    // including both checkerboard radii, without modulating adjacent pixels.
    let radial_cdf = array<f32,64>(
        0.05625000, 0.06250000, 0.07031250, 0.07812500, 0.08437500, 0.09375000, 0.09843750, 0.10937500, 0.11250000, 0.12500000, 0.14062500, 0.15625000, 0.16875000, 0.16875000, 0.18750000, 0.18750000, 0.19687500, 0.21093750, 0.21875000, 0.22500000, 0.23437500, 0.25000000, 0.25312500, 0.28125000, 0.28125000, 0.28125000, 0.29531250, 0.31250000, 0.31250000, 0.32812500, 0.33750000, 0.33750000, 0.35156250, 0.37500000, 0.37500000, 0.39062500, 0.39375000, 0.39375000, 0.42187500, 0.42187500, 0.43750000, 0.43750000, 0.45000000, 0.46875000, 0.46875000, 0.49218750, 0.49218750, 0.50000000, 0.50625000, 0.54687500, 0.54687500, 0.56250000, 0.56250000, 0.59062500, 0.59062500, 0.62500000, 0.65625000, 0.65625000, 0.67500000, 0.68906250, 0.75000000, 0.76562500, 0.78750000, 0.87500000
    );
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
#ifdef DEPTH_PREPASS
#ifndef WEBGL2
            if prepass_utils::prepass_depth(vec4<f32>(pixel, 0.0, 0.0), 0u) > frag_coord.z {
                continue;
            }
#endif
#endif
            let sample = textureSampleLevel(view_bindings::view_transmission_texture,
                view_bindings::view_transmission_sampler, uv, 0.0);
            // Linear radiance, including bright sources. No brightness-dependent
            // clipping or normalize(black): both bias the optical reference.
            result += vec4(sample.rgb * sample.a, sample.a);
        }
    }
    // Return unassociated colour plus coverage. The caller mixes with the
    // environment using this alpha; dividing RGB by taps as well would apply
    // coverage twice and create a dark halo beside foreground frames/objects.
    result = vec4(result.rgb / max(result.a, 0.000001), result.a / f32(taps));
#ifdef TONEMAP_IN_SHADER
    result = approximate_inverse_tone_mapping(result, view_bindings::view.color_grading);
#endif
    return result;
}
