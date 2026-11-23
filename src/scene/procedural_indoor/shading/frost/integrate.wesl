// Each stratum filters the background over its share of the scattering cone.
fn filtered_frost(offset_position: vec2<f32>, frag_coord: vec3<f32>, full_size: vec2<f32>, radius_px: f32, taps: i32) -> vec4<f32> {
    var result = vec4<f32>(0.0);
    var missing = 0.0;
    var on_screen = 0.0;
    // Opposite pairs have an exactly zero first moment. A fixed low-discrepancy
    // disk stratifies the original Ultra kernel's empirical radial distribution,
    // including both checkerboard radii, without modulating adjacent pixels.
    let log_radius = log2(radius_px);
    let last_level = f32(textureNumLevels(view_bindings::view_transmission_texture) - 1u);
    let pairs = taps / 2;
    for (var i = 0; i < pairs; i += 1) {
        let cell = frost_cell(i,pairs);
        let delta = cell.xy * radius_px / full_size;
        let diameter = radius_px * cell.z;
        let lod = clamp(log_radius + cell.w, 0.0, last_level);
        for (var side = 0; side < 2; side += 1) {
            var pixel = (offset_position + select(-delta, delta, side == 1)) * full_size;
            let origin = view_bindings::view.viewport.xy;
            let end = origin + view_bindings::view.viewport.zw;
            var coverage = 1.0;
            if lod > 0.0 {
                // A broad stratum crossing the screen edge is partly covered.
                // Integrate the clipped part at its centroid.
                let half_span = 0.5 * diameter;
                let lo = max(pixel - half_span, origin);
                let hi = min(pixel + half_span, end);
                let span = hi - lo;
                if any(span <= vec2<f32>(0.0)) {continue;}
                coverage = (span.x/diameter) * (span.y/diameter);
                pixel = 0.5*(lo+hi);
            } else if any(pixel < origin) || any(pixel >= end) {continue;}
            let tap = frost_weighted_sample(pixel, full_size, lod, frag_coord.z);
            result += coverage * tap.color;
            missing += coverage * tap.missing;
            on_screen += coverage;
        }
    }
    return resolve_transmission(result, missing, on_screen, f32(taps), frag_coord);
}
