// RGB is associated with background coverage. Missing weight is an in-frame foreground occluder
// or unusable radiance. Sky and off-screen rays can still receive the environment.
struct TransmissionTap {
    color: vec4<f32>,
    missing: f32,
}

fn associate_background(color: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(color.rgb * color.a, color.a);
}

@if(DEPTH_PREPASS && !WEBGL2)
fn foreground_depth(pixel: vec2<i32>) -> f32 {
@if(MULTISAMPLED) {
    // Resolved color can contain a foreground sample even at an MSAA edge.
    var nearest = 0.0;
    let count = i32(textureNumSamples(view_bindings::depth_prepass_texture));
    for (var sample = 0; sample < count; sample++) {
        nearest = max(nearest, textureLoad(view_bindings::depth_prepass_texture, pixel, sample));
    }
    return nearest;
} @else {
    // The transmission sampler is filtering. WebGPU requires a non-filtering
    // sampler for a depth gather, so address the same four texels explicitly.
    return textureLoad(view_bindings::depth_prepass_texture, pixel, 0);
}
}

// Match radiance and reverse depth at all four texels of the bilinear
// footprint. The usual all-background case retains hardware color filtering.
fn transmission_base(pixel: vec2<f32>, full_size: vec2<f32>, pane_z: f32) -> TransmissionTap {
    let viewport = view_bindings::view.viewport;
    let center = clamp(pixel, viewport.xy + 0.5, viewport.xy + viewport.zw - 0.5);
    let uv = center / full_size;
    var eligible = vec4<bool>(true);
    let base = vec2<i32>(floor(center - 0.5));
    let last = vec2<i32>(viewport.xy + viewport.zw) - 1;
    let right = min(base + vec2<i32>(1,0), last);
    let below = min(base + vec2<i32>(0,1), last);
    let diagonal = min(base + 1, last);
@if(DEPTH_PREPASS && !WEBGL2) {
    let depths = vec4<f32>(foreground_depth(base), foreground_depth(right),
        foreground_depth(below), foreground_depth(diagonal));
    eligible = depths <= vec4<f32>(pane_z);
    if !any(eligible) {return TransmissionTap(vec4<f32>(0.0), 1.0);}
}
    if all(eligible) {
        let filtered = textureSampleLevel(view_bindings::view_transmission_texture,
            view_bindings::view_transmission_sampler, uv, 0.0);
        if finite_radiance(filtered) {return TransmissionTap(associate_background(filtered), 0.0);}
    }
    let a = checked_radiance(textureLoad(view_bindings::view_transmission_texture, base, 0));
    let b = checked_radiance(textureLoad(view_bindings::view_transmission_texture, right, 0));
    let c = checked_radiance(textureLoad(view_bindings::view_transmission_texture, below, 0));
    let d = checked_radiance(textureLoad(view_bindings::view_transmission_texture, diagonal, 0));
    let valid = eligible & (vec4<f32>(a.a,b.a,c.a,d.a) >= vec4<f32>(0.0));
    let f = fract(center - 0.5);
    let weights = vec4<f32>(1.0-f.x, f.x, 1.0-f.x, f.x)
        * vec4<f32>(1.0-f.y, 1.0-f.y, f.y, f.y);
    let visible = select(vec4<f32>(0.0), weights, valid);
    let color = visible.x * associate_background(a) + visible.y * associate_background(b)
        + visible.z * associate_background(c) + visible.w * associate_background(d);
    return TransmissionTap(color, dot(select(weights, vec4<f32>(0.0), valid), vec4<f32>(1.0)));
}

fn resolve_transmission(sum: vec4<f32>, missing: f32, on_screen: f32,
                        taps: f32, frag_coord: vec3<f32>) -> vec4<f32> {
    var color = sum;
    if missing > 0.0 {
        // An object between the eye and pane cannot darken the transmitted
        // scene. Estimate blocked rays from the observed part of the cone.
        let observed = max(on_screen - missing, 0.0);
        var estimate = sum / max(observed, 1.0);
        if observed < 1.0 {
            // With less than one tap of evidence, anchor the estimate to the
            // opaque background at the visible pane fragment before refraction.
            let anchor = transmission_base(floor(frag_coord.xy) + 0.5,
                vec2<f32>(textureDimensions(view_bindings::view_transmission_texture)), frag_coord.z);
            estimate += (1.0 - observed) * anchor.color;
        }
        color += missing * estimate;
    }
    if color.a <= 0.0 {return vec4<f32>(0.0);}
    var result = vec4<f32>(color.rgb / color.a, clamp(color.a / taps, 0.0, 1.0));
@if(TONEMAP_IN_SHADER) {
    // Bevy's approximate inverse divides by the length of tone-mapped RGB.
    // True black, or a signal mapped to black by the display curve, is valid.
    let mapped = length(tone_mapping(result, view_bindings::view.color_grading).rgb);
    if mapped > 0.000001 {result = vec4<f32>(result.rgb * (length(result.rgb) / mapped), result.a);}
}
    return result;
}

fn fetch_transmissive_background_non_rough(offset_position: vec2<f32>, frag_coord: vec3<f32>) -> vec4<f32> {
    let full_size = vec2<f32>(textureDimensions(view_bindings::view_transmission_texture));
    let pixel = offset_position * full_size;
    let viewport = view_bindings::view.viewport;
    if any(pixel < viewport.xy) || any(pixel >= viewport.xy + viewport.zw) {return vec4<f32>(0.0);}
    let tap = transmission_base(pixel, full_size, frag_coord.z);
    return resolve_transmission(tap.color, tap.missing, 1.0, 1.0, frag_coord);
}
