// RGB only, after lens projection and tone mapping. Offsets are in final-sensor
// pixels; the source can be supersampled. All neighbors retain lens coverage.

struct Sensor {
    header: vec4<u32>, blur: vec4<f32>, smudge: vec4<f32>, scatter: vec4<f32>,
    ghost: vec4<f32>, tint: vec4<f32>, sensor: vec4<f32>,
}
@group(0) @binding(0) var image: texture_2d<f32>;
@group(0) @binding(1) var<uniform> p: Sensor;

fn tap(pixel: vec2<f32>) -> vec4<f32> {
    // Manual bilinear works with unfilterable RGBA32Float on WebGPU as well.
    let hi = vec2<i32>(textureDimensions(image)) - 1;
    let q = clamp(pixel * f32(p.header.x) - 0.5, vec2(0.), vec2<f32>(hi));
    let a = vec2<i32>(floor(q));
    let b = min(a + 1, hi);
    let f = fract(q);
    return mix(mix(textureLoad(image, a, 0), textureLoad(image, vec2(b.x,a.y), 0), f.x),
               mix(textureLoad(image, vec2(a.x,b.y), 0), textureLoad(image, b, 0), f.x), f.y);
}
fn perpendicular(v: vec2<f32>) -> vec2<f32> { return vec2(-v.y,v.x); }
fn color(s: vec4<f32>, fallback: vec3<f32>) -> vec3<f32> {
    return select(fallback, s.rgb / max(s.a, 1e-6), s.a > 1e-6);
}
// Resolve contrast of bright sources on a coarse sensor grid. Neighboring
// fragments reuse these texels. Interpolation defocuses optical ghosts and
// prevents a second sharp image of window mullions or ceiling fixtures.
fn highlight(pos: vec2<f32>, size: vec2<f32>) -> vec2<f32> {
    if any(pos < vec2(0.)) || any(pos >= size) { return vec2(0.); }
    let raw = textureLoad(image, vec2<i32>(pos*f32(p.header.x)), 0);
    let luminance = dot(color(raw,vec3(0.)), vec3(0.2126,0.7152,0.0722));
    return vec2(smoothstep(p.ghost.w,1.,luminance)*raw.a,raw.a);
}
fn source_cell(center: vec2<f32>, step_px: f32, size: vec2<f32>) -> f32 {
    let axis = p.scatter.xy * step_px;
    let cross_axis = perpendicular(axis);
    let a = axis*0.37;
    let b = cross_axis*0.37;
    let near = highlight(center,size)*2. + highlight(center+a,size) + highlight(center-a,size)
        + highlight(center+b,size) + highlight(center-b,size);
    let c = (axis+cross_axis)*1.65;
    let d = (axis-cross_axis)*1.65;
    let outer = highlight(center+c,size) + highlight(center-c,size)
        + highlight(center+d,size) + highlight(center-d,size);
    let mean = near.x / max(near.y,1e-6);
    let surround = select(mean,outer.x / max(outer.y,1e-6),outer.y>0.);
    return max(mean-surround,0.) * (near.y/6.);
}
fn defocused_highlight(pos: vec2<f32>, step_px: f32, size: vec2<f32>) -> f32 {
    let q = pos / step_px - 0.5;
    let corner = (floor(q)+0.5)*step_px;
    let d = vec2(step_px,0.);
    let t = smoothstep(vec2(0.),vec2(1.),fract(q));
    return mix(mix(source_cell(corner,step_px,size),source_cell(corner+d,step_px,size),t.x),
        mix(source_cell(corner+d.yx,step_px,size),source_cell(corner+d+d.yx,step_px,size),t.x),t.y);
}
fn hash(key: u32) -> u32 {
    var h = key;
    h = (h ^ (h >> 16u)) * 0x7feb352du;
    h = (h ^ (h >> 15u)) * 0x846ca68bu;
    return h ^ (h >> 16u);
}
fn unit_noise(key: u32) -> f32 {
    let h = hash(key);
    let sum = (h & 255u) + ((h >> 8u) & 255u) + ((h >> 16u) & 255u) + (h >> 24u);
    // Four uniform bytes; zero mean and unit variance.
    return (f32(sum) - 510.) * 0.006765875;
}
@fragment
fn sensor_fragment(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let px = pos.xy;
    let base = area_average(image, vec2<i32>(px), p.header.x);
    if base.a <= 0. { return base; }
    let size = vec2<f32>(textureDimensions(image)) / f32(p.header.x);
    let short_edge = min(size.x, size.y);
    let scale = short_edge / 512.;
    var rgb = color(base, vec3(0.));

    if p.blur.z > 0. {
        let major = p.blur.xy * p.blur.z * scale;
        let minor = perpendicular(p.blur.xy) * p.blur.w * scale;
        // Centered, normalized PSF. For minuscule blur, retain the exact box
        // resolve rather than replacing it with a sample of the center texel.
        let neighbors = tap(px+major) + tap(px-major) + tap(px+minor) + tap(px-minor);
        rgb = mix(rgb, color(neighbors,rgb), 0.76 * min(p.blur.z*scale,1.));
    }
    if p.scatter.z > 0. {
        let delta = (px-p.smudge.xy*size) / short_edge;
        let axis = p.scatter.xy;
        let q = vec2(dot(delta,axis),dot(delta,perpendicular(axis))) / p.smudge.zw;
        let mask = (1. - smoothstep(0.,1.,dot(q,q))) * p.scatter.z;
        if mask > 0. {
            let reach = p.scatter.w * scale;
            let along = axis * reach;
            let across = perpendicular(axis) * (0.33*reach);
            let veil = tap(px+along) + tap(px-along) + tap(px+across) + tap(px-across);
            rgb = mix(rgb, color(veil,rgb), mask);
        }
    }
    if p.tint.w > 0. {
        let center = p.ghost.xy * size;
        let src = center + (center-px) / p.ghost.z;
        let radius = max(2.,(14.+p.sensor.z)*scale);
        if all(src >= vec2(-radius)) && all(src < size+radius) {
            let light = defocused_highlight(src,radius,size);
            rgb += (vec3(1.)-clamp(rgb,vec3(0.),vec3(1.))) * p.tint.rgb * (light*p.tint.w);
        }
    }
    if p.sensor.x+p.sensor.y > 0. {
        let key = hash(p.header.y ^ (u32(px.x)*0x9e3779b9u) ^ (u32(px.y)*0x85ebca6bu));
        let n = vec3(unit_noise(key),unit_noise(key+1u),unit_noise(key+2u));
        rgb += n * sqrt(vec3(p.sensor.x*p.sensor.x) + p.sensor.y*p.sensor.y * max(rgb,vec3(0.)));
    }
    return vec4(clamp(rgb,vec3(0.),vec3(1.))*base.a,base.a);
}
