// Reduce linear opaque radiance. Alpha carries the nearest reverse depth in
// each footprint. A negative bound means it includes nonopaque coverage and
// the transmission shader must consult level zero for that sample.
@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var depth: texture_depth_2d;

@vertex
fn fullscreen(@builtin(vertex_index) id: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((id << 1u) & 2u), f32(id & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

fn bound(a: f32, b: f32) -> f32 {
    return select(max(a, b), -1.0, min(a, b) < 0.0);
}

fn source_bound(alpha: f32, first: bool) -> f32 {
    if first {return select(-1.0, 0.0, alpha >= 1.0);}
    return alpha;
}

fn load_radiance(pixel: vec2<u32>, first: bool) -> vec4<f32> {
    let value = textureLoad(source, pixel, 0);
    if first {return checked_radiance(value);}
    return value;
}

fn reduce(pixel: vec2<u32>, first: bool) -> vec4<f32> {
    let src = textureDimensions(source, 0);
    let dst = max(src / 2u, vec2<u32>(1u));
    if all(src == dst * 2u) {
        let p = pixel * 2u;
        let a = load_radiance(p, first);
        let b = load_radiance(p + vec2<u32>(1,0), first);
        let c = load_radiance(p + vec2<u32>(0,1), first);
        let d = load_radiance(p + vec2<u32>(1,1), first);
        let z = bound(bound(source_bound(a.a,first),source_bound(b.a,first)),
                      bound(source_bound(c.a,first),source_bound(d.a,first)));
        return vec4<f32>((a.rgb+b.rgb+c.rgb+d.rgb)*0.25,z);
    }
    // Integer endpoints retain every last row/column for odd dimensions.
    let lower = pixel * src;
    let upper = lower + src;
    let start = lower / dst;
    let stop = (upper + dst - 1u) / dst;
    var rgb = vec3<f32>(0.0);
    var nearest = 0.0;
    for (var y = start.y; y < stop.y; y++) {
        for (var x = start.x; x < stop.x; x++) {
            let p = vec2<u32>(x, y);
            let overlap = min(upper, (p + 1u) * dst) - max(lower, p * dst);
            let sample = load_radiance(p, first);
            rgb += sample.rgb * (f32(overlap.x) / f32(src.x)) * (f32(overlap.y) / f32(src.y));
            nearest = bound(nearest, source_bound(sample.a, first));
        }
    }
    return vec4<f32>(rgb, nearest);
}

@fragment fn opaque_depth(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
    let pixel = vec2<u32>(p.xy);
    let color = reduce(pixel, true);
    if color.a < 0.0 { return color; }
    let src = textureDimensions(source, 0);
    let dst = max(src / 2u, vec2<u32>(1u));
    if all(src == dst * 2u) {
        let p0 = pixel * 2u;
        let a = textureLoad(depth, p0, 0);
        let b = textureLoad(depth, p0 + vec2<u32>(1,0), 0);
        let c = textureLoad(depth, p0 + vec2<u32>(0,1), 0);
        let d = textureLoad(depth, p0 + vec2<u32>(1,1), 0);
        return vec4<f32>(color.rgb,max(max(a,b),max(c,d)));
    }
    let lo = pixel * src / dst;
    let hi = ((pixel + 1u) * src + dst - 1u) / dst;
    var nearest = 0.0;
    for (var y = lo.y; y < hi.y; y++) {
        for (var x = lo.x; x < hi.x; x++) {
            nearest = max(nearest, textureLoad(depth, vec2<u32>(x,y), 0));
        }
    }
    return vec4<f32>(color.rgb, nearest);
}
@fragment fn opaque_color(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
    return reduce(vec2<u32>(p.xy), true);
}
@fragment fn previous_level(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
    return reduce(vec2<u32>(p.xy), false);
}
