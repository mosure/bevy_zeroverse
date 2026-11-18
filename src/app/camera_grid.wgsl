// RGBA32Float capture targets are not filterable on every WebGPU adapter.
// Work in display-linear color and composite a partially covered fisheye disk
// with straight alpha only after integrating its premultiplied radiance.
@group(1) @binding(0) var captured: texture_2d<f32>;
@group(1) @binding(1) var<uniform> filter_rgb: u32;

@fragment
fn fragment(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let size = vec2<i32>(textureDimensions(captured));
    let center = uv * vec2<f32>(size);
    let width = max(fwidth(center), vec2<f32>(1.0));
    if filter_rgb == 0u {
        return textureLoad(captured, clamp(vec2<i32>(floor(center)), vec2<i32>(0), size - vec2<i32>(1)), 0);
    }
    let lo = center - 0.5 * width;
    let hi = center + 0.5 * width;
    let first = vec2<i32>(floor(lo));
    let span = vec2<i32>(ceil(hi)) - first;
    // Exact texel coverage through 12 taps per axis. Bound the cost of
    // exceptionally small thumbnails by stratifying the entire footprint.
    let taps = min(span, vec2<i32>(12));
    var sum = vec4<f32>(0.0);
    for (var y = 0; y < taps.y; y++) {
        let y0 = max(lo.y, f32(first.y + y * span.y / taps.y));
        let y1 = min(hi.y, f32(first.y + (y+1) * span.y / taps.y));
        for (var x = 0; x < taps.x; x++) {
            let x0 = max(lo.x, f32(first.x + x * span.x / taps.x));
            let x1 = min(hi.x, f32(first.x + (x+1) * span.x / taps.x));
            let p = vec2<i32>(floor(0.5 * vec2<f32>(x0+x1, y0+y1)));
            let c = textureLoad(captured, clamp(p, vec2<i32>(0), size - vec2<i32>(1)), 0);
            sum += (x1-x0) * (y1-y0) * vec4<f32>(c.rgb*c.a, c.a);
        }
    }
    return vec4<f32>(sum.rgb / max(sum.a, 1e-12), sum.a / (width.x * width.y));
}
