// CPU counterparts: carpet/dye.rs and flooring/stock.rs. Bake once; no added
// textures, fragment shading or per-frame work.
fn carpet_dye_mask(uv: vec2<f32>) -> f32 {
    let a = noise(uv, r.carpet_dye.y, r.carpet_dye.z, 743u);
    let b = noise(uv, r.carpet_dye.z, r.carpet_dye.y, 751u);
    let nx = f32(r.carpet_meta.y); let ny = f32(r.carpet_meta.z);
    let row = u32(floor(uv.y * ny)) % r.carpet_meta.z;
    let shift = hash(row, 0u, 757u) * r.carpet_mix.w;
    let shear = f32(i32(r.carpet_dye.w) - 1);
    let x = unit((uv.x + shear * uv.y) * nx + (a - 0.5) * r.carpet_mix.z);
    let y = unit(uv.y * ny);
    let aa = (nx * (1. + abs(shear)) + ny +
        f32(max(r.carpet_dye.y, r.carpet_dye.z)) * r.carpet_mix.z) / f32(r.header.y);
    let stripe = smooth_curve((r.carpet.w - abs(x - 0.5)) / aa + 0.5);
    let weave = smooth_curve((r.carpet.w * 0.45 -
        min(abs(fract(x + shift) - 0.5), abs(y - 0.5))) / aa + 0.5);
    let cloud = smooth_curve(a * 0.65 + b * 0.35);
    let stipple = smooth_curve((a - (0.38 + r.carpet.w * 0.5)) * 6.);
    let block = hash(u32(floor(uv.x * nx)) % r.carpet_meta.y, row, 761u);
    // Keep scalar association aligned with Rust's iterator sum.
    return clamp(a*r.carpet_weights.x + stripe*r.carpet_weights.y +
        weave*r.carpet_weights.z + cloud*r.carpet_weights.w +
        stipple*r.carpet_mix.x + block*r.carpet_mix.y, 0., 1.);
}
fn stock_boundary(i: i32, n: u32, jitter: f32, lane: u32, salt: u32) -> f32 {
    let index = u32(((i % i32(n)) + i32(n)) % i32(n));
    return f32(i) + (hash(index, lane, salt) * 2. - 1.) * jitter;
}
fn stock_interval(u: f32, n: u32, jitter: f32, lane: u32, salt: u32) -> vec3<f32> {
    let t = unit(u) * f32(n);
    var i = i32(floor(t));
    if t < stock_boundary(i,n,jitter,lane,salt) { i -= 1; }
    else if t >= stock_boundary(i+1,n,jitter,lane,salt) { i += 1; }
    let lo = stock_boundary(i,n,jitter,lane,salt);
    let width = stock_boundary(i+1,n,jitter,lane,salt) - lo;
    return vec3(f32(((i % i32(n)) + i32(n)) % i32(n)), clamp((t-lo)/width,0.,1.), width/f32(n));
}
struct StockCut { id: vec2<u32>, distance: f32 }
fn stock_floor_cut(uv: vec2<f32>) -> StockCut {
    let a = stock_interval(uv.x, r.floor_meta.z, r.floor_cut.x, 0u, 701u);
    var offset = 0.; var lane = 0u;
    if r.floor_meta.y != 0u { lane = u32(a.x); offset = hash(lane,7u,211u); }
    let b = stock_interval(uv.y + offset/f32(r.floor_meta.w),r.floor_meta.w,r.floor_cut.y,lane,709u);
    return StockCut(vec2(u32(a.x),u32(b.x)),r.metric.x*min(min(a.y,1.-a.y)*a.z,min(b.y,1.-b.y)*b.z));
}
