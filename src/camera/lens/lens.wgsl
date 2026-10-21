#define_import_path zeroverse::lens

struct Lens {
    k: vec4<f32>, // fx, fy, cx, cy
    sensor: vec4<f32>, // skew, width, height, angular flag
    coefficients: vec4<f32>,
    domain: vec4<f32>, // maximum polar angle
}
fn radius(l: Lens, theta: f32) -> f32 {
    let x = theta * theta;
    return theta * (1.0 + x * (l.coefficients.x + x * (l.coefficients.y + x * (l.coefficients.z + x * l.coefficients.w))));
}
// xyz unit direction; w marks a pixel inside the calibrated angular domain.
fn ray(l: Lens, pixel: vec2<f32>) -> vec4<f32> {
    let y = (pixel.y - l.k.w) / l.k.y;
    let x = (pixel.x - l.k.z - l.sensor.x * y) / l.k.x;
    if l.sensor.w == 0.0 { return vec4<f32>(normalize(vec3<f32>(x, -y, -1.0)), 1.0); }
    let r = length(vec2<f32>(x,y));
    if r > radius(l, l.domain.x) + 0.0000001 { return vec4<f32>(0.0); }
    if r < 0.0000001 { return vec4<f32>(0.0,0.0,-1.0,1.0); }
    var lo = 0.0; var hi = l.domain.x;
    for (var i=0u; i<25u; i++) {
        let mid = 0.5 * (lo+hi);
        if radius(l, mid) < r { lo=mid; } else { hi=mid; }
    }
    let theta=0.5*(lo+hi);
    return vec4<f32>(vec2<f32>(x,-y) * (sin(theta)/r), -cos(theta), 1.0);
}
// xy pixel coordinates, z model depth, w valid including image/domain bounds.
fn project(l: Lens, p: vec3<f32>) -> vec4<f32> {
    var a: vec2<f32>; var depth = -p.z;
    if l.sensor.w == 0.0 {
        if depth <= 0.0 { return vec4<f32>(0.0); }
        a = vec2<f32>(p.x,-p.y)/depth;
    } else {
        depth = length(p);
        let transverse=length(p.xy);
        if depth <= 0.0 || (transverse == 0.0 && p.z >= 0.0) { return vec4<f32>(0.0); }
        let theta=atan2(transverse,-p.z);
        if theta > l.domain.x + 0.0000001 { return vec4<f32>(0.0); }
        a=vec2<f32>(p.x,-p.y) * (radius(l,theta)/max(transverse,0.000000001));
    }
    let pixel=vec2<f32>(l.k.x*a.x+l.sensor.x*a.y+l.k.z, l.k.y*a.y+l.k.w);
    let valid=all(pixel >= vec2<f32>(0.0)) && all(pixel < l.sensor.yz);
    return vec4<f32>(pixel,depth,select(0.0,1.0,valid));
}
