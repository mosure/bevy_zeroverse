struct Tap { addresses: vec4<u32>, blend: vec4<f32>, weight_lod: vec4<f32> };
struct Pixel { taps: u32, offset: u32, denominator: f32, kind: u32 };
@group(0) @binding(0) var<storage,read> source: array<vec4<f32>>;
@group(0) @binding(1) var<storage,read> taps: array<Tap>;
@group(0) @binding(2) var<storage,read> pixels: array<Pixel>;
@group(0) @binding(3) var<storage,read_write> output: array<vec2<u32>>;

fn lerp3(a: vec3<f32>, b: vec3<f32>, t: f32) -> vec3<f32> { return a*(1.-t) + b*t; }
fn bilinear(ab: u32, cd: u32, blend: vec2<f32>) -> vec3<f32> {
    let a = source[ab & 65535u].xyz; let b = source[ab >> 16u].xyz;
    let c = source[cd & 65535u].xyz; let d = source[cd >> 16u].xyz;
    return lerp3(lerp3(a,b,blend.x),lerp3(c,d,blend.x),blend.y);
}
@compute @workgroup_size(64)
fn convolve(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= arrayLength(&pixels) { return; }
    let p = pixels[id.x]; var value = vec3<f32>(0.);
    if p.kind == 0u { value = source[p.taps].xyz; }
    else {
        for(var i=0u; i<64u; i+=1u) {
            let t = taps[p.taps+i];
            var sample = lerp3(bilinear(t.addresses.x,t.addresses.y,t.blend.xy),
                bilinear(t.addresses.z,t.addresses.w,t.blend.zw),t.weight_lod.y);
            if p.kind == 2u { sample *= t.weight_lod.x; }
            value += sample;
        }
        value /= p.denominator;
    }
    value = clamp(value,vec3<f32>(0.),vec3<f32>(65000.));
    output[p.offset] = vec2<u32>(pack2x16float(value.xy),pack2x16float(vec2<f32>(value.z,1.)));
}
