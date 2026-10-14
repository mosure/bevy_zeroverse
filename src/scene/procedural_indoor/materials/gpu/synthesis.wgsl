// Explicit CPU/WGSL recipe layout. One-shot bake; no procedural fragment work.
struct Recipe {
    header: vec4<u32>, surface: vec4<u32>, metric: vec4<f32>, phase: vec4<f32>,
    topology: vec4<u32>, cloth0: vec4<f32>, cloth1: vec4<f32>, cloth2: vec4<f32>,
    tint0: vec4<f32>, tint1: vec4<f32>, dye_runs: vec4<f32>, dye_phase: vec4<f32>,
    dye_tint: vec4<f32>, leather0: vec4<f32>, leather1: vec4<f32>,
    carpet: vec4<f32>, carpet_meta: vec4<u32>,
    carpet0: vec4<f32>, carpet1: vec4<f32>, carpet2: vec4<f32>,
    floor_meta: vec4<u32>, floor0: vec4<f32>, floor1: vec4<f32>, floor_tint: vec4<f32>,
}
struct Level { size: u32, offset: u32, stride: u32, moment_offset: u32 }
struct Dispatch { level: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<uniform> r: Recipe;
@group(0) @binding(1) var<storage, read_write> heights: array<f32>;
@group(0) @binding(2) var<storage, read_write> moments: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> pixels: array<u32>;
@group(0) @binding(4) var<storage, read> srgb: array<f32>;
@group(0) @binding(5) var<uniform> dispatch: Dispatch;

fn level_info(l: u32) -> Level {
    var n = r.header.y; var offset = 0u; var moment = 0u;
    for(var i = 0u; i < l; i += 1u) { offset += n * max(n, 64u); moment += n*n; n /= 2u; }
    return Level(n, offset, max(n, 64u), moment);
}
fn plane_size() -> u32 {
    var n = r.header.y; var offset = 0u;
    loop { offset += n * max(n, 64u); if n == 1u { break; } n /= 2u; }
    return offset;
}
fn unpack(v: u32) -> vec3<f32> { return vec3<f32>(f32(v & 255u), f32((v >> 8u) & 255u), f32((v >> 16u) & 255u)); }
fn pack(v: vec3<u32>) -> u32 { return v.x | (v.y << 8u) | (v.z << 16u) | 0xff000000u; }
fn quant(v: vec3<f32>) -> vec3<u32> { return vec3<u32>(clamp(v, vec3(0.), vec3(255.))); }
fn encode_normal(v: vec3<f32>) -> u32 { return pack(quant(floor((v * 0.5 + vec3(0.5))*255. + vec3(0.5)))); }
fn smooth_curve(t: f32) -> f32 { let x = clamp(t, 0., 1.); return x*x*(3.-2.*x); }
fn unit(v: f32) -> f32 { let x = v % 1.; return select(x, x+1., x < 0.); }
fn seed(salt: u32) -> vec2<u32> { let lo = r.header.z + salt; return vec2(lo, r.header.w + select(0u, 1u, lo < r.header.z)); }
fn hash(x: u32, y: u32, salt: u32) -> f32 {
    let s = seed(salt);
    var v = (x*374761393u) ^ (y*668265263u) ^ s.x ^ (s.y*0x9e3779b9u);
    v = (v ^ (v >> 13u))*1274126177u;
    return f32(v ^ (v >> 16u)) / 4294967295.;
}
fn noise(uv: vec2<f32>, nx: u32, ny: u32, salt: u32) -> f32 {
    let xy = vec2(unit(uv.x), unit(uv.y)) * vec2<f32>(f32(nx), f32(ny));
    let i = vec2<u32>(floor(xy)); let t = vec2(smooth_curve(fract(xy.x)), smooth_curve(fract(xy.y)));
    let a = hash(i.x%nx, i.y%ny, salt); let b = hash((i.x+1u)%nx, i.y%ny, salt);
    let c = hash(i.x%nx, (i.y+1u)%ny, salt); let d = hash((i.x+1u)%nx, (i.y+1u)%ny, salt);
    return (a+(b-a)*t.x)*(1.-t.y)+(c+(d-c)*t.x)*t.y;
}
fn wrap_i(v: i32, n: u32) -> u32 { let m = i32(n); return u32(((v%m)+m)%m); }
fn dye(axis: u32, index: u32) -> vec3<f32> {
    if r.dye_tint.w == 0. { return vec3(1.); }
    let count = r.dye_runs[axis]; let yarns = f32(r.surface[axis+2u]);
    let at = unit((f32(index)+0.5)/yarns*count + r.dye_phase[axis]);
    let width = count/yarns*0.5;
    let mask = smooth_curve((r.dye_runs[axis+2u]*0.5+width-abs(at-0.5))/(2.*width));
    return vec3(1.) + (r.dye_tint.xyz-vec3(1.))*mask*r.dye_phase[axis+2u];
}
struct Texel { color: vec3<f32>, height: f32, roughness: f32, ao: f32 }
fn crown(p: f32, w: f32) -> f32 { let q = abs(p/(w*0.5)); return sqrt(max(1.-q*q,0.))*(1.-smooth_curve((q-0.82)/0.18)); }
fn ridge(t: f32) -> f32 { return 1.-2.*abs(unit(t)-0.5); }
fn woven(uv: vec2<f32>) -> Texel {
    let n = r.surface.zw; let xy = uv*vec2<f32>(n); let i = vec2<u32>(floor(xy))%n;
    let slub = vec2(noise(uv,n.x,3u,11u),noise(uv,3u,n.y,17u));
    let jitter = vec2(hash(i.x,0u,0u),hash(i.y,1u,0u))-vec2(0.5);
    let p = fract(xy)-vec2(0.5)-jitter*0.05;
    let width = r.cloth0.xy*(vec2(1.)+r.cloth0.z*(slub-vec2(0.5)));
    let c = vec2(crown(p.x,width.x),crown(p.y,width.y));
    var row = i.y/r.topology.z;
    if (r.topology.w >> 16u) != 0u { let t = row%(2u*r.topology.x); row = min(t,2u*r.topology.x-1u-t); }
    let over = select(0.,1.,(i.x/r.topology.z+row*(r.topology.w&65535u))%r.topology.x < r.topology.y);
    let h = vec2(c.x*(1.-r.cloth0.w*c.y*(1.-over)), c.y*(1.-r.cloth0.w*c.x*over));
    var top = 0.5; if h.x+h.y > 0.0001 { top = h.x/(h.x+h.y); }
    let coverage = max(c.x,c.y);
    var bands = vec2(1.);
    for(var axis=0u;axis<2u;axis+=1u) {
        let count = u32(r.cloth2[axis+2u]);
        if count>0u && (i[axis]*count/n[axis])%3u==0u { bands[axis] = 1.-r.cloth2.y*0.25; }
    }
    let a = r.tint0.xyz*(0.94+r.cloth1.w*(hash(i.x,13u,0u)-0.5))*bands.x*dye(0u,i.x);
    let b = r.tint1.xyz*(0.94+r.cloth1.w*(hash(i.y,19u,0u)-0.5))*bands.y*dye(1u,i.y);
    let twist = r.cloth1.x*(noise(uv,3u,3u,83u)-0.5);
    let fibres = top*ridge(xy.x*2.+twist)+(1.-top)*ridge(xy.y*2.-twist);
    let nap = noise(uv,79u,83u,73u)-0.5;
    let yarn = clamp(vec3(0.76)+(a*top+b*(1.-top)-vec3(0.76))*coverage+vec3(nap*r.cloth1.y*0.035), vec3(0.25),vec3(1.));
    let height = r.metric.z*(max(h.x,h.y)-0.5+fibres*0.045*(1.-r.cloth1.y)+nap*r.cloth1.y*0.09);
    let pile = noise(uv,37u,41u,101u);
    return Texel(clamp(yarn*(1.-r.cloth2.x*0.08*(pile-0.5)),vec3(0.),vec3(1.)),
        height*(1.-r.cloth2.x*0.45)+r.cloth2.x*r.metric.z*(pile-0.5),
        clamp(r.metric.w+r.cloth1.y*nap*0.10+(1.-coverage)*0.10-r.cloth1.z*(fibres-0.5)*0.06,0.30,1.),0.93+0.07*coverage);
}
fn knit(uv: vec2<f32>) -> Texel {
    let n = r.surface.zw; let p = vec2(unit(uv.x),unit(uv.y))*vec2<f32>(n); let cell = vec2<i32>(floor(p));
    let q = p-floor(p)-vec2(0.5); var closest = vec3(1e30,0.,0.); var course = 0u;
    let controls = array<vec2<f32>,9>(vec2(-0.35,-0.7),vec2(-0.02,-0.17),vec2(-0.23,0.18),
        vec2(-0.23,0.18),vec2(0.,0.76),vec2(0.23,0.18),vec2(0.23,0.18),vec2(0.02,-0.17),vec2(0.35,-0.7));
    for(var row=-1;row<=1;row+=1) { for(var col=-1;col<=1;col+=1) {
        let j = hash(wrap_i(cell.x+col,n.x),wrap_i(cell.y+row,n.y),0u);
        let width=0.075+r.cloth0.x*0.095+r.cloth0.z*(j-0.5)*0.09;
        let local=q-vec2(f32(col)+(j-0.5)*0.05,f32(row));
        for(var k=0u;k<9u;k+=3u) { let a=controls[k];let b=controls[k+1u];let c=controls[k+2u];var previous=a;
            for(var i=1u;i<=4u;i+=1u) { let t=f32(i)*0.25;
                // glam lerp is self*(1-s)+rhs*s.
                let ab=a*(1.-t)+b*t;let bc=b*(1.-t)+c*t;let next=ab*(1.-t)+bc*t;
                let d=next-previous;let f=clamp(dot(local-previous,d)/dot(d,d),0.,1.);
                let delta=local-(previous+d*f);let distance=dot(delta,delta)/(width*width);
                if distance<closest.x { closest=vec3(distance,j,local.y);course=wrap_i(cell.y+row,n.y); }
                previous=next;
            }
        }
    }}
    let c=sqrt(max(1.-closest.x,0.));let nap=noise(uv,71u,73u,127u)-0.5;
    let yarn=0.79+c*(0.18+r.cloth1.w*(closest.y-0.5))+r.cloth1.y*nap*0.03;
    return Texel(clamp(r.tint0.xyz*yarn*dye(1u,course),vec3(0.),vec3(1.)),
        r.metric.z*(c*(0.85+r.cloth0.w*closest.z*0.15)-0.45),
        clamp(r.metric.w+(1.-c)*0.09+nap*r.cloth1.y*0.08,0.4,1.),0.89+c*0.11);
}
fn leather(uv: vec2<f32>) -> Texel {
    let n=r.surface.zw;
    let xy=(uv+vec2(noise(uv,3u,5u,11u)-0.5,noise(uv,5u,3u,19u)-0.5)*0.008)*vec2<f32>(n);
    let i=vec2<i32>(floor(xy));var nearest=vec2(1e30);var cell=vec2(0u);
    for(var dy=-1;dy<=1;dy+=1) {for(var dx=-1;dx<=1;dx+=1) {
        let c=i+vec2(dx,dy);let h=vec2(wrap_i(c.x,n.x),wrap_i(c.y,n.y));
        let p=vec2<f32>(c)+vec2(0.15)+0.70*vec2(hash(h.x,h.y,0u),hash(h.x,h.y,31u));
        let delta=p-xy;let d=delta.x*delta.x+delta.y*delta.y;
        if d<nearest.x { nearest=vec2(d,nearest.x);cell=h; } else if d<nearest.y { nearest.y=d; }
    }}
    let edge=smooth_curve(clamp((sqrt(nearest.y)-sqrt(nearest.x))*9.,0.,1.));
    let c=pow(clamp(1.-nearest.x*0.50,0.,1.),r.leather0.x);let grain=edge*(0.55+0.45*c);
    let pore=smooth_curve((noise(uv,83u,79u,47u)-0.68)*5.);let crease=1.-smooth_curve(abs(noise(uv,7u,9u,71u)-0.48)*24.);
    let dye=0.93+(hash(cell.x,cell.y,91u)-0.5)*r.leather0.w*0.35+(noise(uv,11u,13u,101u)-0.5)*r.leather0.w-pore*0.035;
    let nap=noise(uv,97u,91u,113u)-0.5;
    return Texel(vec3(clamp(dye+nap*r.leather1.x*0.08,0.25,1.)),
        r.metric.z*((grain-0.5)*(1.-r.leather1.x*0.8)-pore*r.leather0.y-crease*r.leather0.z+nap*r.leather1.x*0.20),
        clamp(r.metric.w+(1.-grain)*0.16*(1.-r.leather1.x)+pore*0.08+crease*0.04+nap*r.leather1.x*0.12,0.18,1.),0.95+0.05*grain);
}
fn device_finish(uv: vec2<f32>) -> Texel {
    let macro_n=noise(uv,3u,3u,1u);let meso=noise(uv,17u,19u,2u);let micro=noise(uv,97u,91u,3u);
    var value=micro*0.72+meso*0.28;var variation=0.14;
    if r.surface.x==7u { value=noise(uv,103u,5u,44u)*0.78+micro*0.22;variation=0.035; }
    if r.surface.x==6u { value=micro*0.80+meso*0.20; }
    if r.surface.x==22u { variation=0.34; }
    let stain=max((macro_n-0.5)*3.,0.)*r.phase.z;
    let shade=0.96+(value-0.5)*r.phase.w-stain;
    return Texel(vec3(clamp(shade,0.25,1.)),(value-0.5)*r.metric.z,
        clamp(r.metric.w+(value-0.5)*variation+(macro_n-0.5)*r.phase.z+stain*0.3,0.045,1.),1.);
}
fn domain(id: vec2<u32>) -> vec2<f32> {
    var uv=vec2<f32>(id)/f32(r.header.y);
    switch r.surface.y {case 1u:{ uv=vec2(uv.y,1.-uv.x); } case 2u:{uv=vec2(1.)-uv;} case 3u:{uv=vec2(1.-uv.y,uv.x);} default:{} }
    return vec2(unit(uv.x+r.phase.x),unit(uv.y+r.phase.y));
}
fn carpet(uv: vec2<f32>) -> Texel {
    let counts = vec2<f32>(r.carpet_meta.yz);
    let xy = fract(uv * counts);
    let width = max(counts.x, counts.y) / f32(r.header.y);
    let line = r.carpet.w + (noise(uv,17u,11u,271u)-0.5)*r.carpet0.w;
    var mask = noise(uv, 7u, 9u, 241u);
    switch r.carpet_meta.x {
        case 1u: { mask = smooth_curve((line - abs(xy.x - 0.5)) / width + 0.5); }
        case 2u: { mask = smooth_curve((line*0.4 - min(abs(xy.x-0.5), abs(xy.y-0.5))) / width + 0.5); }
        case 3u: { mask = smooth_curve((line*0.7 - abs(abs(xy.x-0.5) + abs(xy.y-0.5) - 0.55)) / width + 0.5); }
        case 4u: { let cell = vec2<u32>(floor(uv * counts)); mask = hash(cell.x, cell.y, 251u); }
        default: {}
    }
    let nap = noise(uv, 173u, 191u, 257u) - 0.5;
    let slub = noise(uv, 11u, 83u, 263u) - 0.5;
    let fleck = noise(uv, 43u, 47u, 269u) - 0.5;
    let dye = clamp(mask * r.carpet.x, 0., 1.);
    let tone = r.carpet.y * (slub * 0.6 + fleck * 0.4) + nap * 0.035;
    let base = r.carpet0.xyz * (1.-dye) + r.carpet1.xyz * dye;
    let stipple = max(fleck*2., 0.) * r.carpet.y;
    var t = Texel(clamp(base*(1.-stipple) + r.carpet2.xyz*stipple + tone*base, vec3(0.01), vec3(0.95)),
        r.metric.z*(nap*0.65 + fleck*r.carpet.z*0.35),
        clamp(r.metric.w + nap*0.055 + fleck*0.025, 0.80, 1.), 0.97+(nap+0.5)*0.03);
    if r.floor_meta.x != 0u {
        let grid = vec2<f32>(r.floor_meta.zw);
        let u = unit(uv.x) * grid.x;
        let x = u32(floor(u)) % r.floor_meta.z;
        var stagger = 0.;
        if r.floor_meta.y != 0u { stagger = hash(x, 7u, 211u); }
        let v = unit(uv.y) * grid.y + stagger;
        let y = u32(floor(v)) % r.floor_meta.w;
        let gain = 1. + (hash(x,y,223u)-0.5)*r.floor0.x + (noise(uv,3u,3u,227u)-0.5)*r.floor0.w;
        let warm = (hash(x,y,229u)-0.5)*r.floor0.y;
        var tint = vec3(1.);
        if hash(x,y,233u) < r.floor1.x { tint = r.floor_tint.xyz; }
        let color = clamp(t.color * gain * vec3(1.+warm,1.,1.-warm) * tint, vec3(0.),vec3(0.97));
        t.color += (color-t.color)*r.floor1.y;
        t.roughness = clamp(t.roughness + (hash(x,y,239u)-0.5)*r.floor0.z,0.12,1.);
        let local = fract(vec2(u,v));
        let metric = r.metric.xx / grid;
        let distance = min(min(local.x,1.-local.x)*metric.x,min(local.y,1.-local.y)*metric.y);
        let seam_width = r.floor1.z + r.metric.x / f32(r.header.y);
        let seam = clamp(1.-distance/seam_width,0.,1.)*r.floor1.z/seam_width*r.floor1.y;
        t.color *= 1.-seam*0.12;
        t.height -= seam*0.00035;
    }
    return t;
}
fn store_texel(id: vec2<u32>, t: Texel) {
    let i=id.y*r.header.y+id.x;let plane=plane_size();heights[i]=t.height;
    pixels[i]=pack(quant(clamp(t.color,vec3(0.),vec3(1.))*255.));
    pixels[2u*plane+i]=pack(vec3(u32(floor(t.ao*255.+0.5)),u32(clamp(t.roughness,0.05,1.)*255.),select(0u,255u,r.surface.x==7u)));
}
// Separate entry points let the driver discard unrelated substrate programs
// and their registers; the texture field and recipe remain identical.
@compute @workgroup_size(8,8)
fn woven_map(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy>=vec2(r.header.y)) {return;}store_texel(id.xy,woven(domain(id.xy)));
}
@compute @workgroup_size(8,8)
fn leather_map(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy>=vec2(r.header.y)) {return;}store_texel(id.xy,leather(domain(id.xy)));
}
@compute @workgroup_size(8,8)
fn knit_map(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy>=vec2(r.header.y)) {return;}store_texel(id.xy,knit(domain(id.xy)));
}
@compute @workgroup_size(8,8)
fn finish_map(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy>=vec2(r.header.y)) {return;}store_texel(id.xy,device_finish(domain(id.xy)));
}
@compute @workgroup_size(8,8)
fn carpet_map(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy>=vec2(r.header.y)) {return;}store_texel(id.xy,carpet(domain(id.xy)));
}
@compute @workgroup_size(8,8)
fn normals(@builtin(global_invocation_id) id: vec3<u32>) {
    let n=r.header.y;if id.x>=n || id.y>=n{return;}let x=id.x;let y=id.y;let i=y*n+x;
    let dx=heights[y*n+(x+1u)%n]-heights[y*n+(x+n-1u)%n];
    let dy=heights[((y+1u)%n)*n+x]-heights[((y+n-1u)%n)*n+x];
    let slope=vec2(f32(n)*0.5)/r.metric.xy;
    let encoded=encode_normal(normalize(vec3(-dx*slope.x,-dy*slope.y,1.)));
    let plane=plane_size();pixels[plane+i]=encoded;
    let decoded=normalize(unpack(encoded)/127.5-vec3(1.));let rough=f32((pixels[2u*plane+i]>>8u)&255u)/255.;
    moments[i]=vec4(decoded,(rough*rough)*(rough*rough));
}
fn to_srgb(v: f32) -> u32 {
    var lo=0u;var hi=256u;
    loop { if lo+1u>=hi {break;} let mid=(lo+hi)/2u; if v>=srgb[256u+mid] {lo=mid;}else{hi=mid;} }
    return lo;
}
@compute @workgroup_size(8,8)
fn mips(@builtin(global_invocation_id) id: vec3<u32>) {
    let dst=level_info(dispatch.level);if id.x>=dst.size || id.y>=dst.size{return;}
    let src=level_info(dispatch.level-1u);let plane=plane_size();var m=vec4(0.);var sums=vec2(0u);var color=vec3(0.);
    let offsets=array<vec2<u32>,4>(vec2(0u,0u),vec2(1u,0u),vec2(0u,1u),vec2(1u,1u));
    for(var k=0u;k<4u;k+=1u){let xy=id.xy*2u+offsets[k];let i=src.offset+xy.y*src.stride+xy.x;
        m+=moments[src.moment_offset+xy.y*src.size+xy.x]*0.25;
        let data=pixels[2u*plane+i];sums+=vec2(data&255u,(data>>16u)&255u);
        let c=pixels[i];color+=vec3(srgb[c&255u],srgb[(c>>8u)&255u],srgb[(c>>16u)&255u]);
    }
    let len=clamp(length(m.xyz),0.0001,1.);let variance=(1.-len)/len;let rough=sqrt(sqrt(min(m.w+variance,1.)));
    let i=dst.offset+id.y*dst.stride+id.x;color*=0.25;
    pixels[i]=pack(vec3(to_srgb(color.x),to_srgb(color.y),to_srgb(color.z)));
    pixels[plane+i]=encode_normal(m.xyz/len);
    pixels[2u*plane+i]=pack(vec3(sums.x/4u,u32(floor(rough*255.+0.5)),sums.y/4u));
    moments[dst.moment_offset+id.y*dst.size+id.x]=m;
}
