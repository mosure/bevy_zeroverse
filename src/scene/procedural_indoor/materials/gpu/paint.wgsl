// Wall paint/plaster: identical metric fields to paint/prepared.rs. Kept
// separate from the fabric/device kernels; a map is baked once per recipe.
fn paint_disk(distance: f32, radius: f32, footprint: f32) -> f32 {
    let aa=max(footprint,0.002);let filtered=max(radius,aa);
    return smooth_curve((filtered+aa-distance)/(2.*aa))*radius*radius/(filtered*filtered+aa*aa*0.20);
}
fn paint_cell(uv: vec2<f32>, n: u32) -> vec2<f32> {
    let xy=vec2(unit(uv.x),unit(uv.y))*f32(n);let base=vec2<i32>(floor(xy));
    var nearest=1e30;var dye=0.;
    for(var dy=-1;dy<=1;dy+=1) { for(var dx=-1;dx<=1;dx+=1) {
        let c=base+vec2(dx,dy);let h=vec2(wrap_i(c.x,n),wrap_i(c.y,n));
        let p=vec2<f32>(c)+vec2(0.05)+0.90*vec2(hash(h.x,h.y,503u),hash(h.x,h.y,534u));
        let delta=p-xy;let distance=delta.x*delta.x+delta.y*delta.y;
        if distance<nearest {nearest=distance;dye=hash(h.x,h.y,574u);}
    }}
    return vec2(sqrt(nearest),dye);
}
fn paint_srgb(c: f32) -> f32 {
    if c<=0.0031308 {return c*12.92;}return 1.055*pow(c,1./2.4)-0.055;
}
fn paint(uv: vec2<f32>) -> Texel {
    let c=r.paint_counts;let f=r.paint_frequencies;
    let cell=paint_cell(uv,c.x);let splat=paint_disk(cell.x,0.20+cell.y*0.28,r.paint_color.w);
    let plateau=min(smooth_curve(splat*1.4),1.-r.paint0.y*0.55);
    let roller=noise(uv,c.y,c.z,509u)-0.5;let film=noise(uv,c.w,c.w,521u)-0.5;
    let brush=noise(uv,f.x,5u,523u)-0.5;
    let warped=uv+(vec2(noise(uv,3u,5u,541u),noise(uv,5u,3u,558u))-vec2(0.5))*0.12;
    let passes=0.64*noise(warped,f.y,f.y,588u)+0.25*noise(warped,f.y*2u,f.y*2u,602u)+0.11*noise(warped,f.y*4u,f.y*4u,624u);
    let lap=smooth_curve((passes-0.35)*1.5);let repair=smooth_curve((passes-0.65)*6.)*r.paint2.z;
    let holes=paint_disk(cell.x,0.08,r.paint_color.w)*smooth_curve((r.paint1.x-cell.y)*10.);
    let texture=(plateau-0.28)*r.paint0.x*(1.-repair);
    let gain=1.+(passes-0.5)*r.paint1.y-holes*0.035;
    let linear=clamp(r.paint_color.xyz*gain,vec3(0.),vec3(1.));
    let color=vec3(paint_srgb(linear.x),paint_srgb(linear.y),paint_srgb(linear.z));
    let height=r.metric.z*(texture*0.80+(lap-0.5)*r.paint0.w*0.22-holes*0.40)
        +0.000025*(roller*r.paint0.z+film*r.paint1.w)*r.paint2.w+0.000016*brush*r.paint2.x*r.paint2.w;
    return Texel(color,height,clamp(r.metric.w+film*0.025+roller*r.paint0.z*0.035+(lap-0.5)*r.paint2.y+holes*0.04-repair*r.paint1.z*0.10,0.16,0.99),max(1.-holes*0.025,0.97));
}
@compute @workgroup_size(8,8)
fn paint_map(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy>=vec2(r.header.y)) {return;}store_texel(id.xy,paint(domain(id.xy)));
}
