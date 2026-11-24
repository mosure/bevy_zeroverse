// Furniture cut anatomy. Same metric height and finished reflectance as timber/figure.rs.
fn wood_lobe(uv: vec2<f32>, center: vec2<f32>, radius: vec2<f32>) -> f32 {
    let v = uv - center + vec2(0.5);
    let d = (vec2(unit(v.x),unit(v.y)) - vec2(0.5)) / radius;
    let inside = max(1. - d.x*d.x - d.y*d.y, 0.);
    return inside*inside;
}
fn wood_band(phase: f32, center: f32, width: f32) -> f32 {
    return smooth_curve(1. - abs(unit(phase - center + 0.5) - 0.5)/width);
}
fn furniture_timber(uv: vec2<f32>) -> Texel {
    let drift = noise(uv,4u,2u,311u) - 0.5;
    let knot = r.wood_pores.z * wood_lobe(uv,r.wood_center.zw,vec2(0.071,0.098));
    let track = uv.x + r.wood_cut.x*wood_lobe(uv,r.wood_center.xy,vec2(r.wood_cut.y,0.48))
        + r.wood_cut.z*drift + 0.075*knot;
    let axial = vec2(track,uv.y);
    let growth = noise(axial,7u,3u,317u) - 0.5;
    let age = track*f32(r.wood_counts.x) + growth*0.72;
    let year = wrap_i(i32(floor(age)), r.wood_counts.x);
    let phase = unit(age);
    let growth_rate = hash(year,11u,353u);
    let density = hash(year,23u,359u);
    let peak = 0.50 + growth_rate*0.23;
    let width = min(r.wood_cut.w*(0.60 + density*0.55),0.99-peak);
    let late = smooth_curve(1.-abs(phase-peak)/width)*(0.45 + density*0.77);
    let early = wood_band(phase,0.16,0.38);
    let fibres = noise(axial,89u,9u,331u) - 0.5;
    let vessel = smooth_curve((noise(axial,r.wood_counts.y,r.wood_counts.z,337u) - 0.56)*4.3)
        * r.wood_pores.x * (1. - r.wood_pores.y*(1.-early));
    let ray = smooth_curve((noise(axial,13u,81u,347u)-0.71)*5.)*r.wood_pores.w;
    let detail = r.wood_finish.w;
    let absorbed = clamp(r.wood_base.w*(1.+(0.28*vessel-0.07*late)*detail),0.,1.);
    let bundle = noise(axial,23u,4u,367u) - 0.5;
    let figure = (0.17-late)*(0.28+0.38*r.wood_pores.y+0.35*growth) + growth*0.52
        + bundle*0.64 + fibres*0.40 - 0.30*vessel + 0.22*ray - 0.73*knot;
    let gain = clamp(1.+figure*r.phase.w*r.wood_finish.y*1.65,0.48,1.45);
    let linear = clamp((r.wood_base.xyz*(1.-absorbed) + r.wood_stain.xyz*absorbed)*gain,vec3(0.),vec3(1.));
    let open_pores = 1. - r.wood_finish.x;
    return Texel(vec3(encode_srgb(linear.x),encode_srgb(linear.y),encode_srgb(linear.z)),
        r.metric.z*((0.09*late + 0.12*fibres)*open_pores - 0.86*vessel*open_pores - 0.07*knot),
        clamp(r.metric.w+detail*(1.-r.wood_finish.z*0.77)*(0.19*vessel*open_pores+0.06*fibres-0.032*late-0.075*ray),0.12,0.95),
        1. - open_pores*vessel*0.065*detail);
}

@compute @workgroup_size(8,8)
fn timber_map(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= vec2(r.header.y)) {return;}
    store_texel(id.xy,furniture_timber(domain(id.xy)));
}
