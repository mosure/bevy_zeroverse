#import bevy_render::view::View


@group(0) @binding(0) var<storage, read_write> view: View;

@group(1) @binding(0) var plucker_u_texture : texture_storage_2d<rgba32float, write>;
@group(1) @binding(1) var plucker_v_texture : texture_storage_2d<rgba32float, write>;
@group(1) @binding(2) var plucker_visual_texture : texture_storage_2d<rgba32float, write>;


fn get_fovy() -> f32 {
    let f = view.clip_from_view[1][1];
    let fovy_radians = 2.0 * atan(1.0 / f);
    return fovy_radians;
}

fn get_st(global_invocation_id: vec3<u32>) -> vec2<f32> {
    return vec2<f32>(global_invocation_id.xy) / vec2<f32>(
        f32(view.viewport.z),
        f32(view.viewport.w),
    );
}

fn safe_normalize(v: vec3<f32>) -> vec3<f32> {
    return select(
        vec3<f32>(0.0),
        normalize(v),
        length(v) > 1e-6,
    );
}

@compute @workgroup_size(16, 16)
fn plucker_kernel(@builtin(global_invocation_id) gid : vec3<u32>) {
    let st = get_st(gid);

    let width = f32(view.viewport.z);
    let height = f32(view.viewport.w);
    let wh = vec2<f32>(width, height);

    let fovy = get_fovy();
    let focal = width * 0.5 / tan(0.5 * fovy);

    let f = (st.xy - 0.5) * wh;

    let frag_dir = vec3<f32>(f.xy / focal, -1.0);
    let rays_d = (view.view_from_world * vec4<f32>(frag_dir, 0.0)).xyz;
    let rays_o = view.view_from_world[3].xyz;

    let normalized_rays_d = safe_normalize(rays_d);

    let plucker_u = cross(rays_o, normalized_rays_d);
    let plucker_v = normalized_rays_d;

    // TODO: scale view resolution to texture resolution (e.g. workgroup -> st -> texture coords)
    textureStore(
        plucker_u_texture,
        gid.xy,
        vec4<f32>(plucker_u, 1.0),
    );
    textureStore(
        plucker_v_texture,
        gid.xy,
        vec4<f32>(plucker_v, 1.0),
    );
    textureStore(
        plucker_visual_texture,
        gid.xy,
        vec4<f32>(plucker_v * 0.5 + 0.5, 1.0),
    );
}
