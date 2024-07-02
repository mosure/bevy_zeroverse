#import bevy_render::view::View


@group(0) @binding(0) var<uniform> view: View;

@group(1) @binding(0) var<storage, write> plucker_u_texture : texture_storage_2d<rgba32float, write>;
@group(1) @binding(1) var<storage, write> plucker_v_texture : texture_storage_2d<rgba32float, write>;
@group(1) @binding(2) var<storage, write> plucker_visual_texture : texture_storage_2d<rgba32float, write>;


fn get_st(global_invocation_id: vec3<u32>) -> vec2<f32> {
    return vec2<f32>(global_invocation_id.xy) / vec2<f32>(
        f32(view.viewport.z),
        f32(view.viewport.w),
    );
}

fn safe_normalize(v: vec3<f32>) -> vec3<f32> {
    return length(v) > 0.0 ? normalize(v) : vec3<f32>(0.0, 0.0, 0.0);
}

@compute @workgroup_size(16, 16)
fn plucker_kernel(@builtin(global_invocation_id) gid : vec3<u32>) {
    let st = get_st(gid);

    let width = f32(view.viewport.z);
    let height = f32(view.viewport.w);
    let wh = vec2<f32>(width, height);

    let fovy = view.exposure;
    let focal = width * 0.5 / tan(0.5 * radians(fovy));

    let f = (st.xy - 0.5) * wh;

    let camera_dir = vec3<f32>(f.xy / focal, -1.0);
    let rays_d = (view.view * vec4<f32>(camera_dir, 0.0)).xyz;
    let rays_o = view.view[3].xyz;

    let normalized_rays_d = safe_normalize(rays_d);

    let plucker_u = cross(rays_o, normalized_rays_d);
    let plucker_v = normalized_rays_d;

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
